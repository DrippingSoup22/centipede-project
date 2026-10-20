"""Tests for the follow-line environment and its parts: the line, the reward,
the observations, the tow and the diagnostics.

The parts are fed hand-made physical states; one test runs the whole
environment on the headless tester on the CPU.
"""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from centipede.diagnostics_category import WindowSummary
from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.environment import Environment
from centipede.environment.line import StraightLine
from centipede.environment.observation_builder import ObservationBuilder
from centipede.environment.reward_function import PART_NAMES, RewardFunction
from centipede.environment.settings import (
    EnvironmentSettings,
    RewardSettings,
    TowSettings,
)
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.tow import Tow

BODY = BodyMeasures(
    unit_weight_n=1e-3,
    leg_push_n=2e-3,
    body_length_m=0.05,
    turning_circle_m=0.03,
    segment_lengths_m=(0.01,) * 5,
    segment_weights_n=(1e-3,) * 5,
    unit_spacing_m=0.01,
)
SEGMENTS, LEGS = 5, 8
HEADLESS = {
    "model_path": str(Path("models/assembly_v5_headless.xml").resolve()),
    "backend": "cpu",
    "world_count": 2,
}


def resting_state(world_count: int) -> PhysicalState:
    """Upright units facing +x, their centres a centimetre apart along it."""
    state = PhysicalState.allocate(world_count, SEGMENTS, LEGS, "cpu")
    state.body_quaternion[..., 0] = 1.0
    state.body_planar_position[..., 0] = -0.01 * torch.arange(SEGMENTS)
    return state


def turned(angle: float) -> torch.Tensor:
    """The quaternion of a body turned ``angle`` rad to the left."""
    return torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def test_the_settings_default_to_the_agreed_design_and_ramp_the_costs():
    settings = EnvironmentSettings.from_section({"simulation": HEADLESS})
    assert settings.observation_radius == 2
    assert settings.reward.line_speed_cap_mm_per_s == 90.0
    assert settings.tow.start == 0.0  # no tow unless asked
    weights = [settings.reward.ramp_weight_at(cycle) for cycle in (1, 65, 129, 300)]
    assert weights == [0.5, 0.75, 1.0, 1.0]


def test_the_line_is_laid_at_its_turn_and_measures_each_centre_on_it():
    """World 0's line runs along +x through the units' mean centre, world 1's
    turned 0.1 rad to the left. The body steps 2 mm forward: every centre of
    world 0 goes 2 mm along its line. It moves 1.5 mm to the left: none goes
    along, every centre lies 1.5 mm from the line, unit 2, turned 30 degrees,
    at that angle."""
    line = StraightLine(2, SEGMENTS, 0.01, 0.025, "cpu")
    state = resting_state(2)
    line.restart(torch.ones(2, dtype=torch.bool), state, torch.tensor([0.0, 0.1]))
    torch.testing.assert_close(line.angle[1], torch.full((SEGMENTS,), -0.1))
    assert line.distance[1, 0].item() == pytest.approx(0.02 * math.sin(0.1))

    state.body_planar_position[..., 0] += 0.002
    progress = line.update(state)
    torch.testing.assert_close(progress[0], torch.full((SEGMENTS,), 0.002))
    torch.testing.assert_close(line.distance[0], torch.zeros(SEGMENTS))

    state.body_planar_position[..., 1] += 0.0015
    state.body_quaternion[0, 2] = turned(math.radians(30))
    progress = line.update(state)
    torch.testing.assert_close(progress[0], torch.zeros(SEGMENTS))
    torch.testing.assert_close(line.distance[0], torch.full((SEGMENTS,), 0.0015))
    assert line.angle[0].tolist() == pytest.approx([0, 0, math.radians(30), 0, 0])


def test_each_unit_sees_the_trail_within_its_sight_in_its_own_frame():
    """Trail points 6 mm apart, a sight of 13 mm. Unit 0 on the line sees all
    five straight ahead and behind; unit 2, 6 mm to the left of the line,
    sees it to its right and not the outer two points (13.4 mm away); unit
    4, turned to face the line's left, sees the points ahead on its right."""
    line = StraightLine(1, SEGMENTS, 0.006, 0.013, "cpu")
    state = resting_state(1)
    line.restart(torch.tensor([True]), state, torch.zeros(1))
    state.body_planar_position[0, 2, 1] = 0.006
    state.body_quaternion[0, 4] = turned(math.pi / 2)
    line.update(state)

    trail = line.trail[0].view(SEGMENTS, 5, 3) * torch.tensor([1e3, 1e3, 1.0])
    offsets = torch.tensor([-12.0, -6.0, 0.0, 6.0, 12.0])
    seen = torch.ones(5)
    zeros = torch.zeros(5)
    torch.testing.assert_close(trail[0], torch.stack((offsets, zeros, seen), dim=-1))
    middle = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
    torch.testing.assert_close(
        trail[2], torch.stack((offsets * middle, -6.0 * middle, middle), dim=-1)
    )
    torch.testing.assert_close(
        trail[4], torch.stack((zeros, -offsets, seen), dim=-1), atol=1e-4, rtol=0
    )


def test_the_reward_pays_the_line_and_charges_the_team_and_each_unit_its_costs():
    """Steps of 2, 0.5, 0.5, -0.5 and 0.5 mm along the line, unit 2 at 2.5 mm
    from it (closeness 0.75 in a 5 mm band), unit 4 facing 60 degrees away:
    paid 1.8 (the cap), 0.5, 0.375, -0.5 (a step back in full) and 0.25, the
    team the mean, 0.485 mm. Joint 1 is pushed together by 0.3 unit weights
    once the tows behind it are taken out, the others pulled or pushed only
    by the tows: the squeeze, 1.06 mm at full weight, over four joints, at
    the ramp's 0.5. Unit 0 is dragged by 0.3, unit 1 pushes with 0.1, units
    3 and 4 brake their own tows of 0.1: force shares of 1.06, 0.118, 0, 0.
    Unit 0 weighs 0.8 units more than the others and hangs that for free."""
    body = replace(BODY, segment_weights_n=(1.8e-3,) + (1e-3,) * (SEGMENTS - 1))
    commanded = torch.ones(SEGMENTS, LEGS + 1)
    commanded[-1, -1] = 0.0
    reward = RewardFunction(RewardSettings.from_section({}), body, commanded, 0.005)
    reward.ramp_weight = 0.5
    state = resting_state(1)
    state.motor_positive_work[0, 0] = 2e-3
    state.motor_negative_work[0, 1] = 4e-3
    command = torch.zeros(1, SEGMENTS, LEGS + 1)
    command[0, 0, 0] = 1.0
    progress = 1e-3 * torch.tensor([[2.0, 0.5, 0.5, -0.5, 0.5]])
    distance = 1e-3 * torch.tensor([[0.0, 0.0, 2.5, 0.0, 0.0]])
    angle = torch.tensor([[0.0, 0.0, 0.0, 0.0, math.radians(60)]])
    averaged = torch.zeros(1, SEGMENTS, 5)
    averaged[0, :, 0] = torch.tensor([0.3, -0.1, 0.0, -0.1, -0.1])  # along
    averaged[0, :, 1] = torch.tensor([1.0, 0.1, 0.0, 0.0, 0.0])  # up
    averaged[0, :, 2] = torch.tensor([0.0, -0.5, 0.2, -0.2, 0.0])  # front joint
    averaged[0, 2, 3] = 0.2  # belly
    averaged[0, 4, 4] = 2.0  # legs pressed
    tow = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.1]])

    rewards, parts, own_energy = reward.compute(
        state, command, progress, distance, angle, averaged, tow
    )

    by_name = dict(zip(PART_NAMES, parts[0].unbind(dim=-1), strict=True))
    every = torch.ones(SEGMENTS)
    torch.testing.assert_close(by_name["line"], 0.485 * every)
    torch.testing.assert_close(by_name["squeeze"], -0.1325 * every)
    # Energy, mJ: the units' mean at the ramp's 0.5; a motor fully on 0.332.
    torch.testing.assert_close(own_energy[0, 0], torch.tensor([2.0, 0.0, 0.332]))
    torch.testing.assert_close(by_name["pushing"], -0.2 * every)
    torch.testing.assert_close(by_name["braking"], -0.1 * every)
    torch.testing.assert_close(by_name["force"], -0.0332 * every)
    torch.testing.assert_close(
        by_name["force_share"], -torch.tensor([1.06, 1.06 / 9, 0.0, 0.0, 0.0])
    )
    # Hanging at 0.664 mJ a unit weight; pain, a leg's strongest push, 0.332.
    torch.testing.assert_close(
        by_name["hanging"], -0.664 * torch.tensor([0.2, 0.1, 0.2, 0.0, 0.0])
    )
    torch.testing.assert_close(by_name["pain"], -0.166 * torch.eye(SEGMENTS)[4])
    team = 0.485 - 0.1325 - 1.59 * (0.2 + 0.1 + 0.0332)
    torch.testing.assert_close(rewards[0, :, 0], team * every)
    torch.testing.assert_close(rewards[0, 0, 1], torch.tensor(-1.06 - 1.59 * 0.1328))


def test_a_block_holds_what_the_unit_senses_then_come_the_trail_and_the_tow():
    builder = ObservationBuilder(SEGMENTS, 2, 2, LEGS, BODY.unit_weight_n, "cpu")
    block = builder.block_size
    assert (block, builder.observation_size) == (47, 5 * 47 + 15 + 1)
    state = resting_state(2)
    state.foot_normal_force[0, 1] = torch.tensor([2e-3, 1e-3])
    state.front_joint_force[0, 1] = -0.5e-3
    state.net_joint_force[0, 1] = torch.tensor([0.2e-3, 0.7e-3])
    last_action = torch.rand(2, SEGMENTS, LEGS + 1)
    # Averaged: along, up, front joint; unit 1 pushes unit 0 with 0.5, and
    # unit 2 is pulled by unit 1 with 2.
    averaged = torch.zeros(2, SEGMENTS, 5)
    averaged[0, 1, :3] = torch.tensor([0.3, 1.0, -0.5])
    averaged[0, 2, 2] = 2.0
    trail = torch.rand(2, SEGMENTS, 15)

    observations = builder.build(
        state, last_action, averaged, trail, torch.tensor([0.0, 0.25])
    )

    own = observations[0, 1, :block]
    torch.testing.assert_close(own[:3], torch.tensor([0.0, 0.0, -1.0]))  # gravity
    torch.testing.assert_close(own[27:29], torch.tensor([2.0, 1.0]))  # feet
    # Averaged: the front joint, the net along and up; then the step's.
    torch.testing.assert_close(own[33:38], torch.tensor([-0.5, 0.3, 1.0, -0.5, 0.2]))
    torch.testing.assert_close(own[38:], last_action[0, 1])
    # Blocks: itself, the units ahead, then those behind; unit 1 sees its
    # rear joint in unit 2's block, and nothing two ahead of it.
    torch.testing.assert_close(observations[0, 1, 3 * block + 33], torch.tensor(2.0))
    assert not observations[0, 1, 2 * block : 3 * block].any()
    torch.testing.assert_close(observations[..., 5 * block : -1], trail)
    torch.testing.assert_close(
        observations[..., -1], torch.tensor([[0.0], [0.25]]).expand(2, SEGMENTS)
    )


def test_the_tow_pushes_less_as_each_unit_walks_faster():
    """A help speed of 10 mm/s and a tow of 0.4: a unit at rest is pushed
    fully, one at 5 mm/s by half, one at 10 mm/s or faster not at all, and
    one walking backward fully, along its forward axis."""
    settings = TowSettings.from_section({"help_speed_mm_per_s": 10.0})
    tow = Tow(settings, 1, (1e-3,) * SEGMENTS, 1e-3, "cpu")
    tow.start(0.4)
    tow.shares.fill_(1.0)
    speed = torch.tensor([[0.0, 0.005, 0.010, 0.020, -0.005]])
    forces = tow.forces(resting_state(1).body_quaternion, speed)
    expected = 0.4 * torch.tensor([[1.0, 0.5, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(forces[..., 0], 1e-3 * expected)
    torch.testing.assert_close(tow.pushes, expected)


def test_the_tow_fades_while_the_untowed_mean_walks_and_draws_from_a_floor():
    """Eight worlds, the last two reserved: the seventh at the top, the eighth
    untowed. Training worlds draw from half the top to the top. The untowed
    world walks, its line and lift-offs above the bars, and the top falls by
    a step; standing, it rises again, never above the start."""
    settings = TowSettings.from_section({"start": 0.4, "step": 0.1, "window_steps": 1})
    tow = Tow(settings, 8, (1e-3,) * SEGMENTS, 1e-3, "cpu")
    tow.start(0.4)
    tow.reserve(1)
    tow.restart(torch.ones(8, dtype=torch.bool))
    levels = tow.levels()
    assert ((levels[:6] >= 0.2) & (levels[:6] <= 0.4)).all()
    assert levels[6:].tolist() == [pytest.approx(0.4), 0.0]
    step = EnvironmentDiagnostics(8, SEGMENTS, 1e-3, "cpu", None).step
    tow.forces(resting_state(8).body_quaternion, torch.zeros(8, SEGMENTS))
    for walking, expected_top in ((True, 0.3), (False, 0.4), (False, 0.4)):
        step.reward_parts[7, :, PART_NAMES.index("line")] = 0.2 if walking else 0.0
        step.lift_off[7] = walking
        tow.record(step)
        tow.adjust()
        assert tow.top == pytest.approx(expected_top)
    assert tow.facts.push.item() == pytest.approx(levels[:6].mean().item())


def test_the_gait_is_read_from_the_feet_and_each_episode_is_summarised():
    """Every foot stands four steps of eight, each unit's a step behind the
    one ahead and each right foot half a cycle behind its left: a wave from
    front to rear of 45 degrees per unit, with the sides alternating. Both
    worlds go 1 mm a step along the line; world 1 ends having lost it."""
    diagnostics = EnvironmentDiagnostics(2, SEGMENTS, BODY.unit_weight_n, "cpu", None)
    window = WindowSummary(diagnostics.step)
    state = resting_state(2)
    line = StraightLine(2, SEGMENTS, 0.01, 0.025, "cpu")
    diagnostics.restart(torch.ones(2, dtype=torch.bool))
    delay = torch.arange(SEGMENTS)[:, None] + torch.tensor([0, 4])
    for step in range(40):
        standing = (step - delay) % 8 < 4
        state.foot_normal_force.copy_(standing * BODY.unit_weight_n)
        diagnostics.record_step(
            state,
            torch.ones(2, SEGMENTS, 2),
            torch.zeros(2, SEGMENTS, len(PART_NAMES)),
            torch.zeros(2, SEGMENTS, 3),
            torch.full((2, SEGMENTS), 1e-3),
            line,
            torch.zeros(2, SEGMENTS, 5),
            torch.tensor([False, step == 39]),
            torch.full((2,), step == 39),
        )
        window.add()

    gait = window.result()
    torch.testing.assert_close(gait["foot_standing"], torch.full((SEGMENTS, 2), 0.5))
    torch.testing.assert_close(
        gait["neighbour_lag"], torch.full((SEGMENTS - 1, 2), math.pi / 4)
    )
    for name in ("neighbour_lag", "left_right_lag"):
        assert torch.allclose(gait[f"{name}_consistency"], torch.tensor(1.0))
        assert torch.allclose(gait[f"{name}_lock"], torch.tensor(1.0))
    assert torch.allclose(gait["left_right_lag"].cos(), torch.tensor(-1.0))
    episode = diagnostics.episode
    assert episode.episode_ended.all() and episode.strayed.tolist() == [False, True]
    torch.testing.assert_close(episode.duration, torch.full((2,), 0.8))
    torch.testing.assert_close(episode.line_progress, torch.full((2,), 0.04))
    torch.testing.assert_close(
        episode.segment_return, torch.full((2, SEGMENTS, 2), 40.0)
    )


def test_the_headless_tester_follows_its_line_and_a_world_that_loses_it_restarts():
    """The whole environment on the headless tester: 251 values for each of
    ten agents, every unit starting within its sight of the line, the spine
    motors receiving 0.35 of their command, two accounts. World 1's line
    moved 20 mm to the side, beyond the sight radius, it restarts at once on
    a new line; world 0 at the time limit."""
    settings = EnvironmentSettings.from_section(
        {
            "max_episode_steps": 3,
            "stagger_first_episodes": False,
            "simulation": HEADLESS,
        }
    )
    environment = Environment(settings)
    observations = environment.reset(seed=0)
    assert observations.shape == (2, 10, 251)
    assert environment.sight_radius_m == pytest.approx(0.0165, abs=1e-4)
    # Units up to 30 mm from the body's middle, the line turned up to 10 deg.
    assert (environment.line.distance < 0.0055).all()
    spine_commands = []
    step = environment.simulation.step

    def recorded_step(legs, spine, body_forces):
        spine_commands.append(spine.clone())
        step(legs, spine, body_forces)

    environment.simulation.step = recorded_step
    line = environment.line
    line.origin[1] += 0.02 * torch.stack((-line.direction[1, 1], line.direction[1, 0]))
    action = torch.zeros(2, 10, 9)
    action[..., -1] = 1.0
    for expected in ([False, True], [False, False], [True, False]):
        observations, rewards, terminated, truncated, _ = environment.step(action)
        assert rewards.shape == (2, 10, 2) and torch.isfinite(rewards).all()
        assert truncated.tolist() == expected and not terminated.any()
        if expected[1]:
            assert environment.diagnostics.episode.strayed.tolist() == [False, True]
    torch.testing.assert_close(spine_commands[0], torch.full((2, 9), 0.35))
    assert (environment.line.distance[1] < 0.0055).all()
