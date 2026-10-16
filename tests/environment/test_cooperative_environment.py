"""Tests for the cooperators' environment and its task, observations, reward
and diagnostics.

The parts are fed hand-made physical states; one test runs the whole
environment on the CPU.
"""

import math
from pathlib import Path

import pytest
import torch

from centipede.diagnostics_category import WindowSummary
from centipede.environment.cooperative_diagnostics import CooperativeDiagnostics
from centipede.environment.cooperative_line import HeadLine
from centipede.environment.cooperative_observations import (
    CooperativeObservationBuilder,
)
from centipede.environment.cooperative_reward import (
    PART_NAMES,
    AveragedForces,
    CooperativeReward,
)
from centipede.environment.cooperative_task import CooperativeTask
from centipede.environment.environment import create_environment
from centipede.environment.reward_function import StepRewards
from centipede.environment.settings import CooperatorsSettings, EnvironmentSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.model_mapping import HeadOutline

SETTINGS = CooperatorsSettings.from_section({})
BODY = BodyMeasures(
    unit_weight_n=1e-3,
    leg_push_n=2e-3,
    body_length_m=0.05,
    turning_circle_m=0.03,
    segment_lengths_m=(0.01,) * 5,
)
SEGMENTS, LEGS = 5, 8


def resting_state(world_count: int) -> PhysicalState:
    """Upright segments facing +x, a centimetre apart, the head's tip at 5 mm
    and each joint halfway between two centres."""
    state = PhysicalState.allocate(world_count, SEGMENTS, LEGS, "cpu")
    state.body_quaternion[..., 0] = 1.0
    state.body_planar_position[..., 0] = -0.01 * torch.arange(SEGMENTS)
    state.head_tip_position[:, 0] = 0.005
    state.front_end_position[..., 0] = 0.005 - 0.01 * torch.arange(SEGMENTS)
    return state


def test_targets_follow_the_level_and_the_stopwatch_moves_it():
    task = CooperativeTask(SETTINGS, BODY, 4096, episode_steps=400, device="cpu")
    state = resting_state(4096)
    every_world = torch.ones(4096, dtype=torch.bool)

    # The circle opens where the near end clears a body length and a turn.
    final_near = 2.7 * BODY.body_length_m
    assert task.full_circle_level == pytest.approx((0.08 - 0.02) / (final_near - 0.02))
    for level, near, half_bearing in ((0.0, 0.02, 90.0), (1.0, final_near, 180.0)):
        task.level = level
        task.place_targets(every_world, state, torch.full((4096,), 400))
        offset = task.target_position - state.head_tip_position[:, :2]
        distance = offset.norm(dim=-1)
        bearing = torch.rad2deg(torch.atan2(offset[:, 1], offset[:, 0])).abs()
        assert distance.min() >= near and distance.max() <= 1.5 * near
        assert bearing.max() <= half_bearing + 1e-3
        assert bearing.max() > 0.95 * half_bearing

    # Over one episode length, a third of the timed targets reached within
    # their stopwatch of 350 steps: the level rises by 0.02 x (1/3 - 0.10).
    # The last quarter of the worlds restart in 100 steps, too soon for a
    # stopwatch to run out: their targets are not counted.
    task.level = 0.5
    worlds = torch.arange(4096)
    task.place_targets(every_world, state, torch.where(worlds < 3072, 400, 100))
    reached = worlds < 1024
    for step in range(400):
        arrived = reached & (step == 100)
        task.time_step(arrived)
        # As the environment walks on: these targets come too late to count.
        task.place_targets(arrived, state, torch.full((4096,), 399 - step))
    task.adjust_level()
    assert task.in_time_share == pytest.approx(1 / 3)
    assert task.level == pytest.approx(0.5 + 0.02 * (1 / 3 - 0.1))


def test_the_goal_pays_the_line_erased_and_charges_the_tail_off_it_and_the_joint():
    """The head steps 1 mm toward its target and onto it; every tail erases
    1 mm of line. The followers' tails lie 0, 1.5, 3 and 12 mm from the line
    they have not erased, the last counted as 10. Averaged forces, in unit
    weights: segment 1's front joint pushed with 0.3, segment 3's pulled with
    0.15; the head held up by 0.8 and on the floor with 0.2, segment 2 held
    up by 1, the rear on the floor with 0.3. A squared cost takes 40 mm/s,
    0.8 mm a step, at its scale: 3 mm off the line, or 0.3 unit weights.
    Energy is shared 40% to itself and 30% to each side; pain stays with the
    pressed."""
    reward = CooperativeReward(
        SETTINGS, SEGMENTS, 2, BODY, torch.ones(SEGMENTS, LEGS + 1)
    )
    torch.testing.assert_close(
        reward.sharing_weights[:3],
        torch.tensor(
            [
                [0.4, 0.4, 0.2, 0.0, 0.0],
                [0.3, 0.4, 0.2, 0.1, 0.0],
                [0.1, 0.2, 0.4, 0.2, 0.1],
            ]
        ),
    )
    state = resting_state(1)
    state.motor_positive_work[0, 2] = 1e-6  # one microjoule pushing
    state.leg_leg_force_squared[0, 3] = BODY.leg_push_n**2  # a full push
    state.net_joint_force[0] = 1e-3 * torch.tensor(
        [[0.0, 0.8], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, -0.4]]
    )
    state.front_joint_force[0] = 1e-3 * torch.tensor([0.0, -0.3, 0.0, 0.15, 0.0])
    state.belly_normal_force[0] = 1e-3 * torch.tensor([0.2, 0.0, 0.0, 0.0, 0.3])
    forces = AveragedForces(1, SEGMENTS, BODY.unit_weight_n, 0.5, "cpu")
    forces.restart(torch.tensor([True]), state)
    head_tip = state.head_tip_position[:, :2]
    zeros = torch.zeros(1, SEGMENTS, LEGS + 1)

    channels, parts, _ = reward.compute(
        state,
        state.body_planar_position - torch.tensor([0.001, 0.0]),
        head_tip - torch.tensor([[0.001, 0.0]]),
        torch.tensor([[0.1, 0.0]]),
        torch.tensor([True]),
        0.025,
        zeros,
        zeros,
        torch.full((1, SEGMENTS), 0.001),
        torch.tensor([[0.005, 0.0, 0.0015, 0.003, 0.012]]),
        forces.values,
    )

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    torch.testing.assert_close(by_name["progress"], torch.tensor([1.0, 0, 0, 0, 0]))
    torch.testing.assert_close(by_name["arrival"], torch.tensor([25.0, 0, 0, 0, 0]))
    torch.testing.assert_close(by_name["line"], torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(
        by_name["off_line"], -0.8 * torch.tensor([0.0, 0.0, 0.25, 1.0, 100 / 9])
    )
    torch.testing.assert_close(
        by_name["front_joint"], -0.8 * torch.tensor([0.0, 1.0, 0.0, 0.25, 0.0])
    )
    one_motor = SETTINGS.force_price_w * 0.02 * 1e3
    torch.testing.assert_close(
        by_name["hanging"], -one_motor * torch.tensor([0.3 + 0.2, 0.0, 1.0, 0.0, 0.3])
    )
    torch.testing.assert_close(by_name["pushing"], -1e-3 * reward.sharing_weights[:, 2])
    # A full push for one step, priced as one motor fully on.
    torch.testing.assert_close(by_name["pain"], torch.tensor([0, 0, 0, -one_motor, 0]))
    goal = sum(by_name[name] for name in PART_NAMES[:5])
    torch.testing.assert_close(
        channels[0], torch.stack((goal, parts.rewards[0] - goal), -1)
    )

    state.net_joint_force[0, 2, 1] = 2e-3
    assert forces.update(state)[0, 2, 1] == pytest.approx(1.5)


def test_the_head_line_is_erased_by_each_tail_and_measures_how_far_it_is_left():
    """The resting body's line runs from the rear's tail, 45 mm behind the
    origin, to the tip at 5 mm. The body steps 3 mm forward: every tail
    erases 3 mm. It steps 2 mm back: nothing is erased and every tail is 2 mm
    behind the end of its line. It moves 1.5 mm to the left: every tail lies
    2.5 mm from the end of its line and 1.5 mm left of the line, its body
    along it."""
    line = HeadLine(1, BODY.segment_lengths_m, episode_steps=10, device="cpu")
    state = resting_state(1)
    line.restart(torch.tensor([True]), state)
    torch.testing.assert_close(line.arc[0, line.count[0] - 1], torch.tensor(0.05))

    def move(dx: float, dy: float) -> tuple[torch.Tensor, torch.Tensor]:
        state.front_end_position[0] += torch.tensor([dx, dy])
        erased, off_line = line.update(state)
        return erased[0], off_line[0]

    erased, off_line = move(0.003, 0.0)
    torch.testing.assert_close(erased, torch.full((SEGMENTS,), 0.003))
    torch.testing.assert_close(off_line, torch.zeros(SEGMENTS), atol=1e-7, rtol=0)
    erased, off_line = move(-0.002, 0.0)
    torch.testing.assert_close(erased, torch.zeros(SEGMENTS))
    torch.testing.assert_close(off_line, torch.full((SEGMENTS,), 0.002))
    erased, off_line = move(0.0, 0.0015)
    torch.testing.assert_close(off_line, torch.full((SEGMENTS,), 0.0025))
    torch.testing.assert_close(
        line.view[0], torch.tensor([[0.0015, 0.0, 0.0025]] * SEGMENTS)
    )


def test_a_block_holds_what_the_segment_senses_and_only_the_head_sees_the_target():
    builder = CooperativeObservationBuilder(
        SEGMENTS, 2, 1, LEGS, BODY.unit_weight_n, "cpu"
    )
    block = builder.block_size
    assert (block, builder.observation_size) == (48, 5 * 48 + 2)
    state = resting_state(1)
    state.foot_normal_force[0, 1] = torch.tensor([2e-3, 1e-3])
    last_action = torch.rand(1, SEGMENTS, LEGS + 1)
    # Averaged forces: along, up, front joint, floor; segment 1 pushes the
    # head with 0.5 and pulls segment 2 with 2.
    averaged = torch.zeros(1, SEGMENTS, 4)
    averaged[0, :3, 2] = torch.tensor([0.0, -0.5, 2.0])
    averaged[0, 1, :2] = torch.tensor([0.3, 1.0])
    line_view = torch.zeros(1, SEGMENTS, 3)
    line_view[0, 1] = torch.tensor([0.001, 0.1, 0.002])

    observations = builder.build(
        state, torch.tensor([[0.025, 0.01]]), last_action, line_view, averaged
    )

    own = observations[0, 1, :block]
    torch.testing.assert_close(own[:3], torch.tensor([0.0, 0.0, -1.0]))  # gravity
    torch.testing.assert_close(own[27:29], torch.tensor([2.0, 1.0]))  # unit weights
    # Front joint, net along and up, then its place on the head's line.
    torch.testing.assert_close(
        own[33:39], torch.tensor([-0.5, 0.3, 1.0, 0.001, 0.1, 0.002])
    )
    torch.testing.assert_close(own[-(LEGS + 1) :], last_action[0, 1])
    # Blocks: itself, the segments ahead, those behind; segment 1 sees its
    # rear joint in segment 2's block.
    torch.testing.assert_close(observations[0, 1, 3 * block + 33], torch.tensor(2.0))
    # The head sees nothing ahead of it, and the target from its tip.
    head = observations[0, 0]
    assert not head[block : 3 * block].any()
    torch.testing.assert_close(head[-2:], torch.tensor([0.02, 0.01]))
    assert not observations[0, 1:, -2:].any()


def test_the_gait_is_read_from_the_feet_and_each_episode_is_summarised():
    """Every foot stands four steps of eight, each segment's a step behind the
    one ahead and each right foot half a cycle behind its left: a wave from
    head to tail of 45 degrees per segment, with the sides alternating."""
    diagnostics = CooperativeDiagnostics(
        2, SEGMENTS, BODY.unit_weight_n, HeadOutline(-0.002, 0.002, 0.002), "cpu", None
    )
    window = WindowSummary(diagnostics.step)
    state = resting_state(2)
    target = torch.tensor([[0.1, 0.0], [0.1, 0.0]])
    every_world = torch.ones(2, dtype=torch.bool)
    diagnostics.new_targets(every_world, every_world, state, target)
    zeros = torch.zeros(2, SEGMENTS)
    step_rewards = StepRewards(
        zeros, torch.zeros(2, SEGMENTS, len(PART_NAMES)), zeros, zeros
    )
    delay = torch.arange(SEGMENTS)[:, None] + torch.tensor([0, 4])
    for step in range(40):
        standing = (step - delay) % 8 < 4
        state.foot_normal_force.copy_(standing * BODY.unit_weight_n)
        arrived = torch.tensor([step % 10 == 9, False])  # world 0, every 0.2 s
        diagnostics.record_step(
            state,
            state.body_planar_position,
            state.head_tip_position[:, :2],
            target,
            torch.zeros(2, SEGMENTS, 2),
            step_rewards,
            torch.zeros(2, SEGMENTS, 3),
            zeros,
            torch.zeros(2, SEGMENTS, 4),
            arrived,
            arrived,
            torch.full((2,), step == 39),  # the time limit
        )
        window.add()
        diagnostics.new_targets(arrived, torch.zeros_like(arrived), state, target)

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
    assert episode.episode_ended.all()
    assert episode.targets_reached.tolist() == [4.0, 0.0]
    assert episode.targets_by_bearing[:, 0].tolist() == [4.0, 1.0]  # all ahead
    assert episode.reached_by_bearing[:, 0].tolist() == [4.0, 0.0]


def test_after_an_arrival_the_body_walks_on_and_restarts_only_at_the_time_limit():
    """The whole environment on v4: two channels, both for every segment, the
    spine motors receiving 0.35 of their command."""
    settings = EnvironmentSettings.from_section(
        {
            "design": "cooperators",
            "max_episode_steps": 3,
            "observation_radius": 2,
            "simulation": {
                "model_path": str(Path("models/assembly_v4.xml").resolve()),
                "backend": "cpu",
                "world_count": 2,
            },
            "cooperators": {"stagger_first_episodes": False},
        }
    )
    environment = create_environment(settings)
    environment.reset(seed=0)
    spine_commands = []
    step = environment.simulation.step

    def recorded_step(legs, spine):
        spine_commands.append(spine.clone())
        step(legs, spine)

    environment.simulation.step = recorded_step
    # World 0's target lies under its head.
    state = environment.simulation.physical_state
    environment.task.target_position[0] = state.body_planar_position[0, 0]
    action = torch.zeros(2, 12, 9)
    action[..., -1] = 1.0

    observations, rewards, terminated, truncated, final = environment.step(action)

    assert rewards.shape == (2, 12, 2) and torch.isfinite(rewards).all()
    assert environment.segment_reward_channels == [(0, 1)] * 12
    assert observations.shape == (2, 12, 5 * 48 + 2)
    torch.testing.assert_close(spine_commands[0], torch.full((2, 11), 0.35))
    assert not terminated.any() and not truncated.any()
    assert rewards[0, 0, 0] > 1.0 and environment.task.target_steps.tolist() == [0, 1]
    assert observations is final
    distance = environment.task.target_position[0] - state.head_tip_position[0, :2]
    assert distance.norm() >= 0.02

    environment.step(action)
    _, _, terminated, truncated, final = environment.step(action)
    assert truncated.all() and not terminated.any()
    assert environment.episode_steps.tolist() == [0, 0]
    assert not torch.equal(final, environment._observations())
