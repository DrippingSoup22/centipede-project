"""Tests for the cooperators' environment and its task, observations, reward,
tow and diagnostics.

The parts are fed hand-made physical states; two tests run the whole
environment on the CPU.
"""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from centipede.diagnostics_category import WindowSummary
from centipede.environment.cooperative_diagnostics import CooperativeDiagnostics
from centipede.environment.cooperative_line import HeadLine, StraightLine
from centipede.environment.cooperative_observations import (
    CooperativeObservationBuilder,
)
from centipede.environment.cooperative_reward import (
    GOAL_PARTS,
    PART_NAMES,
    AveragedForces,
    CooperativeReward,
)
from centipede.environment.cooperative_task import CooperativeTask
from centipede.environment.cooperative_tow import Tow
from centipede.environment.environment import create_environment
from centipede.environment.reward_function import StepRewards
from centipede.environment.settings import CooperatorsSettings, EnvironmentSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.model_mapping import HeadOutline
from centipede.settings_section import SettingsError

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

    # Cycle 17: the targets kept within 15 degrees of straight ahead.
    ahead = CooperativeTask(
        replace(SETTINGS, bearing_limit_deg=15.0), BODY, 4096, 400, "cpu"
    )
    ahead.level = 1.0
    ahead.place_targets(every_world, state, torch.full((4096,), 400))
    offset = ahead.target_position - state.head_tip_position[:, :2]
    bearing = torch.rad2deg(torch.atan2(offset[:, 1], offset[:, 0])).abs()
    assert 0.95 * 15.0 < bearing.max() <= 15.0 + 1e-3

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


def goal_step() -> tuple[PhysicalState, AveragedForces, tuple]:
    """A step to price: the head steps 1 mm toward its target and onto it;
    every tail erases 1 mm of line. The followers' tails lie 0, 1.5, 3 and
    12 mm from the line they have not erased, the last counted as 10.
    Averaged forces, in unit weights: segment 1's front joint pushed with
    0.3, segment 3's pulled with 0.15; the head held up by 0.8 and on the
    floor with 0.2, segment 2 held up by 1, the rear on the floor with 0.3.
    Segment 2 pushes with a microjoule of work, segment 3's legs are pressed
    with a full push. Returns the state, the averages and ``compute``'s
    arguments."""
    state = resting_state(1)
    state.motor_positive_work[0, 2] = 1e-6
    state.leg_leg_force_squared[0, 3] = BODY.leg_push_n**2
    state.net_joint_force[0] = 1e-3 * torch.tensor(
        [[0.0, 0.8], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, -0.4]]
    )
    state.front_joint_force[0] = 1e-3 * torch.tensor([0.0, -0.3, 0.0, 0.15, 0.0])
    state.belly_normal_force[0] = 1e-3 * torch.tensor([0.2, 0.0, 0.0, 0.0, 0.3])
    forces = AveragedForces(1, SEGMENTS, BODY.unit_weight_n, 0.5, "cpu")
    forces.restart(torch.tensor([True]), state)
    head_tip = state.head_tip_position[:, :2]
    zeros = torch.zeros(1, SEGMENTS, LEGS + 1)
    arguments = (
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
        torch.zeros(1, SEGMENTS),
    )
    return state, forces, arguments


def test_the_goal_pays_the_line_erased_and_charges_the_tail_off_it_and_the_joint():
    """goal_step priced as built. A squared cost takes 40 mm/s, 0.8 mm a
    step, at its scale: 3 mm off the line, or 0.3 unit weights. Energy is
    shared 40% to itself and 30% to each side; pain stays with the
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
    state, forces, arguments = goal_step()

    channels, parts, _ = reward.compute(*arguments)

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
    assert not by_name["own_force"].any()
    goal = sum(by_name[name] for name in PART_NAMES[:GOAL_PARTS])
    torch.testing.assert_close(
        channels[0], torch.stack((goal, parts.rewards[0] - goal), -1)
    )

    state.net_joint_force[0, 2, 1] = 2e-3
    assert forces.update(state)[0, 2, 1] == pytest.approx(1.5)


def test_the_shared_reward_pays_followers_the_mean_goal_and_shares_pain_and_hanging():
    """goal_step priced as built and shared (cycle 13): the head keeps its
    goal, every follower is paid the followers' mean of each goal part;
    energy is shared as before, pain, at the ramp's half weight, and hanging
    as energy is. The ramp over 128 cycles: 0, 0.5, 1 at cycles 1, 65, 129."""
    commanded = torch.ones(SEGMENTS, LEGS + 1)
    built = CooperativeReward(SETTINGS, SEGMENTS, 2, BODY, commanded)
    settings = CooperatorsSettings.from_section(
        {"goal_team_share": 1.0, "health_sharing": "all", "pain_ramp_cycles": 128}
    )
    shared = CooperativeReward(settings, SEGMENTS, 2, BODY, commanded)
    shared.pain_weight = settings.pain_weight_at(65)

    own = built.compute(*goal_step()[2])[1].reward_parts[0]
    paid = shared.compute(*goal_step()[2])[1].reward_parts[0]

    torch.testing.assert_close(paid[0, :GOAL_PARTS], own[0, :GOAL_PARTS])
    torch.testing.assert_close(
        paid[1:, :GOAL_PARTS], own[1:, :GOAL_PARTS].mean(0).expand(SEGMENTS - 1, -1)
    )
    energy = slice(PART_NAMES.index("pushing"), PART_NAMES.index("pain"))
    torch.testing.assert_close(paid[:, energy], own[:, energy])
    weights = built.sharing_weights
    pain, hanging = PART_NAMES.index("pain"), PART_NAMES.index("hanging")
    torch.testing.assert_close(paid[:, pain], 0.5 * weights @ own[:, pain])
    torch.testing.assert_close(paid[:, hanging], weights @ own[:, hanging])
    assert [settings.pain_weight_at(c) for c in (1, 65, 129, 200)] == [0, 0.5, 1, 1]
    assert SETTINGS.pain_weight_at(1) == 1.0


def test_the_team_and_own_channels_share_the_line_and_energy_and_keep_the_duties():
    """goal_step priced by owner (cycle 14): the head keeps its target, every
    follower is paid the followers' mean line and distance, every segment
    the body's mean energy, all in the team channel; each keeps its force
    costs, its pain, at the ramp's weight, and its hanging in its own. A
    channel counts a millijoule as 10 mm. The ramp from half over 128
    cycles: 0.5, 0.75, 1 at cycles 1, 65, 129."""
    commanded = torch.ones(SEGMENTS, LEGS + 1)
    built = CooperativeReward(SETTINGS, SEGMENTS, 2, BODY, commanded)
    settings = CooperatorsSettings.from_section(
        {"reward_channels": "team_own", "pain_ramp_cycles": 128, "pain_ramp_start": 0.5}
    )
    split = CooperativeReward(settings, SEGMENTS, 2, BODY, commanded)
    split.pain_weight = settings.pain_weight_at(65)

    own = built.compute(*goal_step()[2])[1].reward_parts[0]
    channels, parts, own_energy = split.compute(*goal_step()[2])

    paid = parts.reward_parts[0]
    index = {name: i for i, name in enumerate(PART_NAMES)}
    line, forces = slice(0, 5), slice(5, GOAL_PARTS)
    energy, harm = slice(GOAL_PARTS, index["pain"]), slice(index["pain"], None)
    torch.testing.assert_close(paid[0, line], own[0, line])
    torch.testing.assert_close(
        paid[1:, line], own[1:, line].mean(0).expand(SEGMENTS - 1, -1)
    )
    torch.testing.assert_close(paid[:, forces], own[:, forces])
    torch.testing.assert_close(
        paid[:, energy], -own_energy[0].mean(0).expand(SEGMENTS, -1)
    )
    torch.testing.assert_close(paid[:, harm], own[:, harm] * torch.tensor([0.75, 1]))
    team = paid[:, line].sum(-1) + 10 * paid[:, energy].sum(-1)
    kept = paid[:, forces].sum(-1) + 10 * paid[:, harm].sum(-1)
    torch.testing.assert_close(channels[0], torch.stack((team, kept), -1))
    assert [settings.pain_weight_at(c) for c in (1, 65, 129)] == [0.5, 0.75, 1]
    with pytest.raises(SettingsError, match="team_own"):
        CooperatorsSettings.from_section(
            {"reward_channels": "team_own", "goal_team_share": 1.0}
        )


def test_cycle_15_pays_the_line_near_it_and_charges_every_follower_the_squeeze():
    """goal_step priced as cycle 15: each follower's 1 mm erased paid by its
    closeness at 5 mm, 1, 0.91, 0.64 and 0 at 0, 1.5, 3 and 12 mm, the
    followers' mean 0.6375, and no distance cost. The squeeze: segment 1's
    front joint pushed with 0.3 unit weights, a full price of 0.8 mm a step
    over four joints, 0.2 for every follower; under a tow of 0.1 per segment
    the legs pull at that joint (0.4 behind it added back), so nothing.
    Ramped, energy and the squeeze are weighed as pain is."""
    settings = CooperatorsSettings.from_section(
        {
            "reward_channels": "team_own",
            "line_closeness_mm": 5.0,
            "squeeze_weight": 1.0,
            "ramp_team_costs": True,
        }
    )
    reward = CooperativeReward(settings, SEGMENTS, 2, BODY, torch.ones(SEGMENTS, 9))
    arguments = goal_step()[2]

    _, parts, own_energy = reward.compute(*arguments)

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    followers = torch.tensor([0.0, 1, 1, 1, 1])
    torch.testing.assert_close(by_name["line"], 0.6375 * followers)
    assert not by_name["off_line"].any()
    torch.testing.assert_close(by_name["squeeze"], -0.2 * followers)
    towed = reward.compute(*arguments[:-1], torch.full((1, SEGMENTS), 0.1))[1]
    assert not towed.reward_parts[0, :, PART_NAMES.index("squeeze")].any()
    reward.pain_weight = 0.5
    halved = reward.compute(*arguments)[1].reward_parts[0]
    energy = slice(PART_NAMES.index("pushing"), PART_NAMES.index("pain"))
    torch.testing.assert_close(halved[:, energy], parts.reward_parts[0, :, energy] / 2)
    torch.testing.assert_close(halved[:, PART_NAMES.index("squeeze")], -0.1 * followers)
    torch.testing.assert_close(
        parts.reward_parts[0, :, energy], -own_energy[0].mean(0).expand(SEGMENTS, -1)
    )


def test_cycle_16_shares_energy_by_neighbourhood_in_the_own_channel_and_averages_pain():
    """goal_step priced as cycle 16: segment 2's microjoule of pushing work
    shared 40% to itself, 25% to each nearest neighbour and 5% to each next
    one, the ends taking a missing side's shares (the head 10%: 5% from each
    side), all in the own channel. Pain averaged: segment 3's legs pressed
    with a full push for good cost what the instant press did; a press that
    only began this step, averaged at 0.02 a step, costs 0.02 squared of it."""
    settings = CooperatorsSettings.from_section(
        {
            "reward_channels": "team_own",
            "energy_side_shares": [0.25, 0.05],
            "pain_averaged": True,
        }
    )
    reward = CooperativeReward(settings, SEGMENTS, 2, BODY, torch.ones(SEGMENTS, 9))
    state, forces, arguments = goal_step()
    state.leg_leg_force[0, 3] = BODY.leg_push_n
    forces.restart(torch.tensor([True]), state)

    channels, parts, _ = reward.compute(*arguments)

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    shares = torch.tensor([0.1, 0.25, 0.4, 0.25, 0.1])
    torch.testing.assert_close(by_name["pushing"], -1e-3 * shares)
    one_motor = SETTINGS.force_price_w * 0.02 * 1e3
    torch.testing.assert_close(by_name["pain"], torch.tensor([0, 0, 0, -one_motor, 0]))
    team = sum(by_name[name] for name in PART_NAMES[:5])
    torch.testing.assert_close(channels[0, :, 0], team)

    brief = AveragedForces(1, SEGMENTS, BODY.unit_weight_n, 0.02, "cpu")
    pressed = state.leg_leg_force.clone()
    state.leg_leg_force.zero_()
    brief.restart(torch.tensor([True]), state)
    state.leg_leg_force.copy_(pressed)
    brushed = reward.compute(*arguments[:-2], brief.update(state), arguments[-1])[1]
    torch.testing.assert_close(
        brushed.reward_parts[0, 3, PART_NAMES.index("pain")],
        torch.tensor(-(0.02**2) * one_motor),
    )


def test_the_line_drifts_where_the_head_keeps_pointing_and_not_where_it_goes():
    """The body walks along +x at 20 mm/s with the head turned 10° to the
    left: its sideslip is 10° and, leaking over 1 s, the drift settles at
    20 mm/s x sin 10° x 1 s, 3.47 mm, the tip drawing that far to the left
    of the body's travel. Pointing where it goes, the line stays at the
    tip."""
    angle = math.radians(10.0)
    for turned, drift in ((angle, 0.02 * math.sin(angle)), (0.0, 0.0)):
        state = resting_state(1)
        state.body_quaternion[0, 0] = torch.tensor(
            [math.cos(turned / 2), 0.0, 0.0, math.sin(turned / 2)]
        )
        line = HeadLine(1, BODY.segment_lengths_m, 400, "cpu", drift_leak_s=1.0)
        line.restart(torch.tensor([True]), state)
        for _ in range(400):
            state.front_end_position[..., 0] += 0.0004
            state.body_planar_position[..., 0] += 0.0004
            line.update(state)
        torch.testing.assert_close(line.sideslip, torch.tensor([turned]))
        torch.testing.assert_close(
            line.drift, torch.tensor([drift]), rtol=1e-3, atol=1e-7
        )
        torch.testing.assert_close(
            line.points[0, line.count[0] - 1],
            torch.stack((state.front_end_position[0, 0, 0], torch.tensor(drift))),
            rtol=1e-3,
            atol=1e-6,
        )


@pytest.mark.parametrize(
    ("force_price", "squared_shares"),
    [
        ("own_ground", [0.0, 1.0, 1 / 9, 0.0, 1.0]),
        ("own_joint", [0.0, 1.0, 1 / 9, 1.0, 0.0]),
    ],
)
def test_the_own_prices_charge_each_follower_its_joints_net_force_with_or_without_tow(
    force_price, squared_shares
):
    """Averaged net joint forces along the body, in unit weights: segment 1
    dragged by 0.3, segment 2 pushing with 0.1, segment 3 held back by its
    neighbours against a tow of 0.3 (its feet doing nothing, so it passes
    the tow on), segment 4 braking its own tow of 0.3. With the tow added
    back: 0.3, -0.1, 0 and 0.3; without: 0.3, -0.1, -0.3 and 0; each priced
    0.8 mm a step at 0.3; the front joint is free."""
    settings = CooperatorsSettings.from_section({"force_price": force_price})
    reward = CooperativeReward(
        settings, SEGMENTS, 2, BODY, torch.ones(SEGMENTS, LEGS + 1)
    )
    state = resting_state(1)
    averaged = torch.zeros(1, SEGMENTS, 5)
    averaged[0, :, 0] = torch.tensor([0.5, 0.3, -0.1, -0.3, 0.0])
    averaged[0, :, 2] = 1.0  # a front joint force that no longer costs
    tow = torch.tensor([[0.0, 0.0, 0.0, 0.3, 0.3]])
    zeros = torch.zeros(1, SEGMENTS, LEGS + 1)

    _, parts, _ = reward.compute(
        state,
        state.body_planar_position,
        state.head_tip_position[:, :2],
        torch.tensor([[0.1, 0.0]]),
        torch.tensor([False]),
        0.025,
        zeros,
        zeros,
        torch.zeros(1, SEGMENTS),
        torch.zeros(1, SEGMENTS),
        averaged,
        tow,
    )

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    torch.testing.assert_close(
        by_name["own_force"], -0.8 * torch.tensor(squared_shares)
    )
    assert not by_name["front_joint"].any()


def test_the_tow_pushes_along_the_floor_and_fades_while_the_towed_mean_walks():
    """Four worlds, the last two reserved: the third always at the top, the
    fourth never towed. Each segment is pushed by its world's level times its
    own weight, along its forward axis laid flat. Over each window of two
    steps the towed worlds' followers erase 0.2 mm a step (10 mm/s) and
    their feet lift off every other step (25 Hz): the top falls by a step;
    standing still, it rises again, never above the start."""
    settings = CooperatorsSettings.from_section(
        {"tow_start": 0.4, "tow_step": 0.1, "tow_window_steps": 2}
    )
    weights_n = (2e-3,) + (1e-3,) * (SEGMENTS - 1)
    tow = Tow(settings, 4, weights_n, BODY.unit_weight_n, 1000, "cpu")
    tow.start(0.4)
    tow.reserve(1)
    tow.restart(torch.ones(4, dtype=torch.bool))
    levels = tow.levels()
    assert 0 <= levels[0] <= 0.4 and 0 <= levels[1] <= 0.4
    assert levels[2:].tolist() == [pytest.approx(0.4), 0.0]

    # Each segment turned 90 degrees to the left and pitched nose down.
    state = resting_state(4)
    turn = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    pitch = torch.tensor([math.cos(0.1), 0.0, math.sin(0.1), 0.0])
    w1, x1, y1, z1 = turn
    w2, x2, y2, z2 = pitch
    state.body_quaternion[:] = torch.tensor(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    forces = tow.forces(state.body_quaternion, torch.zeros(4, SEGMENTS))
    torch.testing.assert_close(forces[2, 0], torch.tensor([0.0, 0.4 * 2e-3, 0.0]))
    torch.testing.assert_close(forces[2, 1], torch.tensor([0.0, 0.4 * 1e-3, 0.0]))
    assert not forces[3].any()
    torch.testing.assert_close(tow.unit_weights()[2, 1], torch.tensor(0.4))

    step = CooperativeDiagnostics(
        4, SEGMENTS, BODY.unit_weight_n, HeadOutline(-0.002, 0.002, 0.002), "cpu", None
    ).step
    for walking, expected_top in ((True, 0.3), (False, 0.4), (False, 0.4)):
        step.reward_parts[..., PART_NAMES.index("line")] = 0.2 if walking else 0.0
        for index in range(2):
            step.lift_off.fill_(walking and index == 0)
            tow.record(step)
            tow.adjust()
        assert tow.top == pytest.approx(expected_top)
    assert tow.facts.line_erased.tolist() == [0.0, 0.0]


def test_the_speed_dependent_tow_pushes_less_as_each_segment_walks_faster():
    """A help speed of 10 mm/s and a tow of 0.4: a segment at rest is pushed
    fully, one at 5 mm/s by half, one at 10 mm/s or faster not at all, and
    one walking backward fully; the price adds back what was pushed."""
    settings = CooperatorsSettings.from_section(
        {"tow_start": 0.4, "tow_help_speed_mm_per_s": 10.0}
    )
    tow = Tow(settings, 1, (1e-3,) * SEGMENTS, 1e-3, 1000, "cpu")
    tow.start(0.4)
    tow.shares.fill_(1.0)
    speed = torch.tensor([[0.0, 0.005, 0.010, 0.020, -0.005]])
    forces = tow.forces(resting_state(1).body_quaternion, speed)
    expected = 0.4 * torch.tensor([[1.0, 0.5, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(forces[..., 0], 1e-3 * expected)
    torch.testing.assert_close(tow.unit_weights(), expected)


def test_the_tow_can_fade_on_the_untowed_walk_and_draw_from_a_floor():
    """Fading on the untowed mean-action worlds: their followers erase line
    and step above the bars, the towed ones stand, and the top falls. Training
    worlds draw from half the top to the top, and at rest are pushed by their
    level, the mean the tow logs."""
    settings = CooperatorsSettings.from_section(
        {
            "tow_start": 0.4,
            "tow_step": 0.1,
            "tow_window_steps": 1,
            "tow_fade_group": "untowed",
            "tow_draw_floor": 0.5,
        }
    )
    tow = Tow(settings, 8, (1e-3,) * SEGMENTS, 1e-3, 1000, "cpu")
    tow.start(0.4)
    tow.reserve(1)
    tow.restart(torch.ones(8, dtype=torch.bool))
    assert ((tow.levels()[:6] >= 0.2) & (tow.levels()[:6] <= 0.4)).all()
    step = CooperativeDiagnostics(
        8, SEGMENTS, BODY.unit_weight_n, HeadOutline(-0.002, 0.002, 0.002), "cpu", None
    ).step
    step.reward_parts[7, :, PART_NAMES.index("line")] = 0.2
    step.lift_off[7] = True
    levels = tow.levels()
    tow.forces(resting_state(8).body_quaternion, torch.zeros(8, SEGMENTS))
    tow.record(step)
    tow.adjust()
    assert tow.top == pytest.approx(0.3)
    assert tow.facts.push.item() == pytest.approx(levels[:6].mean().item())


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


def test_the_straight_line_measures_each_centre_along_it_beside_it_and_its_angle():
    """The resting body's line runs along +x through its centres. The body
    steps 2 mm forward: every centre goes 2 mm along it. It moves 1.5 mm to
    the left: none goes along it, and every centre lies 1.5 mm left of it;
    segment 2 turned 30 degrees left lies at that angle."""
    line = StraightLine(1, SEGMENTS, "cpu")
    state = resting_state(1)
    line.restart(torch.tensor([True]), state)

    state.body_planar_position[0, :, 0] += 0.002
    progress, off_line = line.update(state)
    torch.testing.assert_close(progress[0], torch.full((SEGMENTS,), 0.002))
    torch.testing.assert_close(off_line[0], torch.zeros(SEGMENTS))

    state.body_planar_position[0, :, 1] += 0.0015
    half_turn = math.radians(30) / 2
    state.body_quaternion[0, 2] = torch.tensor(
        [math.cos(half_turn), 0, 0, math.sin(half_turn)]
    )
    progress, off_line = line.update(state)
    torch.testing.assert_close(progress[0], torch.zeros(SEGMENTS))
    torch.testing.assert_close(off_line[0], torch.full((SEGMENTS,), 0.0015))
    angles = torch.zeros(SEGMENTS)
    angles[2] = math.radians(30)
    torch.testing.assert_close(line.view[0, :, 1], angles)


def test_the_line_task_pays_steps_along_it_up_to_the_cap_near_it_and_facing_it():
    """goal_step on the straight line (cycle 18), the cap 1 body length a
    second, 1 mm a step: the centres go 2, 0.5, 0.5, -0.5 and 0.5 mm along
    the line; segment 2 lies 2.5 mm from it (closeness 0.75 at 5 mm),
    segment 4 faces 60 degrees away from it. Paid 1, 0.5, 0.375, -0.5 (a
    step back in full) and 0.25: every segment, the head too, the mean
    0.325 mm, and the squeeze; no target. The first segment weighs 2 units
    against the others' 1.2: it hangs 0.8 for free."""
    settings = CooperatorsSettings.from_section(
        {
            "task": "line",
            "reward_channels": "team_own",
            "line_closeness_mm": 5.0,
            "line_speed_cap_body_lengths_per_s": 1.0,
            "squeeze_weight": 1.0,
        }
    )
    body = replace(BODY, segment_weights_n=(2e-3,) + (1.2e-3,) * (SEGMENTS - 1))
    reward = CooperativeReward(settings, SEGMENTS, 2, body, torch.ones(SEGMENTS, 9))
    arguments = list(goal_step()[2])
    arguments[8] = 1e-3 * torch.tensor([[2.0, 0.5, 0.5, -0.5, 0.5]])
    arguments[9] = 1e-3 * torch.tensor([[0.0, 0.0, 2.5, 0.0, 0.0]])
    angle = torch.tensor([[0.0, 0.0, 0.0, 0.0, math.radians(60)]])

    _, parts, _ = reward.compute(*arguments, angle)

    by_name = dict(zip(PART_NAMES, parts.reward_parts[0].unbind(dim=-1), strict=True))
    every = torch.ones(SEGMENTS)
    torch.testing.assert_close(by_name["line"], 0.325 * every)
    torch.testing.assert_close(by_name["squeeze"], -0.2 * every)
    assert not by_name["progress"].any() and not by_name["arrival"].any()
    torch.testing.assert_close(parts.segment_progress, arguments[8])
    assert reward.free_support.tolist() == pytest.approx([0.8, 0, 0, 0, 0])


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
    averaged = torch.zeros(1, SEGMENTS, 5)
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


def test_a_block_can_hold_the_joint_forces_of_the_step_and_the_tow_level():
    builder = CooperativeObservationBuilder(
        SEGMENTS,
        2,
        2,
        LEGS,
        BODY.unit_weight_n,
        "cpu",
        instant_joint_forces=True,
        tow_input=True,
    )
    block = builder.block_size
    assert (block, builder.observation_size) == (50, 5 * 50 + 3)
    state = resting_state(2)
    state.front_joint_force[1, 1] = -0.5e-3
    state.net_joint_force[1, 1] = torch.tensor([0.2e-3, 0.7e-3])

    observations = builder.build(
        state,
        torch.zeros(2, 2),
        torch.zeros(2, SEGMENTS, LEGS + 1),
        torch.zeros(2, SEGMENTS, 3),
        torch.zeros(2, SEGMENTS, 5),
        torch.tensor([0.0, 0.25]),
    )

    # After the averaged forces: the front joint's and the net along, now.
    torch.testing.assert_close(observations[1, 1, 36:38], torch.tensor([-0.5, 0.2]))
    torch.testing.assert_close(
        observations[:, :, -1], torch.tensor([[0.0], [0.25]]) * torch.ones(2, SEGMENTS)
    )


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
    line = HeadLine(2, BODY.segment_lengths_m, episode_steps=40, device="cpu")
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
            torch.zeros(2, SEGMENTS, 5),
            line,
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


def test_on_the_line_v5_walks_with_no_target_and_restarts_a_strayed_world():
    """The whole environment on v5 with the straight line: the head's
    target values stay zero, no target is placed, every segment starts on
    the line, and the first segment, which carries the head, may hang 0.72
    of a unit's weight on the second for free. World 1's line moved 15 mm
    to the side, beyond line_restart_mm, it restarts at once on a new line,
    and world 0 at the time limit."""
    settings = EnvironmentSettings.from_section(
        {
            "design": "cooperators",
            "max_episode_steps": 3,
            "observation_radius": 2,
            "simulation": {
                "model_path": str(Path("models/assembly_v5.xml").resolve()),
                "backend": "cpu",
                "world_count": 2,
            },
            "cooperators": {
                "stagger_first_episodes": False,
                "task": "line",
                "reward_channels": "team_own",
                "line_closeness_mm": 10.0,
                "line_speed_cap_body_lengths_per_s": 1.0,
                "line_restart_mm": 10.0,
            },
        }
    )
    environment = create_environment(settings)
    observations = environment.reset(seed=0)
    assert observations.shape == (2, 21, 5 * 48 + 2)
    torch.testing.assert_close(
        environment.line.view, torch.zeros(2, 21, 3), atol=1e-6, rtol=0
    )
    assert environment.reward.free_support[0] == pytest.approx(0.72, abs=0.01)
    assert not environment.reward.free_support[1:].any()

    line = environment.line
    line.origin[1] += 0.015 * torch.stack((-line.direction[1, 1], line.direction[1, 0]))
    for expected in ([False, True], [False, False], [True, False]):
        observations, rewards, _, truncated, _ = environment.step(torch.zeros(2, 21, 9))
        assert torch.isfinite(rewards).all() and not observations[..., -2:].any()
        assert truncated.tolist() == expected
    assert not environment.task.target_position.any()


def test_a_tow_above_the_friction_slides_the_body_along_and_one_below_it_does_not():
    """The whole environment on v4, its unpowered legs letting it down onto
    its belly: towed by its own weight, more than the floor's friction of
    0.8 can hold, the body slides forward along its heading; towed by 0.4,
    once settled, it creeps a few hundredths of a millimetre in 0.1 s."""
    settings = EnvironmentSettings.from_section(
        {
            "design": "cooperators",
            "max_episode_steps": 100,
            "simulation": {
                "model_path": str(Path("models/assembly_v4.xml").resolve()),
                "backend": "cpu",
                "world_count": 2,
            },
            "cooperators": {"stagger_first_episodes": False, "tow_start": 1.0},
        }
    )
    environment = create_environment(settings)
    environment.start_tow()
    environment.reset(seed=0)
    environment.tow.shares.copy_(torch.tensor([1.0, 0.4]))
    state = environment.simulation.physical_state
    actions = torch.zeros(2, 12, 9)
    for _ in range(10):  # 0.2 s to settle
        environment.step(actions)
    start = state.head_tip_position[:, :2].clone()
    for _ in range(5):
        environment.step(actions)
    moved = state.head_tip_position[:, :2] - start
    assert moved[0].norm() > 5e-3 and moved[1].norm() < 1e-4
    w, x, y, z = state.body_quaternion[0, 0]
    forward = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z)))
    assert (moved[0] @ forward) / moved[0].norm() > 0.9
