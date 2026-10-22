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
from centipede.environment.line import Line
from centipede.environment.observation_builder import ObservationBuilder
from centipede.environment.reward_function import (
    ENERGY_PARTS,
    PART_NAMES,
    RewardFunction,
)
from centipede.environment.settings import (
    CurveSettings,
    EnvironmentSettings,
    RewardSettings,
    TowSettings,
)
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.tow import Tow
from centipede.settings_section import SettingsError

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
    assert settings.line_shape == "straight"  # stage 1's
    with pytest.raises(SettingsError, match="levels"):
        EnvironmentSettings.from_section(
            {"simulation": HEADLESS, "line_shape": "curved"}
        )
    assert settings.tow.start == 0.0  # no tow unless asked
    weights = [settings.reward.ramp_weight_at(cycle) for cycle in (1, 65, 129, 300)]
    assert weights == [0.5, 0.75, 1.0, 1.0]
    # A ramp that waits (ramp_after_line_share) holds its start, then rises
    # from the cycle it is given.
    assert settings.reward.ramp_weight_at(300, None) == 0.5
    assert settings.reward.ramp_weight_at(300, 236) == 0.75
    # An exponential ramp multiplies by the same factor each cycle.
    exponential = RewardSettings.from_section(
        {"ramp_shape": "exponential", "ramp_start": 0.01, "ramp_cycles": 100}
    )
    weights = [exponential.ramp_weight_at(cycle) for cycle in (1, 51, 101)]
    assert weights == pytest.approx([0.01, 0.1, 1.0])
    # An adaptive ramp, the band 70 to 110 mm/s: raised 1.1 times just above
    # it, 1.1 squared a band width beyond, held within it or when the feet do
    # not step, lowered below it; within its start and 1.
    adaptive = RewardSettings.from_section(
        {"ramp_shape": "adaptive", "ramp_start": 0.01}
    )
    weights = [
        adaptive.adapted_ramp_weight(0.1, speed, stepping)
        for speed, stepping in (
            (110.1, True),
            (150, True),
            (150, False),
            (90, True),
            (69.9, True),
        )
    ]
    assert weights == pytest.approx([0.11, 0.121, 0.1, 0.1, 0.1 / 1.1], rel=1e-3)
    assert adaptive.adapted_ramp_weight(0.0105, 30, True) == 0.01
    assert adaptive.adapted_ramp_weight(0.95, 300, True) == 1.0


def test_the_line_is_laid_at_its_turn_and_measures_each_centre_on_it():
    """World 0's line runs along +x through the units' mean centre, world 1's
    turned 0.1 rad to the left. The body steps 2 mm forward: every centre of
    world 0 goes 2 mm along its line. It moves 1.5 mm to the left: none goes
    along, every centre lies 1.5 mm from the line, unit 2, turned 30 degrees,
    at that angle."""
    line = Line(2, SEGMENTS, 0.01, 0.025, "cpu")
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
    line = Line(1, SEGMENTS, 0.006, 0.013, "cpu")
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


def test_a_curved_line_bends_smoothly_beyond_the_head_and_measures_each_centre_on_it():
    """One level, its tightest radius a body length (five units 10 mm apart:
    50 mm), no straight pieces, pieces of one to two body lengths, each bend
    changed over a body length. The line runs straight through the body and
    half a body length beyond the head (20 mm ahead of the middle), to 45
    mm; then its curvature reaches toward 20 /m but never past it, and moves
    by at most 40 /m over a body length, 0.8 /m a point: no corner. Its
    points stay a millimetre apart. A unit set 2 mm to the left of the line
    where it bends most, turned 10 degrees from it, measures that distance,
    that angle, the point's place along the line and its curvature."""
    curves = CurveSettings.from_section(
        {
            "level_tightest_radius_body_lengths": [1.0],
            "level_straight_shares": [0.0],
            "piece_body_lengths": [1.0, 2.0],
        }
    )
    line = Line(1, SEGMENTS, 0.01, 0.025, "cpu", curves)
    state = resting_state(1)
    generator = torch.Generator().manual_seed(0)
    line.restart(torch.tensor([True]), state, torch.zeros(1), generator)

    curvature, along = line.curvatures[0], line.point_along
    assert not curvature[along <= 0.045].any()
    assert 10.0 < curvature.abs().max() <= 20.0 + 1e-4
    assert curvature.diff().abs().max() <= 0.8 + 1e-4
    gaps = line.points[0].diff(dim=0).norm(dim=-1)
    torch.testing.assert_close(gaps, torch.full_like(gaps, 1e-3))
    assert line.drawn_points().shape == (1, 21, 2)

    bend = int(curvature.abs().argmax())
    direction = line.directions[0, bend]
    left = torch.stack((-direction[1], direction[0]))
    state.body_planar_position[0, 2] = line.points[0, bend] + 0.002 * left
    yaw = math.atan2(direction[1], direction[0]) + math.radians(10)
    state.body_quaternion[0, 2] = turned(yaw)
    line.nearest[0, 2] = bend
    line.update(state)
    assert line.distance[0, 2].item() == pytest.approx(0.002, abs=1e-6)
    assert line.angle[0, 2].item() == pytest.approx(math.radians(10), abs=1e-5)
    assert line.along[0, 2].item() == pytest.approx(along[bend].item(), abs=1e-6)
    assert line.curvature[0, 2] == curvature[bend]


def test_the_reward_pays_the_line_and_charges_the_team_and_each_unit_its_costs():
    """Steps of 2, 0.5, 0.5, -0.5 and 0.5 mm along the line, unit 2 at 2.5 mm
    from it (closeness 0.75 in a 5 mm band), unit 4 facing 60 degrees away:
    paid 1.8 (the cap), 0.5, 0.375, -0.5 (a step back in full) and 0.25, the
    team the mean, 0.485 mm. Joint 1 is pushed together by 0.3 unit weights
    once the tows behind it are taken out, the others pulled or pushed only
    by the tows: the squeeze, 1.06 mm at full weight, over four joints, at
    the ramp's 0.5. Unit 0 is dragged by 0.3, unit 1 pushes with 0.1, units
    3 and 4 brake their own tows of 0.1: force shares of 1.06, 0.118, 0, 0.
    Unit 0 weighs 0.8 units more than the others and hangs that for free.
    Unit 1's leg joint turns at 10 rad/s, unit 0's spine joint at 3 and the
    rear's at 5, which is not its own: the motion, at 0.1 mm a second for
    each (rad/s)², is 0.018 and 0.2 mm, 0.0218 for the team at the ramp's 0.5.
    Unit 1's leg joint speeds up by 100 rad/s², the rear's spine joint by 50:
    the accelerations, at 0.001 mm a second for each (rad/s²)², 0.02 mm for
    the team. Unit 1's feet carry half its weight on average: the carry, at
    10 mm a second for each unit weight, 0.1 mm; unit 0's carry its own,
    beyond its free support, and more. With one account the team's holds
    every part, and the own account none."""
    body = replace(BODY, segment_weights_n=(1.8e-3,) + (1e-3,) * (SEGMENTS - 1))
    commanded = torch.ones(SEGMENTS, LEGS + 1)
    commanded[-1, -1] = 0.0
    settings = RewardSettings.from_section(
        {"motion_price": 0.1, "acceleration_price": 1e-3, "carry_price": 10.0}
    )
    reward = RewardFunction(settings, body, commanded, 0.005, 2)
    reward.ramp_weight = 0.5
    state = resting_state(1)
    state.leg_joint_velocity[0, 1, 0] = 10.0
    state.spine_yaw_velocity[0, 0] = 3.0
    state.spine_yaw_velocity[0, -1] = 5.0
    state.motor_positive_work[0, 0] = 2e-3
    state.motor_negative_work[0, 1] = 4e-3
    state.motor_force[0, 0, 0] = 1.0
    progress = 1e-3 * torch.tensor([[2.0, 0.5, 0.5, -0.5, 0.5]])
    distance = 1e-3 * torch.tensor([[0.0, 0.0, 2.5, 0.0, 0.0]])
    angle = torch.tensor([[0.0, 0.0, 0.0, 0.0, math.radians(60)]])
    averaged = torch.zeros(1, SEGMENTS, 8)
    averaged[0, :, 0] = torch.tensor([0.3, -0.1, 0.0, -0.1, -0.1])  # along
    averaged[0, :, 1] = torch.tensor([1.0, 0.1, 0.0, 0.0, 0.0])  # up
    averaged[0, :, 2] = torch.tensor([0.0, -0.5, 0.2, -0.2, 0.0])  # front joint
    averaged[0, 2, 3] = 0.2  # belly
    averaged[0, 4, 4] = 2.0  # legs pressed
    averaged[0, :, 5] = torch.tensor([1.2, 0.5, 1.0, 1.0, 1.0])  # feet
    tow = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.1]])
    acceleration = torch.zeros(1, SEGMENTS, LEGS + 1)
    acceleration[0, 1, 0] = 100.0
    acceleration[0, -1, -1] = 50.0

    zeros = torch.zeros(1, SEGMENTS)
    step = (state, progress, zeros, zeros, distance, angle)
    rest = (
        averaged,
        tow,
        acceleration,
        torch.zeros(1, SEGMENTS),
        torch.zeros(1, SEGMENTS),
        torch.zeros(1, SEGMENTS, LEGS + 1),
        torch.zeros(1),
    )
    rewards, parts, own_energy = reward.compute(*step, *rest)

    by_name = dict(zip(PART_NAMES, parts[0].unbind(dim=-1), strict=True))
    every = torch.ones(SEGMENTS)
    torch.testing.assert_close(by_name["line"], 0.485 * every)
    torch.testing.assert_close(by_name["squeeze"], -0.1325 * every)
    # Energy, mJ: the units' mean at the ramp's 0.5; a motor fully on 0.332.
    torch.testing.assert_close(own_energy[0, 0], torch.tensor([2.0, 0.0, 0.332, 0.0]))
    torch.testing.assert_close(by_name["pushing"], -0.2 * every)
    torch.testing.assert_close(by_name["braking"], -0.1 * every)
    torch.testing.assert_close(by_name["force"], -0.0332 * every)
    torch.testing.assert_close(by_name["motion"], -0.0218 * every)
    torch.testing.assert_close(by_name["acceleration"], -0.02 * every)
    torch.testing.assert_close(by_name["carry"], -0.1 * torch.eye(SEGMENTS)[1])
    torch.testing.assert_close(
        by_name["force_share"], -torch.tensor([1.06, 1.06 / 9, 0.0, 0.0, 0.0])
    )
    # Hanging at 0.664 mJ a unit weight; pain, a leg's strongest push, 0.332.
    torch.testing.assert_close(
        by_name["hanging"], -0.664 * torch.tensor([0.2, 0.1, 0.2, 0.0, 0.0])
    )
    torch.testing.assert_close(by_name["pain"], -0.166 * torch.eye(SEGMENTS)[4])
    team = 0.485 - 0.1325 - 1.59 * (0.2 + 0.1 + 0.0332) - 0.0218 - 0.02
    torch.testing.assert_close(rewards[0, :, 0], team * every)
    torch.testing.assert_close(rewards[0, 0, 1], torch.tensor(-1.06 - 1.59 * 0.1328))
    one = RewardFunction(
        replace(settings, own_account=False), body, commanded, 0.005, 2
    )
    one.ramp_weight = 0.5
    one_account, _, _ = one.compute(*step, *rest)
    torch.testing.assert_close(one_account[..., 0], rewards.sum(dim=-1))
    assert not one_account[..., 1].any()


@pytest.mark.parametrize(
    ("slope", "paid_above"),
    [(0.0, (90 / 135, 90 / 200)), (1.0, (45 / 135, -20 / 200))],
)
def test_the_averaged_speed_caps_the_line_or_sets_its_target(slope, paid_above):
    """With ``line_cap_averaged``, steps of 2 mm (100 mm/s, above the 90 mm/s
    cap) are paid in full while the unit's averaged speed is 50 or 90 mm/s;
    at 135 and 200 mm/s the cap pays 90/v of them, and the target, slope 1,
    (90 - (v - 90))/v: a third, then a charge. A step back is charged in
    full."""
    settings = RewardSettings.from_section(
        {"line_cap_averaged": True, "line_overspeed_slope": slope}
    )
    commanded = torch.ones(SEGMENTS, LEGS + 1)
    reward = RewardFunction(settings, BODY, commanded, 0.005, 2)
    progress = 1e-3 * torch.tensor([[2.0, 2.0, 2.0, -0.5, 2.0]])
    averaged_speed = 1e-3 * torch.tensor([[50.0, 90.0, 135.0, 135.0, 200.0]])
    zeros = torch.zeros(1, SEGMENTS)

    _, parts, _ = reward.compute(
        resting_state(1),
        progress,
        averaged_speed,
        zeros,
        zeros,
        zeros,
        torch.zeros(1, SEGMENTS, 8),
        zeros,
        torch.zeros(1, SEGMENTS, LEGS + 1),
        zeros,
        zeros,
        torch.zeros(1, SEGMENTS, LEGS + 1),
        torch.zeros(1),
    )

    at_135, at_200 = paid_above
    paid = torch.tensor([2.0, 2.0, 2.0 * at_135, -0.5, 2.0 * at_200])
    torch.testing.assert_close(
        parts[0, :, PART_NAMES.index("line")], paid.mean() * torch.ones(SEGMENTS)
    )


@pytest.mark.parametrize(
    ("setting", "part"),
    [("acceleration_halving", "acceleration"), ("time_value_mj_per_s", "energy_share")],
)
def test_the_accelerations_or_energy_can_take_a_share_of_the_forward_pay(setting, part):
    """With ``acceleration_halving`` 10⁴ (rad/s²)², or ``time_value_mj_per_s``
    10⁴ mJ/s, at the ramp's 0.5, the units' averaged accelerations, or
    energy, 2·10⁴ on average, keep half of the forward pay (1.8 at the cap,
    0.5, 0.5 and 0.5 mm, a mean of 0.66) and the step back is charged in full
    (a mean of -0.1): the team gets 0.23 mm."""
    settings = RewardSettings.from_section({setting: 1e4})
    reward = RewardFunction(settings, BODY, torch.ones(SEGMENTS, LEGS + 1), 0.005, 2)
    reward.ramp_weight = 0.5
    zeros = torch.zeros(1, SEGMENTS)
    averaged = 1e4 * torch.tensor([[3.0, 1.0, 2.0, 2.0, 2.0]])
    accelerations, energy = (
        (averaged, zeros) if part == "acceleration" else (zeros, averaged)
    )

    rewards, parts, _ = reward.compute(
        resting_state(1),
        1e-3 * torch.tensor([[2.0, 0.5, 0.5, -0.5, 0.5]]),
        zeros,
        zeros,
        zeros,
        zeros,
        torch.zeros(1, SEGMENTS, 8),
        zeros,
        torch.zeros(1, SEGMENTS, LEGS + 1),
        accelerations,
        energy,
        torch.zeros(1, SEGMENTS, LEGS + 1),
        torch.zeros(1),
    )

    by_name = dict(zip(PART_NAMES, parts[0].unbind(dim=-1), strict=True))
    torch.testing.assert_close(by_name["line"], 0.56 * torch.ones(SEGMENTS))
    torch.testing.assert_close(by_name[part], -0.33 * torch.ones(SEGMENTS))
    torch.testing.assert_close(rewards[0, :, 0], 0.23 * torch.ones(SEGMENTS))


def test_the_team_is_paid_near_a_commanded_speed_and_charged_its_neighbours_energy():
    """A commanded speed of 67 mm/s: units' forward lines averaged at 60 to
    100 mm/s, a mean of 80, pay 67 - 13 = 54 mm a second, 1.08 mm a step.
    World 1, in a bend whose averaged command has fallen to 40 mm/s, walks
    at a mean of 40 and keeps the full 67 mm a second, 1.34 mm a step. The forward line:
    a step of 1 mm at 2.5 mm from the line (closeness 0.75 in a 5 mm band),
    one facing 60 degrees away (half), a step back in full. Unit 0 pushes
    2 mJ: each unit within two of it is charged the mean of the units it
    sees, 2/3, 2/4 and 2/5 mJ, at 0.4 mm a millijoule."""
    settings = RewardSettings.from_section(
        {
            "commanded_speed_mm_per_s": 67.0,
            "neighbourhood_energy": True,
            "mm_per_mj": 0.4,
        }
    )
    reward = RewardFunction(settings, BODY, torch.ones(SEGMENTS, LEGS + 1), 0.005, 2)
    state = resting_state(2)
    state.motor_positive_work[0, 0] = 2e-3
    zeros = torch.zeros(2, SEGMENTS)
    averaged_forward_line = torch.tensor(
        [[60.0, 70.0, 80.0, 90.0, 100.0], [30.0, 40.0, 50.0, 30.0, 50.0]]
    )

    rewards, parts, _ = reward.compute(
        state,
        zeros,
        zeros,
        averaged_forward_line,
        zeros,
        zeros,
        torch.zeros(2, SEGMENTS, 8),
        zeros,
        torch.zeros(2, SEGMENTS, LEGS + 1),
        zeros,
        zeros,
        torch.zeros(2, SEGMENTS, LEGS + 1),
        torch.tensor([67.0, 40.0]),
    )

    by_name = dict(zip(PART_NAMES, parts.unbind(dim=-1), strict=True))
    every = torch.ones(SEGMENTS)
    torch.testing.assert_close(
        by_name["line"], torch.stack((1.08 * every, 1.34 * every))
    )
    charged = torch.tensor([2 / 3, 2 / 4, 2 / 5, 0.0, 0.0])
    torch.testing.assert_close(by_name["pushing"][0], -charged)
    torch.testing.assert_close(rewards[0, :, 0], 1.08 - 0.4 * charged)
    forward = reward.forward_line(
        1e-3 * torch.tensor([[1.0, 1.0, 1.0, -0.5, 0.0]]),
        1e-3 * torch.tensor([[0.0, 2.5, 0.0, 2.5, 0.0]]),
        torch.tensor([[0.0, 0.0, math.radians(60), 0.0, 0.0]]),
    )
    torch.testing.assert_close(forward, torch.tensor([[1.0, 0.75, 0.5, -0.5, 0.0]]))
    with pytest.raises(SettingsError):
        RewardSettings.from_section(
            {"commanded_speed_mm_per_s": 67.0, "line_cap_averaged": True}
        )


def test_with_the_units_behind_weighed_less_each_unit_answers_more_for_those_ahead():
    """Unit 2 pushes 2 mJ, the units behind weighed at a half: units 0 and 1,
    ahead of it, are charged 0.5 / 2 and 0.5 / 3 of it; unit 2 itself 1 / 4;
    units 3 and 4, behind it, 1 / 3.5 and 1 / 3 (alike: 2/5, 2/4, 2/3)."""
    settings = RewardSettings.from_section(
        {
            "commanded_speed_mm_per_s": 67.0,
            "neighbourhood_energy": True,
            "energy_weight_behind": 0.5,
            "mm_per_mj": 1.0,
        }
    )
    reward = RewardFunction(settings, BODY, torch.ones(SEGMENTS, LEGS + 1), 0.005, 2)
    state = resting_state(1)
    state.motor_positive_work[0, 2] = 2e-3
    zeros = torch.zeros(1, SEGMENTS)

    _, parts, _ = reward.compute(
        state,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        torch.zeros(1, SEGMENTS, 8),
        zeros,
        torch.zeros(1, SEGMENTS, LEGS + 1),
        zeros,
        zeros,
        torch.zeros(1, SEGMENTS, LEGS + 1),
        torch.zeros(1),
    )

    pushing = parts[0, :, PART_NAMES.index("pushing")]
    charged = 2 * torch.tensor([0.5 / 2, 0.5 / 3, 1 / 4, 1 / 3.5, 1 / 3])
    torch.testing.assert_close(pushing, -charged)


def test_with_the_carry_per_leg_each_foot_is_charged_beyond_its_band():
    """Each foot's averaged load against half its unit's weight, 0.5, free
    within a quarter of it: feet carrying 0.5 and 0.5, or 0.6 and 0.4, pay
    nothing; one foot carrying the unit, 1.0 and 0, pays 0.375 twice; a unit
    holding up its neighbours, 0.9 and 0.9, 0.275 twice; a foot carrying
    0.2, 0.175. At 10 mm a second for each unit weight, 0.2 mm a step."""
    settings = RewardSettings.from_section({"carry_price": 10.0, "carry_per_leg": True})
    reward = RewardFunction(settings, BODY, torch.ones(SEGMENTS, LEGS + 1), 0.005, 2)
    zeros = torch.zeros(1, SEGMENTS)
    averaged = torch.zeros(1, SEGMENTS, 8)
    averaged[0, :, 6] = torch.tensor([0.5, 0.6, 1.0, 0.9, 0.2])  # left feet
    averaged[0, :, 7] = torch.tensor([0.5, 0.4, 0.0, 0.9, 0.5])  # right feet

    _, parts, _ = reward.compute(
        resting_state(1),
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        averaged,
        zeros,
        torch.zeros(1, SEGMENTS, LEGS + 1),
        zeros,
        zeros,
        torch.zeros(1, SEGMENTS, LEGS + 1),
        torch.zeros(1),
    )

    by_name = dict(zip(PART_NAMES, parts.unbind(dim=-1), strict=True))
    torch.testing.assert_close(
        by_name["carry"][0], -0.2 * torch.tensor([0.0, 0.0, 0.75, 0.55, 0.175])
    )


def test_energy_holds_force_to_its_exponent_and_charges_switching_by_its_rate():
    """A motor of unit 0 at half its strength, held at 1 mW for a motor fully
    on to the power 1: 0.5 mW, 0.01 mJ over a 20 ms step. Its force changed
    by 0.2 over the step, 10 a second: switching at 2 mW for a rate of 1 a
    second, squared, 0.2 W, 0.004 mJ. The rear's spine joint, which it does
    not command, costs nothing."""
    commanded = torch.ones(SEGMENTS, LEGS + 1)
    commanded[-1, -1] = 0.0
    settings = RewardSettings.from_section(
        {"force_price_w": 1e-3, "force_exponent": 1.0, "switching_price": 2e-3}
    )
    reward = RewardFunction(settings, BODY, commanded, 0.005, 2)
    state = resting_state(1)
    state.motor_force[0, 0, 0] = 0.5
    change = torch.zeros(1, SEGMENTS, LEGS + 1)
    change[0, 0, 0] = 0.2
    change[0, -1, -1] = 1.0

    energy = reward.energy(state, change)

    expected = torch.zeros(SEGMENTS, 4)
    expected[0, 2:] = torch.tensor([0.01, 0.004])
    torch.testing.assert_close(energy[0], expected)


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
    averaged = torch.zeros(2, SEGMENTS, 8)
    averaged[0, 1, :3] = torch.tensor([0.3, 1.0, -0.5])
    averaged[0, 1, 5] = 0.7  # the feet's load
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
    assert builder.behind_entries.nonzero().flatten().tolist() == list(
        range(3 * block, 5 * block)
    )
    torch.testing.assert_close(observations[..., 5 * block : -1], trail)
    torch.testing.assert_close(
        observations[..., -1], torch.tensor([[0.0], [0.25]]).expand(2, SEGMENTS)
    )
    # Without the priced forces: no legs' press, no averaged joint forces.
    lean = ObservationBuilder(SEGMENTS, 2, 2, LEGS, BODY.unit_weight_n, "cpu", False)
    lean_observations = lean.build(
        state, last_action, averaged, trail, torch.tensor([0.0, 0.25])
    )
    assert lean.block_size == 43
    torch.testing.assert_close(
        lean_observations[0, 1, :43], torch.cat((own[:32], own[36:]))
    )
    # With the carry: the feet's averaged load before the step's joint forces.
    carry = ObservationBuilder(
        SEGMENTS, 2, 2, LEGS, BODY.unit_weight_n, "cpu", False, True
    )
    carry_observations = carry.build(
        state, last_action, averaged, trail, torch.tensor([0.0, 0.25])
    )
    torch.testing.assert_close(
        carry_observations[0, 1, :44],
        torch.cat((own[:32], torch.tensor([0.7]), own[36:])),
    )
    # With the carry per leg: each foot's averaged load, left then right.
    averaged[0, 1, 6:8] = torch.tensor([0.5, 0.2])
    per_leg = ObservationBuilder(
        SEGMENTS,
        2,
        2,
        LEGS,
        BODY.unit_weight_n,
        "cpu",
        False,
        True,
        carry_per_leg=True,
    )
    per_leg_observations = per_leg.build(
        state, last_action, averaged, trail, torch.tensor([0.0, 0.25])
    )
    assert per_leg.block_size == 45
    torch.testing.assert_close(
        per_leg_observations[0, 1, :45],
        torch.cat((own[:32], torch.tensor([0.5, 0.2]), own[36:])),
    )
    # With the energy: the unit's averaged energy after the carry.
    energy = ObservationBuilder(
        SEGMENTS, 2, 2, LEGS, BODY.unit_weight_n, "cpu", False, True, True
    )
    averaged_energy = torch.zeros(2, SEGMENTS)
    averaged_energy[0, 1] = 3.5
    energy_observations = energy.build(
        state, last_action, averaged, trail, torch.tensor([0.0, 0.25]), averaged_energy
    )
    torch.testing.assert_close(
        energy_observations[0, 1, :45],
        torch.cat((own[:32], torch.tensor([0.7, 3.5]), own[36:])),
    )
    # With the line speed: the unit's averaged forward line after the energy.
    speed = ObservationBuilder(
        SEGMENTS, 2, 2, LEGS, BODY.unit_weight_n, "cpu", False, True, True, True
    )
    averaged_forward_line = torch.zeros(2, SEGMENTS)
    averaged_forward_line[0, 1] = 64.0
    speed_observations = speed.build(
        state,
        last_action,
        averaged,
        trail,
        torch.tensor([0.0, 0.25]),
        averaged_energy,
        averaged_forward_line,
    )
    torch.testing.assert_close(
        speed_observations[0, 1, :46],
        torch.cat((own[:32], torch.tensor([0.7, 3.5, 64.0]), own[36:])),
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


def test_the_gait_is_read_from_the_feet_and_legs_and_each_episode_is_summarised():
    """Every foot stands four steps of eight, each unit's a step behind the
    one ahead and each right foot half a cycle behind its left: a wave from
    front to rear of 45 degrees per unit, with the sides alternating. Each
    left leg's lift joint swings 20 degrees either way every four steps, a
    stroke every 80 ms, while the right legs' stay still. Both worlds go 1 mm
    a step along the line; world 1 ends having lost it."""
    diagnostics = EnvironmentDiagnostics(2, SEGMENTS, BODY.unit_weight_n, "cpu", None)
    window = WindowSummary(diagnostics.step)
    state = resting_state(2)
    line = Line(2, SEGMENTS, 0.01, 0.025, "cpu")
    diagnostics.restart(torch.ones(2, dtype=torch.bool))
    delay = torch.arange(SEGMENTS)[:, None] + torch.tensor([0, 4])
    for step in range(40):
        standing = (step - delay) % 8 < 4
        state.foot_normal_force.copy_(standing * BODY.unit_weight_n)
        state.leg_joint_position[..., 1] = math.radians(20) * math.sin(
            math.pi * step / 2
        )
        diagnostics.record_step(
            state,
            torch.ones(2, SEGMENTS, 2),
            torch.zeros(2, SEGMENTS, len(PART_NAMES)),
            torch.zeros(2, SEGMENTS, len(ENERGY_PARTS)),
            torch.full((2, SEGMENTS), 1e-3),
            line,
            torch.zeros(2, SEGMENTS, 5),
            torch.tensor([False, step == 39]),
            torch.full((2,), step == 39),
            torch.full((2,), 67.0),
        )
        window.add()

    gait = window.result()
    torch.testing.assert_close(gait["foot_standing"], torch.full((SEGMENTS, 2), 0.5))
    strokes = torch.tensor([0.25, 0.0]).expand(SEGMENTS, 2)
    torch.testing.assert_close(gait["leg_stroke"], strokes)
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
    """The whole environment on the headless tester, much as run 12 sets it:
    246 values for each of ten agents (no priced forces; the feet's averaged
    load, the energy and the forward line), every unit starting within its
    sight of the line, the spine motors receiving 0.35 of their command, two
    accounts, a commanded speed, the neighbourhood's energy, the
    accelerations and the carry priced. World 1's line moved 20 mm to the
    side, beyond the sight radius, it restarts at once on a new line, its
    joints' last speeds taken afresh and its forward line averaged from 0;
    world 0 at the time limit. With curves at two levels, judged every
    cycle, and the command observed: world 1, the untowed measuring world,
    lost its line once in an episode's worth of steps, kept none, and the
    level stays; an episode's worth kept raises it, once, to the last."""
    settings = EnvironmentSettings.from_section(
        {
            "max_episode_steps": 3,
            "stagger_first_episodes": False,
            "observe_priced_forces": False,
            "observe_carry": True,
            "observe_energy": True,
            "observe_line_speed": True,
            "observe_command": True,
            "line_shape": "curved",
            "simulation": HEADLESS,
            "curves": {
                "level_tightest_radius_body_lengths": [3.0, 1.0],
                "level_straight_shares": [0.5, 0.2],
                "level_judged_cycles": 1,
            },
            "reward": {
                "commanded_speed_mm_per_s": 67.0,
                "slow_in_bends": True,
                "neighbourhood_energy": True,
                "acceleration_price": 1e-3,
                "carry_price": 50.0,
            },
        }
    )
    environment = Environment(settings)
    environment.reserve_measuring_worlds(1)
    observations = environment.reset(seed=0)
    assert observations.shape == (2, 10, 247)
    # The body on its line's straight part: the full command.
    torch.testing.assert_close(observations[..., -1], torch.full((2, 10), 67.0))
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
    direction = line.directions[1, line.middle_point]
    line.points[1] += 0.02 * torch.stack((-direction[1], direction[0]))
    action = torch.zeros(2, 10, 9)
    action[..., -1] = 1.0
    for expected in ([False, True], [False, False], [True, False]):
        observations, rewards, terminated, truncated, _ = environment.step(action)
        assert rewards.shape == (2, 10, 2) and torch.isfinite(rewards).all()
        assert truncated.tolist() == expected and not terminated.any()
        if expected[1]:
            assert environment.diagnostics.episode.strayed.tolist() == [False, True]
            torch.testing.assert_close(
                environment.joint_velocity[1], environment._joint_velocity()[1]
            )
            assert not environment.averaged_forward_line[1].any()
    torch.testing.assert_close(spine_commands[0], torch.full((2, 9), 0.35))
    assert (environment.line.distance[1] < 0.0055).all()
    assert not environment.adjust_curve_level()
    assert environment.kept_line_share == 0.0 and environment.line.level == 0
    environment.judged_steps = 3
    assert environment.adjust_curve_level() and environment.line.level == 1
    assert not environment.adjust_curve_level()
