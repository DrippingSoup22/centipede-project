"""Tests for the reward function, with hand-made states of a three-segment body.

The body lies along the x axis: centres at 0, −4 and −8 mm, the head's tip at
+5 mm, and the target 15 mm ahead of the tip. World 0 changes nothing during the
step; world 1 changes what each test describes. Every segment but the rear
commands the spine joint behind it, as with spine control.
"""

import math

import pytest
import torch

from centipede.environment.clocks import ClockStep
from centipede.environment.reward_function import RewardFunction
from centipede.environment.settings import RewardSettings
from centipede.environment.simulation import PhysicalState

EPISODE_STEPS = 256
ARRIVAL_RADIUS = 0.001
SETTINGS = RewardSettings.from_section({})  # the rules' defaults
CENTRES = torch.tensor([[0.0, 0.0], [-0.004, 0.0], [-0.008, 0.0]])
TIP = torch.tensor([0.005, 0.0])
TARGET = torch.tensor([0.020, 0.0])
COMMANDED_JOINTS = torch.ones(3, 7)
COMMANDED_JOINTS[2, 6] = 0.0  # the rear has no spine joint behind it


def unchanged_state() -> PhysicalState:
    state = PhysicalState.allocate(2, 3, "cpu")
    state.body_planar_position[:] = CENTRES
    state.head_tip_position[:, :2] = TIP
    return state


def rewards_for(
    state,
    arrived=(False, False),
    settings=SETTINGS,
    joint_action=None,
    foot_slip_speed=None,
    clock=None,
):
    function = RewardFunction(settings, EPISODE_STEPS, ARRIVAL_RADIUS, COMMANDED_JOINTS)
    step = function.compute(
        state,
        previous_body_planar_position=CENTRES.expand(2, 3, 2).clone(),
        previous_head_tip_position=TIP.expand(2, 2).clone(),
        target_position=TARGET.expand(2, 2).clone(),
        arrived=torch.tensor(arrived),
        previous_joint_position=torch.zeros(2, 3, 7),
        joint_action=torch.zeros(2, 3, 7) if joint_action is None else joint_action,
        foot_slip_speed=(
            torch.zeros(2, 3, 2) if foot_slip_speed is None else foot_slip_speed
        ),
        clock=clock,
    )
    parts = dict(zip(function.term_names, step.reward_parts.unbind(-1), strict=True))
    return step, parts, function.weights.per_step


def test_the_worst_arrival_balances_to_zero_through_the_discount():
    """Rule R1: every cost on every step and an arrival on the last one add up
    to zero seen from the start, with the discount of rule R0; the costs take
    their parts of that budget, and a halving is worth one episode of step cost.
    With a cost horizon, the same holds for an arrival on the horizon's last step."""
    weights = SETTINGS.weights(EPISODE_STEPS)
    discount = 2 ** (-1 / EPISODE_STEPS)
    short = RewardSettings.from_section({"cost_horizon_steps": 64})
    for horizon, horizon_weights in (
        (EPISODE_STEPS, weights),
        (64, short.weights(EPISODE_STEPS)),
    ):
        per_step = horizon_weights.per_step
        worst_cost = sum(per_step[name] for name in horizon_weights.episode_shares)
        discounted_steps = sum(discount**step for step in range(horizon))
        last_arrival = discount ** (horizon - 1)
        assert last_arrival - worst_cost * discounted_steps == pytest.approx(
            0, abs=1e-9
        )

    shares = weights.episode_shares
    assert sum(shares.values()) == pytest.approx(0.6941, abs=1e-4)  # about ln 2
    assert shares["body_contact"] == pytest.approx(3 * shares["leg_contact"])
    assert shares["step_cost"] == pytest.approx(2 * shares["leg_contact"])
    assert weights.per_step["progress"] == pytest.approx(shares["step_cost"])

    # Costs switched off keep the others' parts, the budget staying in six
    # parts, and are left out of the terms.
    any_cost = RewardSettings.from_section(
        {
            "body_contact_cost_parts": 0,
            "leg_contact_cost_parts": 0,
            "cost_budget_parts": 6,
        }
    ).weights(EPISODE_STEPS)
    step_cost = weights.per_step["step_cost"]
    assert any_cost.per_step["step_cost"] == pytest.approx(step_cost)
    assert "body_contact" not in any_cost.per_step
    assert "leg_contact" not in any_cost.per_step


def test_progress_counts_the_heads_halvings_for_every_segment():
    state = unchanged_state()
    state.head_tip_position[1, 0] += 0.0075  # head: 15 mm → 7.5 mm, one halving
    state.body_planar_position[1, 1, 0] += 0.002  # 4 mm → 2 mm from the leader's spot
    state.body_planar_position[1, 2, 0] -= 0.004  # 4 mm → 8 mm: moving away

    step, parts, weights = rewards_for(state)

    head = weights["progress"]
    assert not parts["progress"][0].any()
    assert parts["progress"][1].tolist() == pytest.approx([head] * 3, rel=1e-5)
    assert step.segment_progress[1].tolist() == pytest.approx(
        [0.0075, 0.002, -0.004], abs=1e-7
    )

    # The first runs of the rules paid followers for their own goals instead.
    own_goals = RewardSettings.from_section({"follower_progress_ratio": 0.1})
    _, parts, _ = rewards_for(state, settings=own_goals)
    assert parts["progress"][1].tolist() == pytest.approx(
        [head, 0.1 * head, -0.1 * head], rel=1e-5
    )

    # Landing on the target's centre counts as reaching the arrival radius.
    state.head_tip_position[1, 0] = TARGET[0]
    _, parts, _ = rewards_for(state, arrived=(False, True))
    assert parts["progress"][1, 0].item() == pytest.approx(head * math.log2(15))


def test_costs_and_arrival_take_their_weights_and_add_up():
    state = unchanged_state()
    state.body_ground_contact[1, 0] = True
    state.leg_leg_contact[1, 1:] = True

    step, parts, weights = rewards_for(state, arrived=(False, True))

    assert parts["arrival"][1].tolist() == [1.0, 1.0, 1.0]
    assert not parts["arrival"][0].any()
    assert torch.allclose(parts["step_cost"], torch.tensor(-weights["step_cost"]))
    assert parts["body_contact"][1].tolist() == pytest.approx(
        [-weights["body_contact"], 0.0, 0.0]
    )
    assert parts["leg_contact"][1].tolist() == pytest.approx(
        [0.0, -weights["leg_contact"], -weights["leg_contact"]]
    )
    assert torch.allclose(step.rewards, step.reward_parts.sum(dim=-1))


def test_the_task_can_be_the_heads_alone_with_progress_in_parts_of_the_budget():
    """The oscillator reward: no step cost, so a halving is given in parts of the
    budget; the arrival pays half; the followers get neither arrival nor
    progress. A budget in fewer parts than the costs take makes them weigh more."""
    settings = RewardSettings.from_section(
        {
            "arrival_payout": 0.5,
            "follower_arrival_share": 0,
            "follower_progress_share": 0,
            "step_cost_parts": 0,
            "body_contact_cost_parts": 2.5,
            "leg_contact_cost_parts": 2,
            "foot_slip_cost_parts": 2,
            "progress_parts": 3,
            "cost_budget_parts": 7,
        }
    )
    state = unchanged_state()
    state.head_tip_position[1, 0] += 0.0075  # one halving

    _, parts, weights = rewards_for(state, arrived=(False, True), settings=settings)

    budget = sum(SETTINGS.weights(EPISODE_STEPS).episode_shares.values())
    assert weights["progress"] == pytest.approx(3 / 7 * budget)
    assert weights["leg_contact"] == pytest.approx(2 / 7 * budget / EPISODE_STEPS)
    assert "step_cost" not in weights
    assert parts["arrival"][1].tolist() == [0.5, 0.0, 0.0]
    assert parts["progress"][1].tolist() == pytest.approx(
        [weights["progress"], 0.0, 0.0], rel=1e-5
    )


def test_walking_costs_charge_slipping_feet_and_legs_and_tempo_off_the_clock():
    settings = RewardSettings.from_section(
        {
            "foot_slip_cost_parts": 2,
            "legs_off_tempo_cost_parts": 1.5,
            "out_of_tempo_cost_parts": 1,
        }
    )
    foot_slip_speed = torch.zeros(2, 3, 2)
    foot_slip_speed[1, 0, 0] = 0.020  # twice the unit: counts 1; the other foot 0
    foot_slip_speed[1, 1] = 0.005  # half the unit, both feet
    leg_squared_difference = torch.zeros(2, 3)
    leg_squared_difference[1, 0] = math.radians(10) ** 2  # (10 / 20)² = 0.25
    leg_squared_difference[1, 1:] = math.radians(40) ** 2  # at most 1 ...
    known = torch.ones(2, 3, dtype=torch.bool)
    known[1, 2] = False  # ... and nothing in the clock's first turn
    mismatch = torch.zeros(2, 3)
    mismatch[1] = torch.tensor([0.4, 0.2, 0.1])
    clock = ClockStep(leg_squared_difference, known, mismatch)

    _, parts, weights = rewards_for(
        unchanged_state(),
        settings=settings,
        foot_slip_speed=foot_slip_speed,
        clock=clock,
    )

    slip, legs, tempo = (
        weights[name] for name in ("foot_slip", "legs_off_tempo", "out_of_tempo")
    )
    assert parts["foot_slip"][1].tolist() == pytest.approx([-slip / 2, -slip / 2, 0])
    assert parts["legs_off_tempo"][1].tolist() == pytest.approx(
        [-legs * 0.25, -legs, 0.0]
    )
    # The head pays a quarter of its tempo mismatch.
    assert parts["out_of_tempo"][1].tolist() == pytest.approx(
        [-tempo * 0.1, -tempo * 0.2, -tempo * 0.1]
    )
    for name in ("foot_slip", "legs_off_tempo", "out_of_tempo"):
        assert not parts[name][0].any()


def test_movement_costs_the_square_of_how_far_the_commanded_joints_moved():
    settings = RewardSettings.from_section({"movement_cost_parts": 1})
    state = unchanged_state()
    state.leg_joint_position[1, 0, 0] = math.radians(10)  # one of the head's 7
    state.leg_joint_position[1, 1] = math.radians(30)  # all of segment 1's ...
    state.spine_yaw_position[1, 1] = math.radians(30)  # ... spine joint included
    state.spine_yaw_position[1, 2] = 0.5  # the rear commands no spine joint

    step, parts, weights = rewards_for(state, settings=settings)

    # The default budget is now the costs' 7 parts, one of them movement's.
    shares = settings.weights(EPISODE_STEPS).episode_shares
    assert shares["movement"] == pytest.approx(shares["step_cost"] / 2)
    assert sum(shares.values()) == pytest.approx(0.6941, abs=1e-4)
    # Mean squared movement over the joints, in units of 25° squared, at most 1.
    assert parts["movement"][1].tolist() == pytest.approx(
        [-weights["movement"] * (10 / 25) ** 2 / 7, -weights["movement"], 0.0]
    )
    assert not parts["movement"][0].any()
    assert step.joint_movement[1, 0].item() == pytest.approx(math.radians(10) ** 2 / 7)
    # Switched off, as by default, the term is left out.
    assert "movement" not in SETTINGS.weights(EPISODE_STEPS).per_step


def test_command_cost_charges_the_mean_squared_command_outside_the_budget():
    settings = RewardSettings.from_section({"command_cost_ratio": 4})
    joint_action = torch.zeros(2, 3, 7)
    joint_action[1, 0] = -1.0  # the head: every command at full
    joint_action[1, 1, :6] = 0.5  # segment 1: legs at half, spine at zero
    joint_action[1, 2, 6] = 1.0  # the rear's seventh column is padding

    _, parts, weights = rewards_for(
        unchanged_state(), settings=settings, joint_action=joint_action
    )

    assert weights["command"] == pytest.approx(4 * weights["step_cost"])
    assert parts["command"][1].tolist() == pytest.approx(
        [-weights["command"], -weights["command"] * 0.25 * 6 / 7, 0.0]
    )
    assert not parts["command"][0].any()
    # The budget keeps its shares; switched off, as by default, the term is
    # left out.
    default_weights = SETTINGS.weights(EPISODE_STEPS)
    shares = settings.weights(EPISODE_STEPS).episode_shares
    assert shares == default_weights.episode_shares
    assert "command" not in default_weights.per_step


def test_the_reward_used_before_2026_10_08_keeps_its_terms():
    settings = RewardSettings.from_section(
        {"efficiency_cost": 0.003, "body_contact_cost": 0.01, "leg_contact_cost": 0.0}
    )
    state = unchanged_state()
    state.leg_leg_contact[:] = True
    state.head_tip_position[1, 0] += 0.005  # head: 15 mm → 10 mm from the target

    step, parts, _ = rewards_for(state, settings=settings)

    assert list(parts) == ["arrival", "efficiency", "body_contact", "leg_contact"]
    assert parts["efficiency"][0].tolist() == pytest.approx([-0.003] * 3)
    # −0.003 × (ε + distance after) / (ε + distance before), ε = 0.001 mm.
    assert parts["efficiency"][1, 0].item() == pytest.approx(
        -0.003 * 10.001 / 15.001, rel=1e-5
    )
    assert not parts["leg_contact"].any()  # a zero weight switches its term off
