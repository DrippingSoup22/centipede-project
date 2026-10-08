"""Tests for the reward function, with hand-made states of a three-segment body.

The body lies along the x axis: centres at 0, −4 and −8 mm, the head's tip at
+5 mm, and the target 15 mm ahead of the tip. World 0 changes nothing during the
step; world 1 changes what each test describes.
"""

import math

import pytest
import torch

from centipede.environment.reward_function import RewardFunction
from centipede.environment.settings import RewardSettings
from centipede.environment.simulation import PhysicalState

EPISODE_STEPS = 256
ARRIVAL_RADIUS = 0.001
SETTINGS = RewardSettings.from_section({})  # the rules' defaults
CENTRES = torch.tensor([[0.0, 0.0], [-0.004, 0.0], [-0.008, 0.0]])
TIP = torch.tensor([0.005, 0.0])
TARGET = torch.tensor([0.020, 0.0])


def unchanged_state() -> PhysicalState:
    state = PhysicalState.allocate(2, 3, "cpu")
    state.body_planar_position[:] = CENTRES
    state.head_tip_position[:, :2] = TIP
    return state


def rewards_for(state, arrived=(False, False), settings=SETTINGS):
    function = RewardFunction(settings, EPISODE_STEPS, ARRIVAL_RADIUS)
    step = function.compute(
        state,
        previous_body_planar_position=CENTRES.expand(2, 3, 2).clone(),
        previous_head_tip_position=TIP.expand(2, 2).clone(),
        target_position=TARGET.expand(2, 2).clone(),
        arrived=torch.tensor(arrived),
    )
    parts = dict(zip(function.term_names, step.reward_parts.unbind(-1), strict=True))
    return step, parts, function.weights.per_step


def test_the_worst_arrival_balances_to_zero_through_the_discount():
    """Rule R1: every cost on every step and an arrival on the last one add up
    to zero seen from the start, with the discount of rule R0; the costs take
    their parts of that budget, and a halving is worth one episode of step cost."""
    weights = SETTINGS.weights(EPISODE_STEPS)
    discount = 2 ** (-1 / EPISODE_STEPS)
    worst_cost = sum(weights.per_step[name] for name in weights.episode_shares)
    discounted_steps = sum(discount**step for step in range(EPISODE_STEPS))
    last_arrival = discount ** (EPISODE_STEPS - 1)
    assert last_arrival - worst_cost * discounted_steps == pytest.approx(0, abs=1e-9)

    shares = weights.episode_shares
    assert sum(shares.values()) == pytest.approx(0.6941, abs=1e-4)  # about ln 2
    assert shares["body_contact"] == pytest.approx(3 * shares["leg_contact"])
    assert shares["step_cost"] == pytest.approx(2 * shares["leg_contact"])
    assert weights.per_step["progress"] == pytest.approx(shares["step_cost"])

    # Costs switched off keep the others' parts: the budget stays in six parts.
    any_cost = RewardSettings.from_section(
        {
            "body_contact_cost_parts": 0,
            "leg_contact_cost_parts": 0,
            "cost_budget_parts": 6,
        }
    ).weights(EPISODE_STEPS)
    step_cost = weights.per_step["step_cost"]
    assert any_cost.per_step["step_cost"] == pytest.approx(step_cost)
    assert any_cost.per_step["body_contact"] == any_cost.per_step["leg_contact"] == 0


def test_progress_counts_halvings_of_each_distance_followers_at_their_share():
    state = unchanged_state()
    state.head_tip_position[1, 0] += 0.0075  # head: 15 mm → 7.5 mm, one halving
    state.body_planar_position[1, 1, 0] += 0.002  # 4 mm → 2 mm from the leader's spot
    state.body_planar_position[1, 2, 0] -= 0.004  # 4 mm → 8 mm: moving away

    step, parts, weights = rewards_for(state)

    head = weights["progress"]
    assert not parts["progress"][0].any()
    assert parts["progress"][1].tolist() == pytest.approx(
        [head, 0.1 * head, -0.1 * head], rel=1e-5
    )
    assert step.segment_progress[1].tolist() == pytest.approx(
        [0.0075, 0.002, -0.004], abs=1e-7
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
