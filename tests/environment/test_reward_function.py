"""Tests for the reward function, with hand-made states of a three-segment body.

The body lies along the x axis: centres at 0, −4 and −8 mm, the head's tip at
+5 mm, and the target 15 mm ahead of the tip. World 0 changes nothing during the
step; world 1 changes what each test describes.
"""

import pytest
import torch

from centipede.environment.reward_function import RewardFunction
from centipede.environment.settings import RewardSettings
from centipede.environment.simulation import PhysicalState

SETTINGS = RewardSettings(
    arrival_reward=1.0,
    efficiency_cost=0.003,
    body_contact_cost=0.010,
    leg_contact_cost=0.005,
    distance_ratio_epsilon_m=1e-6,
)
CENTRES = torch.tensor([[0.0, 0.0], [-0.004, 0.0], [-0.008, 0.0]])
TIP = torch.tensor([0.005, 0.0])
TARGET = torch.tensor([0.020, 0.0])
ARRIVAL, EFFICIENCY, BODY, LEG = range(4)


def unchanged_state() -> PhysicalState:
    state = PhysicalState.allocate(2, 3, "cpu")
    state.body_planar_position[:] = CENTRES
    state.head_tip_position[:, :2] = TIP
    return state


def rewards_for(state, arrived=(False, False), settings=SETTINGS):
    return RewardFunction(settings).compute(
        state,
        previous_body_planar_position=CENTRES.expand(2, 3, 2).clone(),
        previous_head_tip_position=TIP.expand(2, 2).clone(),
        target_position=TARGET.expand(2, 2).clone(),
        arrived=torch.tensor(arrived),
    )


def test_efficiency_follows_progress_toward_each_goal():
    state = unchanged_state()
    state.head_tip_position[1, 0] += 0.005  # head: 15 mm → 10 mm from the target
    state.body_planar_position[1, 1, 0] += 0.002  # 4 mm → 2 mm from leader's spot
    state.body_planar_position[1, 2, 0] -= 0.004  # 4 mm → 8 mm: moving away

    step = rewards_for(state)
    efficiency = step.reward_parts[..., EFFICIENCY]

    assert torch.allclose(efficiency[0], torch.full((3,), -0.003))

    def ratio(after_mm, before_mm):  # with the epsilon of 1e-6 m = 0.001 mm
        return -0.003 * (0.001 + after_mm) / (0.001 + before_mm)

    assert efficiency[1].tolist() == pytest.approx(
        [ratio(10, 15), ratio(2, 4), ratio(8, 4)], rel=1e-5
    )
    assert step.segment_progress[0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert step.segment_progress[1].tolist() == pytest.approx(
        [0.005, 0.002, -0.004], abs=1e-7
    )


def test_contacts_cost_their_weight_and_arrival_rewards_every_segment():
    state = unchanged_state()
    state.body_ground_contact[1, 0] = True
    state.leg_leg_contact[1, 1:] = True

    step = rewards_for(state, arrived=(False, True))

    parts = step.reward_parts[1]
    assert parts[:, ARRIVAL].tolist() == [1.0, 1.0, 1.0]
    assert parts[:, BODY].tolist() == pytest.approx([-0.010, 0.0, 0.0])
    assert parts[:, LEG].tolist() == pytest.approx([0.0, -0.005, -0.005])
    assert not step.reward_parts[0][:, [ARRIVAL, BODY, LEG]].any()
    assert torch.allclose(step.rewards, step.reward_parts.sum(dim=-1))


def test_a_zero_weight_switches_its_term_off():
    state = unchanged_state()
    state.leg_leg_contact[:] = True
    settings = RewardSettings(1.0, 0.003, 0.010, 0.0, 1e-6)

    function = RewardFunction(settings)
    step = rewards_for(state, settings=settings)

    assert function.term_names == [
        "arrival",
        "efficiency",
        "body_contact",
        "leg_contact",
    ]
    assert not step.reward_parts[..., LEG].any()
    assert torch.allclose(step.rewards, torch.full((2, 3), -0.003))
