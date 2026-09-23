"""Checks for the first reward calculation."""

from dataclasses import replace

import numpy as np
import pytest

from centipede.environment.rewards import (
    DEFAULT_REWARD_CONFIG,
    RewardTerms,
    calculate_reward_terms,
    distance_ratio_cost,
)
from centipede.environment.simulation import PhysicalSnapshot


@pytest.fixture
def previous_snapshot() -> PhysicalSnapshot:
    """Create a controlled eight-segment state for reward calculations."""
    segment_count = 8
    body_positions = np.zeros((segment_count, 2), dtype=np.float64)
    body_positions[:, 0] = -0.004 * np.arange(segment_count)
    return PhysicalSnapshot(
        body_height=np.zeros(segment_count),
        body_quaternion=np.zeros((segment_count, 4)),
        leg_joint_position=np.zeros((segment_count, 6)),
        leg_joint_velocity=np.zeros((segment_count, 6)),
        body_linear_velocity=np.zeros((segment_count, 3)),
        body_angular_velocity=np.zeros((segment_count, 3)),
        body_planar_position=body_positions,
        left_foot_ground_contact=np.ones(segment_count, dtype=np.bool_),
        right_foot_ground_contact=np.ones(segment_count, dtype=np.bool_),
        body_ground_contact=np.zeros(segment_count, dtype=np.bool_),
        leg_leg_contact=np.zeros(segment_count, dtype=np.bool_),
        head_tip_position=np.array([0.0, 0.0, 0.005]),
    )


def test_reward_terms_total_sums_signed_components() -> None:
    """Combine already weighted contributions without applying coefficients twice."""
    terms = RewardTerms(
        arrival=1.0,
        efficiency=-0.002,
        body_contact=-0.010,
        leg_contact=-0.005,
    )

    assert terms.total == pytest.approx(0.983)


def test_distance_ratio_cost_orders_progress_stationary_and_retreat() -> None:
    """Approaching must cost less, and retreating more, than staying still."""
    approaching = distance_ratio_cost(0.010, 0.005)
    stationary = distance_ratio_cost(0.010, 0.010)
    retreating = distance_ratio_cost(0.010, 0.020)

    assert stationary == pytest.approx(-DEFAULT_REWARD_CONFIG.c_efficiency)
    assert approaching > stationary > retreating
    assert approaching < 0.0


def test_reward_uses_head_target_and_follower_predecessor_paths(
    previous_snapshot: PhysicalSnapshot,
) -> None:
    """Use target distance for agent zero and frozen predecessor points for others."""
    current_positions = previous_snapshot.body_planar_position.copy()
    current_positions[1] = [-0.003, 0.0]
    current_positions[2] = [-0.009, 0.0]
    current_snapshot = replace(
        previous_snapshot,
        body_planar_position=current_positions,
        head_tip_position=np.array([0.005, 0.0, 0.005]),
    )
    target = np.array([0.010, 0.0])

    rewards = calculate_reward_terms(
        previous_snapshot,
        current_snapshot,
        target,
        target_reached=False,
    )

    assert set(rewards) == set(range(8))
    # Expected ratios are written directly from the accepted equation so this
    # check cannot pass merely because both reward and test call the same helper.
    coefficient = DEFAULT_REWARD_CONFIG.c_efficiency
    epsilon = DEFAULT_REWARD_CONFIG.epsilon_ratio
    assert rewards[0].efficiency == pytest.approx(
        -coefficient * (epsilon + 0.005) / (epsilon + 0.010)
    )
    assert rewards[1].efficiency == pytest.approx(
        -coefficient * (epsilon + 0.003) / (epsilon + 0.004)
    )
    assert rewards[2].efficiency == pytest.approx(
        -coefficient * (epsilon + 0.005) / (epsilon + 0.004)
    )
    assert rewards[3].efficiency == pytest.approx(-DEFAULT_REWARD_CONFIG.c_efficiency)


def test_arrival_is_shared_and_contact_costs_are_local(
    previous_snapshot: PhysicalSnapshot,
) -> None:
    """Copy arrival to everyone while charging only contact-owning segments."""
    body_contacts = np.zeros(8, dtype=np.bool_)
    leg_contacts = np.zeros(8, dtype=np.bool_)
    body_contacts[3] = True
    leg_contacts[[4, 5]] = True
    current_snapshot = replace(
        previous_snapshot,
        body_ground_contact=body_contacts,
        leg_leg_contact=leg_contacts,
    )

    rewards = calculate_reward_terms(
        previous_snapshot,
        current_snapshot,
        target_position=np.array([0.010, 0.0]),
        target_reached=True,
    )

    assert all(
        terms.arrival == DEFAULT_REWARD_CONFIG.c_arrival for terms in rewards.values()
    )
    assert rewards[3].body_contact == -DEFAULT_REWARD_CONFIG.c_body
    assert rewards[4].leg_contact == -DEFAULT_REWARD_CONFIG.c_leg
    assert rewards[5].leg_contact == -DEFAULT_REWARD_CONFIG.c_leg
    assert rewards[2].body_contact == 0.0
    assert rewards[3].leg_contact == 0.0
    assert rewards[6].leg_contact == 0.0


def test_foot_ground_contact_has_no_reward_cost(
    previous_snapshot: PhysicalSnapshot,
) -> None:
    """Keep normal support contacts separate from penalized body and leg contacts."""
    rewards = calculate_reward_terms(
        previous_snapshot,
        previous_snapshot,
        target_position=np.array([0.010, 0.0]),
        target_reached=False,
    )

    assert all(terms.body_contact == 0.0 for terms in rewards.values())
    assert all(terms.leg_contact == 0.0 for terms in rewards.values())
