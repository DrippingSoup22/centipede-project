"""Checks for complete partial-observation construction.

Containment in the declared PettingZoo observation spaces is tested with the
public environment after those spaces are implemented.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from centipede.environment.observations import (
    HEAD_OBSERVATION_SIZE,
    INTERIOR_OBSERVATION_SIZE,
    REAR_OBSERVATION_SIZE,
    SEGMENT_BLOCK_SIZE,
    build_observations,
    build_segment_block,
    head_target_displacement,
)
from centipede.environment.simulation import CentipedeSimulation, PhysicalSnapshot

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "assembly.xml"


@pytest.fixture
def controlled_snapshot() -> PhysicalSnapshot:
    """Create distinct values so field omissions and ordering errors are visible."""
    return PhysicalSnapshot(
        body_height=np.arange(8, dtype=np.float64) + 10.0,
        body_quaternion=np.arange(32, dtype=np.float64).reshape(8, 4) + 20.0,
        leg_joint_position=np.arange(48, dtype=np.float64).reshape(8, 6) + 60.0,
        leg_joint_velocity=np.arange(48, dtype=np.float64).reshape(8, 6) + 120.0,
        body_linear_velocity=np.arange(24, dtype=np.float64).reshape(8, 3) + 180.0,
        body_angular_velocity=np.arange(24, dtype=np.float64).reshape(8, 3) + 210.0,
        body_planar_position=np.arange(16, dtype=np.float64).reshape(8, 2) + 240.0,
        left_foot_ground_contact=np.array(
            [False, True, False, True, False, True, False, True]
        ),
        right_foot_ground_contact=np.array(
            [True, False, True, False, True, False, True, False]
        ),
        body_ground_contact=np.array(
            [False, False, True, True, False, False, True, True]
        ),
        leg_leg_contact=np.array([True, True, False, False, True, True, False, False]),
        head_tip_position=np.array([300.0, 301.0, 302.0]),
    )


@pytest.mark.parametrize("segment_id", [0, 3, 7])
def test_segment_block_matches_fixed_contract(
    controlled_snapshot: PhysicalSnapshot,
    segment_id: int,
) -> None:
    """Check the same 27-value layout for head, interior, and rear segments."""
    expected = np.concatenate(
        (
            np.atleast_1d(controlled_snapshot.body_height[segment_id]),
            controlled_snapshot.body_quaternion[segment_id],
            controlled_snapshot.leg_joint_position[segment_id],
            controlled_snapshot.body_linear_velocity[segment_id],
            controlled_snapshot.body_angular_velocity[segment_id],
            controlled_snapshot.leg_joint_velocity[segment_id],
            np.array(
                [
                    controlled_snapshot.left_foot_ground_contact[segment_id],
                    controlled_snapshot.right_foot_ground_contact[segment_id],
                    controlled_snapshot.body_ground_contact[segment_id],
                    controlled_snapshot.leg_leg_contact[segment_id],
                ]
            ),
        )
    ).astype(np.float32)

    actual = build_segment_block(controlled_snapshot, segment_id)

    assert actual.shape == (SEGMENT_BLOCK_SIZE,)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_segment_block_is_independent_from_snapshot(
    controlled_snapshot: PhysicalSnapshot,
) -> None:
    """Ensure changing an emitted observation cannot mutate physical state."""
    original_height = controlled_snapshot.body_height[2]
    observation = build_segment_block(controlled_snapshot, 2)

    observation[0] = -1.0

    assert controlled_snapshot.body_height[2] == original_height


def snapshot_with_head_pose(
    snapshot: PhysicalSnapshot,
    yaw: float,
    head_tip_xy: tuple[float, float] = (10.0, -4.0),
) -> PhysicalSnapshot:
    """Return a snapshot with a normalized yaw-only head orientation."""
    quaternions = snapshot.body_quaternion.copy()
    quaternions[0] = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    head_tip = np.array([head_tip_xy[0], head_tip_xy[1], 1.0])
    return replace(snapshot, body_quaternion=quaternions, head_tip_position=head_tip)


@pytest.mark.parametrize(
    ("yaw", "target", "expected"),
    [
        (0.0, (12.0, -4.0), (2.0, 0.0)),
        (0.0, (10.0, -1.0), (0.0, 3.0)),
        (np.pi / 2.0, (10.0, -1.0), (3.0, 0.0)),
        (np.pi / 2.0, (8.0, -4.0), (0.0, 2.0)),
        (np.pi, (8.0, -4.0), (2.0, 0.0)),
    ],
)
def test_head_target_displacement_uses_head_forward_and_left_axes(
    controlled_snapshot: PhysicalSnapshot,
    yaw: float,
    target: tuple[float, float],
    expected: tuple[float, float],
) -> None:
    """Check forward and left signs for simple world-aligned orientations."""
    snapshot = snapshot_with_head_pose(controlled_snapshot, yaw)

    actual = head_target_displacement(snapshot, np.asarray(target))

    assert actual.shape == (2,)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_head_target_displacement_is_independent_of_world_translation(
    controlled_snapshot: PhysicalSnapshot,
) -> None:
    """Moving head and target equally must preserve their relative displacement."""
    yaw = np.pi / 4.0
    first = snapshot_with_head_pose(controlled_snapshot, yaw, (0.0, 0.0))
    shifted = snapshot_with_head_pose(controlled_snapshot, yaw, (5.0, -3.0))

    first_displacement = head_target_displacement(first, np.array([2.0, 1.0]))
    shifted_displacement = head_target_displacement(
        shifted,
        np.array([7.0, -2.0]),
    )

    np.testing.assert_allclose(first_displacement, shifted_displacement, atol=1e-6)


def test_build_observations_uses_complete_partial_layout(
    controlled_snapshot: PhysicalSnapshot,
) -> None:
    """Check keys, shapes, dtype, and block order for all segment positions."""
    snapshot = snapshot_with_head_pose(controlled_snapshot, yaw=0.0)
    target = np.array([12.0, -1.0])
    blocks = {
        segment_id: build_segment_block(snapshot, segment_id) for segment_id in range(8)
    }

    observations = build_observations(snapshot, target)

    assert set(observations) == set(range(8))
    np.testing.assert_array_equal(
        observations[0],
        np.concatenate((blocks[0], blocks[1], np.array([2.0, 3.0]))),
    )
    for segment_id in range(1, 7):
        np.testing.assert_array_equal(
            observations[segment_id],
            np.concatenate(
                (
                    blocks[segment_id],
                    blocks[segment_id - 1],
                    blocks[segment_id + 1],
                )
            ),
        )
    np.testing.assert_array_equal(
        observations[7],
        np.concatenate((blocks[7], blocks[6])),
    )

    expected_shapes = {
        0: (HEAD_OBSERVATION_SIZE,),
        **{segment_id: (INTERIOR_OBSERVATION_SIZE,) for segment_id in range(1, 7)},
        7: (REAR_OBSERVATION_SIZE,),
    }
    for segment_id, observation in observations.items():
        assert observation.shape == expected_shapes[segment_id]
        assert observation.dtype == np.float32
        assert np.isfinite(observation).all()


def test_only_head_observation_changes_with_target(
    controlled_snapshot: PhysicalSnapshot,
) -> None:
    """Ensure followers receive no direct or indirect target information."""
    snapshot = snapshot_with_head_pose(controlled_snapshot, yaw=0.0)

    first = build_observations(snapshot, np.array([12.0, -1.0]))
    second = build_observations(snapshot, np.array([8.0, -7.0]))

    assert not np.array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[0][:-2], second[0][:-2])
    for segment_id in range(1, 8):
        np.testing.assert_array_equal(first[segment_id], second[segment_id])


def test_real_simulation_snapshot_builds_finite_observations() -> None:
    """Smoke-test the observation helpers against actual MuJoCo snapshot shapes."""
    simulation = CentipedeSimulation(model_path=MODEL_PATH)
    try:
        simulation.reset(seed=0)
        snapshot = simulation.snapshot()
        target = snapshot.head_tip_position[:2] + np.array([0.015, 0.0])

        observations = build_observations(snapshot, target)

        assert observations[0].shape == (HEAD_OBSERVATION_SIZE,)
        assert observations[3].shape == (INTERIOR_OBSERVATION_SIZE,)
        assert observations[7].shape == (REAR_OBSERVATION_SIZE,)
        assert all(np.isfinite(value).all() for value in observations.values())
    finally:
        simulation.close()
