"""Build partial policy observations from a complete physical snapshot.

This module does not read MuJoCo directly. It receives the trusted snapshot
created by :class:`CentipedeSimulation` and exposes only the fields that each
segment agent is allowed to observe.
"""

import numpy as np
from numpy.typing import NDArray

from centipede.environment.simulation import PhysicalSnapshot

# One segment contributes 23 physical values and four contact flags.
SEGMENT_BLOCK_SIZE = 27

# Observation sizes follow the fixed radius-one visibility rule. The head and
# rear have one neighbor, interior segments have two, and only the head adds
# the two-value target displacement.
HEAD_OBSERVATION_SIZE = 56
INTERIOR_OBSERVATION_SIZE = 81
REAR_OBSERVATION_SIZE = 54


def build_segment_block(
    snapshot: PhysicalSnapshot,
    segment_id: int,
) -> NDArray[np.float32]:
    """Return one segment's 27 values in the documented fixed order.

    The block contains body pose, leg positions, local body velocities, leg
    velocities, and contact flags. World planar position is deliberately not
    included because it exists only for task and reward calculations.
    """

    # Contact flags follow the physical values so neighboring agents receive
    # the same complete block when this segment is visible to them.
    contacts = np.array(
        [
            snapshot.left_foot_ground_contact[segment_id],
            snapshot.right_foot_ground_contact[segment_id],
            snapshot.body_ground_contact[segment_id],
            snapshot.leg_leg_contact[segment_id],
        ]
    )

    return np.concatenate(
        (
            np.atleast_1d(snapshot.body_height[segment_id]),
            snapshot.body_quaternion[segment_id],
            snapshot.leg_joint_position[segment_id],
            snapshot.body_linear_velocity[segment_id],
            snapshot.body_angular_velocity[segment_id],
            snapshot.leg_joint_velocity[segment_id],
            contacts,
        )
    ).astype(np.float32)


def head_target_displacement(
    snapshot: PhysicalSnapshot,
    target_position: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Express target displacement as forward and lateral head-frame values.

    The calculation uses the head-tip world position and the yaw represented by
    segment zero's quaternion. It must not prescribe how the head should move.
    """

    # World-space planar vector pointing from the head tip to the target.
    delta_x = target_position[0] - snapshot.head_tip_position[0]
    delta_y = target_position[1] - snapshot.head_tip_position[1]

    # MuJoCo stores quaternions scalar-first. Rotating the head's local +x
    # (forward) axis by this quaternion gives the two planar components below.
    w, x, y, z = snapshot.body_quaternion[0]
    forward_world_y = 2.0 * (w * z + x * y)
    forward_world_x = 1.0 - 2.0 * (y * y + z * z)

    # atan2 returns the signed heading angle from world +x to head forward.
    yaw = np.arctan2(forward_world_y, forward_world_x)

    # Project the target vector onto the head's local +x (forward) and +y
    # (left) unit axes, expressed in the world plane using the heading angle.
    forward = np.cos(yaw) * delta_x + np.sin(yaw) * delta_y
    lateral = -np.sin(yaw) * delta_x + np.cos(yaw) * delta_y

    return np.array([forward, lateral], dtype=np.float32)


def build_observations(
    snapshot: PhysicalSnapshot,
    target_position: NDArray[np.float64],
) -> dict[int, NDArray[np.float32]]:
    """Build one partial observation for every segment agent.

    Each observation starts with the agent's own block, followed by its existing
    neighbor ahead and then its existing neighbor behind. Only the head receives
    the final two target-displacement values.
    """

    segment_count = len(snapshot.body_height)

    # Build each segment block once even though adjacent observations reuse it.
    blocks = {
        segment_id: build_segment_block(snapshot, segment_id)
        for segment_id in range(segment_count)
    }

    observations = {}

    for segment_id in range(segment_count):
        parts = [blocks[segment_id]]

        # The preceding segment is ahead
        if segment_id > 0:
            parts.append(blocks[segment_id - 1])

        # The following segment is behind
        if segment_id < segment_count - 1:
            parts.append(blocks[segment_id + 1])

        # Only the head receives navigation information
        if segment_id == 0:
            parts.append(head_target_displacement(snapshot, target_position))

        observations[segment_id] = np.concatenate(parts)

    return observations
