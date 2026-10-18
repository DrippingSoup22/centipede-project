"""Every segment's observation in the cooperators' design.

Each segment senses a block of values about itself; it sees its own block,
those of the segments within the observation radius, and, for the head only,
the target. The layout is fixed in docs/cooperators/observations.md.
"""

import torch

from centipede.environment.head import target_seen_from_head
from centipede.environment.observation_builder import (
    TARGET_VALUE_COUNT,
    neighbour_table,
)
from centipede.environment.simulation import PhysicalState


def block_size(leg_motor_count: int, instant_joint_forces: bool = False) -> int:
    """Values in a segment's block, for M leg motors: gravity's direction,
    velocity and turning rate (3 each), leg angles and speeds (M each), the
    spine joint behind (2), the feet's loads (4), the belly's load (1), the
    legs pressed (1), the front joint's force (1), the net joint force (2),
    with ``instant_joint_forces`` both along the body again, unaveraged (2),
    its place on the head's line (3), and the last action (M + 1)."""
    return 24 + 3 * leg_motor_count + (2 if instant_joint_forces else 0)


class CooperativeObservationBuilder:
    """Builds every segment's observation from the physical state.

    A block, in order: the direction of gravity in the segment's own frame;
    its velocity and its angular velocity in its frame; its leg joints'
    angles, then their speeds; the angle and speed of the spine joint behind
    it (zero for the rear); each foot's force from the floor, pressing down
    (left, right), then along the segment (left, right); the belly's force
    from the floor; the force of its legs pressed by others; the force of
    the segment ahead on it, along the body (positive pulls it, negative is
    it pushing; zero for the head), so that the blocks seen hold the joints
    around it; the net force of its two joints on it, along the body and up;
    its tail's place on the head's line (``HeadLine.view``): its signed
    distance from the line, its body's angle from the line, and its distance
    from the line it has not erased; and the action it sent last (the rear's
    last value is padding). Forces are in unit weights, the whole body's
    weight shared by its segments; the joint forces are the averaged ones
    the reward prices; with ``instant_joint_forces`` the front joint's and
    the net force along the body of the step itself follow them, so that a
    segment feels a pull at once. The blocks are laid end to end in the
    neighbour table's order, a missing neighbour's all zeros, followed by the
    target seen from the head's tip, forward and to the left, which is zero
    for every other segment and, without ``target_input`` (the straight line
    of cycle 18), for the head too, and, with ``tow_input``, the world's tow
    level.
    """

    def __init__(
        self,
        segment_count: int,
        observation_radius: int,
        world_count: int,
        leg_motor_count: int,
        unit_weight_n: float,
        device: torch.device | str,
        instant_joint_forces: bool = False,
        tow_input: bool = False,
        target_input: bool = True,
    ) -> None:
        """Build the neighbour table and the reusable block tensor, whose last
        row stays zero for missing neighbours."""
        self.segment_count = segment_count
        self.unit_weight_n = unit_weight_n
        self.instant_joint_forces = instant_joint_forces
        self.tow_input = tow_input
        self.target_input = target_input
        self.block_size = block_size(leg_motor_count, instant_joint_forces)
        self.neighbour_table = neighbour_table(
            segment_count, observation_radius, device
        )
        self.blocks = torch.zeros(
            (world_count, segment_count + 1, self.block_size),
            dtype=torch.float32,
            device=device,
        )
        self.observation_size = (
            self.neighbour_table.shape[1] * self.block_size
            + TARGET_VALUE_COUNT
            + (1 if tow_input else 0)
        )

    def build(
        self,
        physical_state: PhysicalState,
        target_position: torch.Tensor,
        last_action: torch.Tensor,
        line_view: torch.Tensor,
        averaged_forces: torch.Tensor,
        tow_levels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Every segment's observation, ``(W, N, observation_size)``.

        ``target_position`` is each world's flat target, ``(W, 2)``,
        ``last_action`` the joint action of the step before, ``(W, N, M + 1)``,
        ``line_view`` ``HeadLine.view``, ``(W, N, 3)``, ``averaged_forces``
        ``AveragedForces.values``, and ``tow_levels`` each world's tow level,
        ``(W,)``, read with ``tow_input``. The gather
        creates a new tensor, so a returned observation never changes when
        ``build`` runs again.
        """
        state = physical_state
        # Gravity, straight down, in the segment's frame: minus the third row
        # of the rotation matrix of its (w, x, y, z) quaternion.
        w, x, y, z = state.body_quaternion.unbind(dim=-1)
        gravity = -torch.stack(
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
            dim=-1,
        )
        loads = torch.cat(
            (
                state.foot_normal_force,
                state.foot_forward_force,
                state.belly_normal_force[..., None],
                state.leg_leg_force[..., None],
            ),
            dim=-1,
        )
        forces = [
            loads / self.unit_weight_n,
            averaged_forces[..., 2:3],
            averaged_forces[..., :2],
        ]
        if self.instant_joint_forces:
            forces.append(
                torch.stack(
                    (state.front_joint_force, state.net_joint_force[..., 0]), dim=-1
                )
                / self.unit_weight_n
            )
        forces = torch.cat(forces, dim=-1)
        self.blocks[:, : self.segment_count] = torch.cat(
            (
                gravity,
                state.body_linear_velocity,
                state.body_angular_velocity,
                state.leg_joint_position,
                state.leg_joint_velocity,
                state.spine_yaw_position[..., None],
                state.spine_yaw_velocity[..., None],
                forces,
                line_view,
                last_action,
            ),
            dim=-1,
        )
        neighbour_blocks = self.blocks[:, self.neighbour_table].flatten(start_dim=2)
        target_values = neighbour_blocks.new_zeros(
            (*neighbour_blocks.shape[:2], TARGET_VALUE_COUNT)
        )
        if self.target_input:
            target_values[:, 0] = target_seen_from_head(state, target_position)
        inputs = [neighbour_blocks, target_values]
        if self.tow_input:
            inputs.append(tow_levels[:, None, None].expand(-1, self.segment_count, 1))
        return torch.cat(inputs, dim=-1)
