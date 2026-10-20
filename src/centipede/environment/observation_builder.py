"""Every unit's observation (docs/follow-line/observations.md): its own block
and its neighbours', the trail it sees, and its world's tow level."""

import torch

from centipede.environment.line import TRAIL_VALUES
from centipede.environment.simulation import PhysicalState


def block_size(
    leg_motor_count: int, priced_forces: bool = True, carry: bool = False
) -> int:
    """Values in a unit's block, for M leg motors: gravity's direction,
    velocity and turning rate (3 each), leg angles and speeds (M each), the
    spine joint behind (2), the feet's loads (4), the belly's load (1), the
    legs' press (1) and the averaged joint forces (3) if ``priced_forces``,
    the feet's averaged load (1) if ``carry``, the step's joint forces (2),
    and the last action (M + 1): 47 with ankles, 43 without the priced
    forces, 44 with the carry instead."""
    return 19 + 3 * leg_motor_count + (4 if priced_forces else 0) + int(carry)


def neighbour_table(
    segment_count: int, observation_radius: int, device: torch.device | str
) -> torch.Tensor:
    """The blocks each unit sees, ``(N, 2 r + 1)``, in observation order:
    itself, the units ahead (toward the front, lower numbers), nearest first,
    then the units behind. A missing neighbour points to row ``segment_count``
    of the block tensor, which stays zero."""
    rows = []
    for segment_index in range(segment_count):
        ahead = [segment_index - offset for offset in range(1, observation_radius + 1)]
        behind = [segment_index + offset for offset in range(1, observation_radius + 1)]
        seen = [
            index if 0 <= index < segment_count else segment_count
            for index in ahead + behind
        ]
        rows.append([segment_index] + seen)
    return torch.tensor(rows, dtype=torch.long, device=device)


class ObservationBuilder:
    """Builds every unit's observation from the physical state.

    A block, in order: the direction of gravity in the unit's own frame; its
    velocity and its angular velocity in its frame; its leg joints' angles,
    then their speeds; the angle and speed of the spine joint behind it (zero
    for the rear); each foot's force from the floor, pressing down (left,
    right), then along the unit (left, right); the belly's force from the
    floor; the force of its legs pressed by others; the averaged front
    joint's force along the body (positive pulls it forward) and the
    averaged net force of its two joints along the body and up; the step's
    front joint force and net force along the body; and the action it sent
    last (the rear's last value goes to no motor). Without ``priced_forces``
    the legs' press and the averaged joint forces are left out; with
    ``carry`` the feet's averaged load follows the averaged joint forces.
    Forces are in unit weights. The blocks are laid end to end in the
    neighbour table's order, followed by the unit's trail
    (``StraightLine.trail``) and its world's tow level.
    """

    def __init__(
        self,
        segment_count: int,
        observation_radius: int,
        world_count: int,
        leg_motor_count: int,
        unit_weight_n: float,
        device: torch.device | str,
        priced_forces: bool = True,
        carry: bool = False,
    ) -> None:
        """Build the neighbour table and the reusable block tensor, whose last
        row stays zero for missing neighbours."""
        self.segment_count = segment_count
        self.unit_weight_n = unit_weight_n
        self.priced_forces = priced_forces
        self.carry = carry
        self.block_size = block_size(leg_motor_count, priced_forces, carry)
        self.neighbour_table = neighbour_table(
            segment_count, observation_radius, device
        )
        self.blocks = torch.zeros(
            (world_count, segment_count + 1, self.block_size), device=device
        )
        self.observation_size = (
            self.neighbour_table.shape[1] * self.block_size + TRAIL_VALUES + 1
        )

    def build(
        self,
        physical_state: PhysicalState,
        last_action: torch.Tensor,
        averaged_forces: torch.Tensor,
        trail: torch.Tensor,
        tow_levels: torch.Tensor,
    ) -> torch.Tensor:
        """Every unit's observation, ``(W, N, observation_size)``.

        ``last_action`` is the joint action of the step before, ``(W, N, M +
        1)``; ``averaged_forces`` ``AveragedForces.values``; ``trail``
        ``StraightLine.trail``; ``tow_levels`` each world's tow level,
        ``(W,)``. The gather creates a new tensor, so a returned observation
        never changes when ``build`` runs again.
        """
        state = physical_state
        # Gravity, straight down, in the unit's frame: minus the third row of
        # the rotation matrix of its (w, x, y, z) quaternion.
        w, x, y, z = state.body_quaternion.unbind(dim=-1)
        gravity = -torch.stack(
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
            dim=-1,
        )
        loads = [
            state.foot_normal_force,
            state.foot_forward_force,
            state.belly_normal_force[..., None],
        ]
        averaged = []
        if self.priced_forces:
            loads.append(state.leg_leg_force[..., None])
            averaged = [averaged_forces[..., 2:3], averaged_forces[..., :2]]
        if self.carry:
            averaged.append(averaged_forces[..., 5:6])
        step_forces = torch.cat(
            (state.front_joint_force[..., None], state.net_joint_force[..., :1]),
            dim=-1,
        )
        self.blocks[:, : self.segment_count] = torch.cat(
            (
                gravity,
                state.body_linear_velocity,
                state.body_angular_velocity,
                state.leg_joint_position,
                state.leg_joint_velocity,
                state.spine_yaw_position[..., None],
                state.spine_yaw_velocity[..., None],
                torch.cat(loads, dim=-1) / self.unit_weight_n,
                *averaged,
                step_forces / self.unit_weight_n,
                last_action,
            ),
            dim=-1,
        )
        neighbour_blocks = self.blocks[:, self.neighbour_table].flatten(start_dim=2)
        tow = tow_levels[:, None, None].expand(-1, self.segment_count, 1)
        return torch.cat((neighbour_blocks, trail, tow), dim=-1)
