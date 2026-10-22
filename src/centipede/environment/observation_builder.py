"""Every unit's observation (docs/follow-line/observations.md): its own block
and its neighbours', the trail it sees, and its world's tow level."""

import torch

from centipede.environment.line import TRAIL_VALUES
from centipede.environment.simulation import PhysicalState


def block_size(
    leg_motor_count: int,
    priced_forces: bool = True,
    carry: bool = False,
    energy: bool = False,
    line_speed: bool = False,
    carry_per_leg: bool = False,
) -> int:
    """Values in a unit's block, for M leg motors: gravity's direction,
    velocity and turning rate (3 each), leg angles and speeds (M each), the
    spine joint behind (2), the feet's loads (4), the belly's load (1), the
    legs' press (1) and the averaged joint forces (3) if ``priced_forces``,
    the feet's averaged load (1, or each foot's, 2, with ``carry_per_leg``)
    if ``carry``, the averaged energy (1) if
    ``energy``, the averaged forward line (1) if ``line_speed``, the step's
    joint forces (2), and the last action (M + 1): 47 with ankles, 43 without
    the priced forces, 44 with the carry instead."""
    return (
        19
        + 3 * leg_motor_count
        + (4 if priced_forces else 0)
        + int(carry) * (2 if carry_per_leg else 1)
        + int(energy)
        + int(line_speed)
    )


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
    ``carry`` the feet's averaged load follows the averaged joint forces
    (each foot's, left then right, with ``carry_per_leg``),
    with ``energy`` the unit's averaged energy in mJ/s follows them, and
    with ``line_speed`` its averaged forward line in mm/s.
    Forces are in unit weights. The blocks are laid end to end in the
    neighbour table's order, followed by the unit's trail
    (``Line.trail``) and its world's tow level, and with ``command`` its
    world's averaged commanded speed in mm/s. ``behind_entries``
    marks the entries of the units behind, ``(observation_size,)``.
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
        energy: bool = False,
        line_speed: bool = False,
        carry_per_leg: bool = False,
        command: bool = False,
    ) -> None:
        """Build the neighbour table and the reusable block tensor, whose last
        row stays zero for missing neighbours."""
        self.segment_count = segment_count
        self.unit_weight_n = unit_weight_n
        self.priced_forces = priced_forces
        self.carry = carry
        self.energy = energy
        self.line_speed = line_speed
        self.carry_per_leg = carry_per_leg
        self.command = command
        self.block_size = block_size(
            leg_motor_count, priced_forces, carry, energy, line_speed, carry_per_leg
        )
        self.neighbour_table = neighbour_table(
            segment_count, observation_radius, device
        )
        self.blocks = torch.zeros(
            (world_count, segment_count + 1, self.block_size), device=device
        )
        self.observation_size = (
            self.neighbour_table.shape[1] * self.block_size
            + TRAIL_VALUES
            + 1
            + int(command)
        )
        # The units behind are the table's last observation_radius blocks.
        self.behind_entries = torch.zeros(
            self.observation_size, dtype=torch.bool, device=device
        )
        first_behind = (1 + observation_radius) * self.block_size
        self.behind_entries[
            first_behind : first_behind + observation_radius * self.block_size
        ] = True

    def build(
        self,
        physical_state: PhysicalState,
        last_action: torch.Tensor,
        averaged_forces: torch.Tensor,
        trail: torch.Tensor,
        tow_levels: torch.Tensor,
        averaged_energy: torch.Tensor | None = None,
        averaged_forward_line: torch.Tensor | None = None,
        averaged_command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Every unit's observation, ``(W, N, observation_size)``.

        ``last_action`` is the joint action of the step before, ``(W, N, M +
        1)``; ``averaged_forces`` ``AveragedForces.values``; ``trail``
        ``Line.trail``; ``tow_levels`` each world's tow level,
        ``(W,)``; ``averaged_energy`` each unit's averaged energy, ``(W, N)``
        mJ/s, read with ``energy``; ``averaged_forward_line`` each unit's
        averaged forward line, ``(W, N)`` mm/s, read with ``line_speed``;
        ``averaged_command`` each world's averaged commanded speed, ``(W,)``
        mm/s, read with ``command``. The gather creates a new tensor, so a
        returned observation never changes when ``build`` runs again.
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
            feet = slice(6, 8) if self.carry_per_leg else slice(5, 6)
            averaged.append(averaged_forces[..., feet])
        if self.energy:
            averaged.append(averaged_energy[..., None])
        if self.line_speed:
            averaged.append(averaged_forward_line[..., None])
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
        world_values = [tow_levels]
        if self.command:
            world_values.append(averaged_command)
        worlds = torch.stack(world_values, dim=-1)[:, None]
        return torch.cat(
            (neighbour_blocks, trail, worlds.expand(-1, self.segment_count, -1)),
            dim=-1,
        )
