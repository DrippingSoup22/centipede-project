"""Every walker's observation following footsteps
(docs/follow-footsteps/observations.md): the walker itself, its leader, the
unit ahead, its marks and its command, and nothing else. The environment
builds it in place of follow-line's blocks."""

import torch

from centipede.environment.line import heading
from centipede.environment.simulation import PhysicalState


def walker_observation_size(leg_motor_count: int) -> int:
    """Values in a walker's observation, for M leg motors: 3 M + 41, 65 on
    v6's legs."""
    return 3 * leg_motor_count + 41


def from_leader(values: torch.Tensor) -> torch.Tensor:
    """Each walker's leader's ``values`` ``(W, N, ...)``: the unit ahead's,
    zeros for unit 1."""
    return torch.cat((torch.zeros_like(values[:, :1]), values[:, :-1]), dim=1)


def in_walker_frame(points: torch.Tensor, state: PhysicalState) -> torch.Tensor:
    """``points`` ``(W, N, K, 3)`` world m seen from each walker's centre,
    level and turning with its heading: forward, to the left and up."""
    centre = torch.cat(
        (state.body_planar_position, state.body_height[..., None]), dim=-1
    )
    relative = points - centre[:, :, None]
    yaw = heading(state.body_quaternion)[..., None]
    forward = relative[..., 0] * yaw.cos() + relative[..., 1] * yaw.sin()
    left = relative[..., 1] * yaw.cos() - relative[..., 0] * yaw.sin()
    return torch.stack((forward, left, relative[..., 2]), dim=-1)


def walker_observations(
    state: PhysicalState,
    marks: torch.Tensor,
    mark_distances: torch.Tensor,
    feet_down: torch.Tensor,
    held_command: torch.Tensor,
    averaged_command: torch.Tensor,
    averaged_speed: torch.Tensor,
    averaged_forces: torch.Tensor,
    averaged_energy: torch.Tensor,
    last_action: torch.Tensor,
    tow_levels: torch.Tensor,
    unit_weight_n: float,
) -> torch.Tensor:
    """Every walker's observation, ``(W, N, 3 M + 41)``, in the order of
    observations.md's table: its leg angles, then their speeds; its feet
    pressing down (left, right); gravity's direction in its frame; the angle
    and speed of the spine joint it commands; its next mark on each side,
    the newer of ``marks`` ``(W, N, 2, 2, 3)``, forward, left and up in its
    frame; each foot's distance to its nearer mark, ``mark_distances``
    ``(W, N, 2)`` m; the command it holds, averaged (``averaged_command``
    ``(W, N)`` mm/s); its averaged speed along its heading (mm/s), feet's
    loads (``AveragedForces.values``) and energy (mJ/s); its leader's
    averaged energy; its leader's feet, where in its frame and whether down
    (``feet_down`` ``(W, N, 2)``); the command it holds as it stands,
    ``held_command`` ``(W, N)``; its feet pushing fore or aft; its velocity
    and turning speed in its frame and its belly's load; its last action
    ``(W, N, M + 1)``; and its world's tow level ``(W,)``. Forces are in
    unit weights. A leader's values are zeros for unit 1, whose leader is
    the script (later the head).
    """
    w, x, y, z = state.body_quaternion.unbind(dim=-1)
    gravity = -torch.stack(
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        dim=-1,
    )
    next_marks = in_walker_frame(marks[..., 1, :], state).flatten(start_dim=2)
    leader_feet = in_walker_frame(from_leader(state.foot_position), state)
    leader_feet = leader_feet.flatten(start_dim=2)
    # Unit 1 has no leader's feet, not feet at the world's origin.
    leader_feet[:, 0] = 0.0
    unit = unit_weight_n
    worlds = state.body_quaternion.shape[0]
    return torch.cat(
        (
            state.leg_joint_position,
            state.leg_joint_velocity,
            state.foot_vertical_force / unit,
            gravity,
            state.spine_yaw_position[..., None],
            state.spine_yaw_velocity[..., None],
            next_marks,
            mark_distances,
            averaged_command[..., None],
            averaged_speed[..., None],
            averaged_forces[..., 6:8],
            averaged_energy[..., None],
            from_leader(averaged_energy)[..., None],
            leader_feet,
            from_leader(feet_down),
            held_command[..., None],
            state.foot_forward_force / unit,
            state.body_linear_velocity,
            state.body_angular_velocity,
            state.belly_vertical_force[..., None] / unit,
            last_action,
            tow_levels[:, None, None].expand(worlds, state.body_height.shape[1], 1),
        ),
        dim=-1,
    )
