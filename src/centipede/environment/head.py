"""The head's geometry on the ground: where it points, and where a target lies
from it. Tasks, observations and diagnostics share it."""

import torch

from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.model_mapping import HeadOutline


def head_forward_direction(head_quaternion: torch.Tensor) -> torch.Tensor:
    """The head's forward direction on the ground, as unit vectors ``(W, 2)``.

    ``head_quaternion`` is ``(W, 4)`` in MuJoCo's ``(w, x, y, z)`` order. The
    head's forward axis is its body's x axis; turned into the world, its flat
    ``x, y`` components are the first column of the rotation matrix. They are
    scaled to length 1; the length is kept above a small minimum so that a head
    pointing straight up or down gives a finite direction instead of NaN.
    """
    w, x, y, z = head_quaternion.unbind(dim=-1)
    forward = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z)), dim=-1)
    length = forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return forward / length


def target_seen_from_head(
    physical_state: PhysicalState, target_position: torch.Tensor
) -> torch.Tensor:
    """Each world's flat target ``(W, 2)`` from the head's tip, ``(W, 2)``:
    along the head's forward direction, and along its left (forward turned 90
    degrees anticlockwise)."""
    offset = target_position - physical_state.head_tip_position[:, :2]
    forward = head_forward_direction(physical_state.body_quaternion[:, 0])
    left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    along_forward = (offset * forward).sum(dim=-1)
    along_left = (offset * left).sum(dim=-1)
    return torch.stack((along_forward, along_left), dim=-1)


def target_under_head(
    physical_state: PhysicalState, target_position: torch.Tensor, outline: HeadOutline
) -> torch.Tensor:
    """Worlds whose target lies under the head's outline, seen from above, ``(W,)``.

    The target is measured from the head's centre along the head's forward
    direction and to its side, and compared with the rectangle that encloses
    the head's body shape.
    """
    offset = target_position - physical_state.body_planar_position[:, 0]
    forward = head_forward_direction(physical_state.body_quaternion[:, 0])
    along = (offset * forward).sum(dim=-1)
    across = forward[:, 0] * offset[:, 1] - forward[:, 1] * offset[:, 0]
    return (
        (along >= outline.rear_m)
        & (along <= outline.front_m)
        & (across.abs() <= outline.half_width_m)
    )
