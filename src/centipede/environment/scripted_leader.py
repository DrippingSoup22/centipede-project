"""Stage A's leader, a script (docs/follow-footsteps/task.md, stage A's
leader): unit 1's footholds laid along each world's line, which nobody
sees; it only places the footholds. The environment asks for the first
footholds at a restart and for the next at each touchdown of unit 1's feet
(``Footprints``)."""

import torch

from centipede.environment.line import Line

# How far either way of a place on the line its direction is measured, m.
DIRECTION_SPAN_M = 5e-4


class ScriptedLeader:
    """Each side's next foothold lies ``stride_m`` along the line beyond that
    side's last, ``reach_m`` to that side of it, at the height unit 1's feet
    stood at at the restart (flat ground). The right side's footholds run
    half a stride behind the left's, as a centipede's left and right legs
    step in turn. A foothold follows the one given before, not where the
    foot landed, so a missed step does not bend the path. ``along`` ``(W,
    2)`` holds each side's last foothold's place along the line, m.
    """

    def __init__(
        self, world_count: int, stride_m: float, reach_m: float, device
    ) -> None:
        self.stride_m = stride_m
        self.reach_m = reach_m
        self.along = torch.zeros(world_count, 2, device=device)
        self.height = torch.zeros(world_count, device=device)
        # How far each side's first foothold lies beyond unit 1's centre,
        # and to which side of the line, in strides and reaches.
        self.first_strides = torch.tensor([1.0, 0.5], device=device)
        self.sides = torch.tensor([1.0, -1.0], device=device)

    def restart(
        self, world_mask: torch.Tensor, line: Line, foot_height: torch.Tensor
    ) -> torch.Tensor:
        """Lay the ``(W,)`` masked worlds' first footholds, a stride beyond
        unit 1's centre on the line on the left, half a stride on the right,
        at ``foot_height`` ``(W,)`` m; returns every world's footholds,
        ``(W, 2, 3)``."""
        first = line.along[:, :1] + self.stride_m * self.first_strides
        self.along[world_mask] = first[world_mask]
        self.height[world_mask] = foot_height[world_mask]
        return self._footholds(line)

    def next_footholds(self, given: torch.Tensor, line: Line) -> torch.Tensor:
        """Move each side whose foot of unit 1 touched down, ``given`` ``(W,
        2)``, a stride on; returns every world's footholds, ``(W, 2, 3)``."""
        self.along += self.stride_m * given
        return self._footholds(line)

    def _footholds(self, line: Line) -> torch.Tensor:
        points = line.points_at(self.along)
        forward = line.points_at(self.along + DIRECTION_SPAN_M) - line.points_at(
            self.along - DIRECTION_SPAN_M
        )
        forward = forward / forward.norm(dim=-1, keepdim=True)
        left = torch.stack((-forward[..., 1], forward[..., 0]), dim=-1)
        planar = points + self.reach_m * self.sides[:, None] * left
        height = self.height[:, None, None].expand(-1, 2, 1)
        return torch.cat((planar, height), dim=-1)
