"""The straight line each world's body follows, and what each unit sees of it
(docs/follow-line/task.md and observations.md).

At every restart a world's line is laid through the mean of the units'
centres, pointing where the front unit points turned by a given angle, and
it stays for the episode. Each unit's place on it is measured from its
centre: how far along the line, how far from it, and its body's angle from
the line's direction. The trail a unit sees is five points of the line, one
unit spacing apart, centred on the line's point nearest its centre, each
seen only within the unit's sight radius.
"""

import math

import torch

from centipede.environment.simulation import PhysicalState

TRAIL_POINTS = 5
# Each trail point: forward and to the left in the unit's frame, and seen.
TRAIL_VALUES = 3 * TRAIL_POINTS


def heading(body_quaternion: torch.Tensor) -> torch.Tensor:
    """The yaw of each body's forward axis from its (w, x, y, z) quaternion."""
    w, x, y, z = body_quaternion.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_the_left(axis: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """How far ``vector`` reaches to the left of the unit ``axis``, both in
    the plane with x and y last."""
    return axis[..., 0] * vector[..., 1] - axis[..., 1] * vector[..., 0]


def forward_axis(state: PhysicalState) -> torch.Tensor:
    """Each unit's forward axis in the plane, ``(W, N, 2)``."""
    yaw = heading(state.body_quaternion)
    return torch.stack((yaw.cos(), yaw.sin()), dim=-1)


class StraightLine:
    """Every world's line and every unit's place on it.

    After each ``restart`` or ``update``: ``distance`` ``(W, N)``, each
    centre's distance from the line (m); ``angle`` ``(W, N)``, its body's
    angle from the line's direction (rad, positive to the left); and
    ``trail`` ``(W, N, 15)``, the trail each unit sees: for each point, from
    the farthest back along the line to the farthest ahead, its position
    forward and to the left in the unit's frame (m) and 1 if it lies within
    ``sight_radius_m`` of the unit's centre seen from above, else three
    zeros.
    """

    def __init__(
        self,
        world_count: int,
        segment_count: int,
        spacing_m: float,
        sight_radius_m: float,
        device: torch.device | str,
    ) -> None:
        self.sight_radius_m = sight_radius_m
        half = TRAIL_POINTS // 2
        self.trail_offsets = spacing_m * torch.arange(
            -half, half + 1, dtype=torch.float32, device=device
        )
        self.origin = torch.zeros((world_count, 2), device=device)
        self.direction = torch.zeros((world_count, 2), device=device)
        # Each centre's place along the line after the last step.
        self.along = torch.zeros((world_count, segment_count), device=device)
        self.distance = torch.zeros_like(self.along)
        self.angle = torch.zeros_like(self.along)
        self.trail = torch.zeros(
            (world_count, segment_count, TRAIL_VALUES), device=device
        )

    def restart(
        self, world_mask: torch.Tensor, state: PhysicalState, turn: torch.Tensor
    ) -> None:
        """Lay the masked worlds' lines through the body as it stands, the
        front unit's heading turned by ``turn`` ``(W,)`` rad to the left;
        every world is measured again as it stands."""
        yaw = heading(state.body_quaternion[:, 0]) + turn
        direction = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        mask = world_mask[:, None]
        self.origin.copy_(
            torch.where(mask, state.body_planar_position.mean(dim=1), self.origin)
        )
        self.direction.copy_(torch.where(mask, direction, self.direction))
        self.along.copy_(self._measure(state))

    def update(self, state: PhysicalState) -> torch.Tensor:
        """Measure every centre; return how far it went along the line on the
        step, ``(W, N)`` m, negative backward."""
        along = self._measure(state)
        progress = along - self.along
        self.along.copy_(along)
        return progress

    def _measure(self, state: PhysicalState) -> torch.Tensor:
        """Refresh ``distance``, ``angle`` and ``trail``; return each centre's
        place along the line, ``(W, N)`` m."""
        centre = state.body_planar_position
        direction = self.direction[:, None]
        relative = centre - self.origin[:, None]
        along = (relative * direction).sum(dim=-1)
        yaw = heading(state.body_quaternion)
        turned = yaw - torch.atan2(direction[..., 1], direction[..., 0])
        self.distance.copy_(to_the_left(direction, relative).abs())
        self.angle.copy_(torch.remainder(turned + math.pi, 2 * math.pi) - math.pi)

        # The trail's points from each centre, (W, N, 5, 2), in the unit's frame.
        nearest = self.origin[:, None] + along[..., None] * direction
        offsets = self.trail_offsets[:, None] * direction[:, :, None]
        seen_from = nearest[:, :, None] + offsets - centre[:, :, None]
        forward = torch.stack((yaw.cos(), yaw.sin()), dim=-1)[:, :, None]
        ahead = (seen_from * forward).sum(dim=-1)
        to_left = to_the_left(forward, seen_from)
        seen = (seen_from.norm(dim=-1) <= self.sight_radius_m).float()
        trail = torch.stack((ahead * seen, to_left * seen, seen), dim=-1)
        self.trail.copy_(trail.flatten(start_dim=2))
        return along
