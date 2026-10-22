"""The line each world's body follows, and what each unit sees of it
(docs/follow-line/task.md and observations.md).

At every restart a world's line is laid through the mean of the units'
centres, pointing where the front unit points turned by a given angle, and
it stays for the episode. It runs straight through the body; with curves
(``CurveSettings``) it then bends piece by piece beyond the head, its
curvature never jumping, so it has no corner: a continuous-curvature path
(Fraichard & Scheuer 2004). The line is kept as points a millimetre apart,
each with its direction and curvature. Each unit's place on it is measured
from its centre at the line's nearest point: how far along the line, how far
from it, and its body's angle from the line's direction there. The trail a
unit sees is five points of the line, one unit spacing apart along it,
centred on the line's point nearest its centre, each seen only within the
unit's sight radius.
"""

import math

import torch

from centipede.environment.settings import CurveSettings
from centipede.environment.simulation import PhysicalState

TRAIL_POINTS = 5
# Each trail point: forward and to the left in the unit's frame, and seen.
TRAIL_VALUES = 3 * TRAIL_POINTS
POINT_SPACING_M = 1e-3
# How far the line runs ahead of the body's middle: a 20 s episode at twice
# the commanded speed of a body length a second.
AHEAD_BODY_LENGTHS = 40
# How far either way of its last nearest point a unit's nearest point is
# looked for after a step, in points: a unit moves less than 2 mm a step.
SEARCH_POINTS = 10
# The points a recording draws, from a body length behind the units' middle
# to a body length ahead.
DRAWN_POINTS = 21


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


class Line:
    """Every world's line and every unit's place on it.

    The body is ``segment_count`` units ``spacing_m`` apart long. With
    ``curves`` the line bends at the ``level`` set (``set_level``) when a
    world restarts. After each ``restart`` or ``update``: ``distance``
    ``(W, N)``, each centre's distance from the line (m); ``angle`` ``(W,
    N)``, its body's angle from the line's direction (rad, positive to the
    left); ``curvature`` ``(W, N)``, the line's there (1/m, positive turning
    left); and ``trail`` ``(W, N, 15)``, the trail each unit sees: for each
    point, from the farthest back along the line to the farthest ahead, its
    position forward and to the left in the unit's frame (m) and 1 if it lies
    within ``sight_radius_m`` of the unit's centre seen from above, else
    three zeros.
    """

    def __init__(
        self,
        world_count: int,
        segment_count: int,
        spacing_m: float,
        sight_radius_m: float,
        device: torch.device | str,
        curves: CurveSettings | None = None,
    ) -> None:
        self.sight_radius_m = sight_radius_m
        self.curves = curves
        self.level = 0
        self.body_length_m = segment_count * spacing_m
        # The points run from behind the rear unit's trail; the one at the
        # body's middle lies a whole number of points from the first.
        behind = math.ceil((self.body_length_m + sight_radius_m) / POINT_SPACING_M)
        ahead = math.ceil(AHEAD_BODY_LENGTHS * self.body_length_m / POINT_SPACING_M)
        self.middle_point = behind
        # Each point's place along the line from the body's middle at the
        # restart.
        self.point_along = POINT_SPACING_M * torch.arange(
            -behind, ahead + 1, dtype=torch.float32, device=device
        )
        point_count = self.point_along.shape[0]
        self.points = torch.zeros((world_count, point_count, 2), device=device)
        self.directions = torch.zeros_like(self.points)
        self.curvatures = torch.zeros((world_count, point_count), device=device)
        self.search_offsets = torch.arange(
            -SEARCH_POINTS, SEARCH_POINTS + 1, device=device
        )
        half = TRAIL_POINTS // 2
        self.trail_offsets = spacing_m * torch.arange(
            -half, half + 1, dtype=torch.float32, device=device
        )
        self.drawn_offsets = torch.linspace(
            -self.body_length_m, self.body_length_m, DRAWN_POINTS, device=device
        )
        # Each centre's nearest point and place along the line after the
        # last step.
        self.nearest = torch.zeros(
            (world_count, segment_count), dtype=torch.long, device=device
        )
        self.along = torch.zeros((world_count, segment_count), device=device)
        self.distance = torch.zeros_like(self.along)
        self.angle = torch.zeros_like(self.along)
        self.curvature = torch.zeros_like(self.along)
        self.trail = torch.zeros(
            (world_count, segment_count, TRAIL_VALUES), device=device
        )

    @property
    def level_count(self) -> int:
        """How many levels the curves have; 0 for a straight line."""
        if self.curves is None:
            return 0
        return len(self.curves.level_tightest_radius_body_lengths)

    def set_level(self, level: int) -> None:
        """Lay every line from the next restart on at ``level``."""
        self.level = level

    def restart(
        self,
        world_mask: torch.Tensor,
        state: PhysicalState,
        turn: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> None:
        """Lay the masked worlds' lines through the body as it stands, the
        front unit's heading turned by ``turn`` ``(W,)`` rad to the left, with
        curves drawn from ``generator``; every world is measured again as it
        stands."""
        centres = state.body_planar_position
        origin = centres.mean(dim=1)
        yaw = heading(state.body_quaternion[:, 0]) + turn
        start = torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        # Each centre's place along the straight part through the body.
        along = ((centres - origin[:, None]) * start[:, None]).sum(dim=-1)
        if self.level_count:
            curvatures = self._curvatures(along[:, 0], generator)
        else:
            curvatures = torch.zeros_like(self.curvatures)
        # Each point's heading and position: the curvature turned through and
        # the direction walked over the points before it, from the middle.
        turned = (curvatures.cumsum(dim=1) - curvatures) * POINT_SPACING_M
        headings = yaw[:, None] + turned
        directions = torch.stack((headings.cos(), headings.sin()), dim=-1)
        walked = (directions.cumsum(dim=1) - directions) * POINT_SPACING_M
        points = origin[:, None] + walked - walked[:, self.middle_point, None]

        mask = world_mask[:, None]
        self.points.copy_(torch.where(mask[..., None], points, self.points))
        self.directions.copy_(torch.where(mask[..., None], directions, self.directions))
        self.curvatures.copy_(torch.where(mask, curvatures, self.curvatures))
        nearest = (along / POINT_SPACING_M).round().long() + self.middle_point
        self.nearest.copy_(torch.where(mask, nearest, self.nearest))
        self.along.copy_(self._measure(state))

    def update(self, state: PhysicalState) -> torch.Tensor:
        """Measure every centre; return how far it went along the line on the
        step, ``(W, N)`` m, negative backward."""
        along = self._measure(state)
        progress = along - self.along
        self.along.copy_(along)
        return progress

    def drawn_points(self) -> torch.Tensor:
        """The line from a body length behind the units' middle to a body
        length ahead, ``(W, 21, 2)`` world x and y, for recordings."""
        middle = self.along.mean(dim=1, keepdim=True)
        return self._points_at(middle + self.drawn_offsets)

    def _curvatures(
        self, head_along: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Every world's curvature at each point, ``(W, P)`` 1/m: none up to
        the lead beyond the head at ``head_along`` ``(W,)``, then piece by
        piece at the level."""
        curves = self.curves
        body = self.body_length_m
        tightest = 1 / (curves.level_tightest_radius_body_lengths[self.level] * body)
        straight_share = curves.level_straight_shares[self.level]
        shortest, longest = curves.piece_body_lengths
        world_count = head_along.shape[0]
        piece_count = math.ceil(AHEAD_BODY_LENGTHS / shortest) + 1
        draws = torch.rand(
            (world_count, piece_count, 3),
            generator=generator,
            device=head_along.device,
        )
        lengths = (shortest + (longest - shortest) * draws[..., 0]) * body
        bends = torch.where(
            draws[..., 1] < straight_share, 0.0, (2 * draws[..., 2] - 1) * tightest
        )
        first = head_along + curves.lead_body_lengths * body
        starts = first[:, None] + lengths.cumsum(dim=1) - lengths
        before = torch.cat((torch.zeros_like(bends[:, :1]), bends[:, :-1]), dim=1)
        along = self.point_along.expand(world_count, -1).contiguous()
        piece = torch.searchsorted(starts, along, right=True) - 1
        index = piece.clamp(min=0)
        changed = (
            (along - starts.gather(1, index)) / (curves.bend_change_body_lengths * body)
        ).clamp(0.0, 1.0)
        was = before.gather(1, index)
        curvature = was + (bends.gather(1, index) - was) * changed
        return torch.where(piece >= 0, curvature, 0.0)

    def _measure(self, state: PhysicalState) -> torch.Tensor:
        """Find each centre's nearest point near the last one and refresh
        ``distance``, ``angle``, ``curvature`` and ``trail``; return each
        centre's place along the line, ``(W, N)`` m."""
        centre = state.body_planar_position
        last_point = self.point_along.shape[0] - 1
        window = (self.nearest[..., None] + self.search_offsets).clamp(0, last_point)
        gaps = (_gather(self.points, window) - centre[:, :, None]).square().sum(dim=-1)
        nearest = window.gather(-1, gaps.argmin(dim=-1, keepdim=True)).squeeze(-1)
        self.nearest.copy_(nearest)

        point = _gather(self.points, nearest)
        direction = _gather(self.directions, nearest)
        relative = centre - point
        along = self.point_along[nearest] + (relative * direction).sum(dim=-1)
        yaw = heading(state.body_quaternion)
        turned = yaw - torch.atan2(direction[..., 1], direction[..., 0])
        self.distance.copy_(to_the_left(direction, relative).abs())
        self.angle.copy_(torch.remainder(turned + math.pi, 2 * math.pi) - math.pi)
        self.curvature.copy_(self.curvatures.gather(1, nearest))

        # The trail's points from each centre, (W, N, 5, 2), in the unit's frame.
        trail_points = self._points_at(along[..., None] + self.trail_offsets)
        seen_from = trail_points - centre[:, :, None]
        forward = torch.stack((yaw.cos(), yaw.sin()), dim=-1)[:, :, None]
        ahead = (seen_from * forward).sum(dim=-1)
        to_left = to_the_left(forward, seen_from)
        seen = (seen_from.norm(dim=-1) <= self.sight_radius_m).float()
        trail = torch.stack((ahead * seen, to_left * seen, seen), dim=-1)
        self.trail.copy_(trail.flatten(start_dim=2))
        return along

    def _points_at(self, along: torch.Tensor) -> torch.Tensor:
        """The line's positions at the places ``along`` it ``(W, ...)`` m,
        between its stored points: ``(W, ..., 2)``."""
        place = (along - self.point_along[0]) / POINT_SPACING_M
        place = place.clamp(0.0, self.point_along.shape[0] - 1.001)
        first = place.floor().long()
        share = (place - first)[..., None]
        return (1 - share) * _gather(self.points, first) + share * _gather(
            self.points, first + 1
        )


def _gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Each world's ``values`` ``(W, P, 2)`` at the points ``index`` ``(W,
    ...)``: ``(W, ..., 2)``."""
    flat = index.reshape(index.shape[0], -1, 1).expand(-1, -1, values.shape[-1])
    return values.gather(1, flat).reshape(*index.shape, values.shape[-1])
