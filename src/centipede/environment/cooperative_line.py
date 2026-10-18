"""The line the head's tip draws on the floor, and each segment's place on it
(docs/cooperators/reward.md, cycle 7).

Every segment has its own copy of the line, which its tail, its rear end,
erases as it passes: a copy runs from the furthest point its tail has
reached to the tip. A follower is paid for the line its tail erases and
charged for how far its tail lies from the line it has not erased yet,
beside it or behind its end. The line is kept as points, one each time the
tip has moved SPACING_M, with their arc lengths along the line; a world's
line starts at each restart as the body's own centreline.

With a drift (cycle 15) the line goes where the head keeps pointing: the
tip draws it to the side by a drift that grows with the head's sideslip,
the angle between where the head points and where its centre goes, both
averaged over a step cycle, and leaks away. A body that follows the head
leaves it none, so the line is the tip's path; a body that ignores a bend
the head holds leaves the head slipping sideways, and the line moves away
from the body the longer it does.

The straight line (cycle 18) is a fixed line through the body's start pose,
which every segment's centre walks along; it offers the same ``restart``,
``update`` and ``view``.
"""

import math

import torch

from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.constants import ACTION_DURATION_S

# The tip adds a point once it has moved this far from the last one, so a
# head standing still draws nothing.
SPACING_M = 0.0005
# A tail looks for itself this many pieces either way of where it was last.
WINDOW_PIECES = 64
# The head's pointing and its centre's travel are averaged over a step cycle
# (0.4 s at the walk's 2.5 Hz), so that the wag of each step cancels.
POINTING_AVERAGE_S = 0.4


def heading(body_quaternion: torch.Tensor) -> torch.Tensor:
    """The yaw of each segment's forward axis from its (w, x, y, z) quaternion."""
    w, x, y, z = body_quaternion.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def forward_axis(state: PhysicalState) -> torch.Tensor:
    """Each segment's forward axis in the plane, ``(W, N, 2)``."""
    yaw = heading(state.body_quaternion)
    return torch.stack((yaw.cos(), yaw.sin()), dim=-1)


class HeadSideslip:
    """The head's sideslip in every world, ``sideslip`` ``(W,)``: the angle
    between where it points and where its centre goes, both averaged over a
    step cycle (rad, positive when it points left of where it goes)."""

    def __init__(self, world_count: int, device: torch.device | str) -> None:
        self.pointing = torch.zeros((world_count, 2), device=device)
        self.travel = torch.zeros((world_count, 2), device=device)
        self.last_centre = torch.zeros((world_count, 2), device=device)
        self.sideslip = torch.zeros(world_count, device=device)

    def restart(self, world_mask: torch.Tensor, state: PhysicalState) -> None:
        """Start the masked worlds pointing as the head does, not moving."""
        mask = world_mask[:, None]
        self.pointing.copy_(torch.where(mask, forward_axis(state)[:, 0], self.pointing))
        self.travel.masked_fill_(mask, 0.0)
        self.last_centre.copy_(
            torch.where(mask, state.body_planar_position[:, 0], self.last_centre)
        )
        self.sideslip.masked_fill_(world_mask, 0.0)

    def update(self, state: PhysicalState) -> tuple[torch.Tensor, torch.Tensor]:
        """Average in the step; return the head's averaged speed ``(W,)`` and
        the direction it goes, ``(W, 2)``."""
        rate = ACTION_DURATION_S / POINTING_AVERAGE_S
        self.pointing += rate * (forward_axis(state)[:, 0] - self.pointing)
        centre = state.body_planar_position[:, 0]
        self.travel += rate * ((centre - self.last_centre) - self.travel)
        self.last_centre.copy_(centre)
        speed = self.travel.norm(dim=-1)
        going = self.travel / speed.clamp(min=1e-12)[:, None]
        pointing = self.pointing / self.pointing.norm(dim=-1, keepdim=True).clamp(
            min=1e-12
        )
        sideways = going[:, 0] * pointing[:, 1] - going[:, 1] * pointing[:, 0]
        self.sideslip.copy_(torch.asin(sideways.clamp(-1.0, 1.0)))
        return speed, going


class HeadLine:
    """Every world's line and every segment's place on it.

    After each ``update``, ``view`` ``(W, N, 3)`` holds for each segment's
    tail: its signed distance from the line (m, positive to the left of the
    line's direction), the angle of its body from the line's direction
    there (rad, positive to the left), and its distance from the line it has
    not erased (m). The head's own copy, erased by its neck, is measured too.
    ``sideslip`` ``(W,)`` holds the head's averaged sideslip (rad, positive
    when it points left of where it goes) and ``drift`` ``(W,)`` how far to
    its left the tip draws (m; zero without a drift).
    """

    def __init__(
        self,
        world_count: int,
        segment_lengths_m: tuple[float, ...],
        episode_steps: int,
        device: torch.device | str,
        drift_leak_s: float = 0.0,
    ) -> None:
        """Room for the starting centreline and one point per step of an
        episode, the most the tip can add before the world restarts. With
        ``drift_leak_s`` the line drifts with the head's sideslip, the drift
        leaking away over that time; 0 draws it at the tip."""
        self.drift_leak_s = drift_leak_s
        self.segment_count = segment_count = len(segment_lengths_m)
        self.rear_length_m = segment_lengths_m[-1]
        # The centreline's pieces, rear first: each segment from its tail to
        # its front end, cut into parts at most SPACING_M long.
        parts = [math.ceil(length / SPACING_M) for length in segment_lengths_m[::-1]]
        self.start_count = sum(parts) + 1
        self.piece_of_point = torch.repeat_interleave(
            torch.arange(segment_count), torch.tensor(parts)
        ).to(device)
        self.share_of_point = torch.cat(
            [torch.arange(part) / part for part in parts]
        ).to(device)
        # The point where each segment's tail lies on the centreline.
        corner = torch.cumsum(torch.tensor([0, *parts]), dim=0)
        self.tail_point = corner.flip(0)[1:].to(device)
        capacity = self.start_count + episode_steps + 1
        self.points = torch.zeros((world_count, capacity, 2), device=device)
        self.arc = torch.zeros((world_count, capacity), device=device)
        self.count = torch.zeros(world_count, dtype=torch.long, device=device)
        self.nearest_piece = torch.zeros(
            (world_count, segment_count), dtype=torch.long, device=device
        )
        self.erased = torch.zeros((world_count, segment_count), device=device)
        self.erased_point = torch.zeros((world_count, segment_count, 2), device=device)
        self.view = torch.zeros((world_count, segment_count, 3), device=device)
        self.window = torch.arange(-WINDOW_PIECES, WINDOW_PIECES, device=device)
        self.head = HeadSideslip(world_count, device)
        self.sideslip = self.head.sideslip
        self.drift = torch.zeros(world_count, device=device)

    def tails(self, state: PhysicalState) -> torch.Tensor:
        """Each segment's rear end, ``(W, N, 2)``: the joint behind it, and
        for the rear its front end one segment length back along its body."""
        front = state.front_end_position
        rear = front[:, -1] - self.rear_length_m * forward_axis(state)[:, -1]
        return torch.cat((front[:, 1:], rear[:, None]), dim=1)

    def restart(self, world_mask: torch.Tensor, state: PhysicalState) -> None:
        """Start the masked worlds' lines as the body's centreline, from the
        rear's tail to the tip, every tail on it with nothing erased."""
        tails = self.tails(state)
        corners = torch.cat(
            (tails[:, -1:], tails.flip(1)[:, 1:], state.front_end_position[:, :1]),
            dim=1,
        )
        start = corners[:, self.piece_of_point]
        along = corners[:, self.piece_of_point + 1] - start
        points = torch.cat(
            (start + self.share_of_point[:, None] * along, corners[:, -1:]), dim=1
        )
        arc = torch.cat(
            (
                torch.zeros_like(points[:, :1, 0]),
                torch.cumsum(points.diff(dim=1).norm(dim=-1), dim=1),
            ),
            dim=1,
        )
        mask = world_mask[:, None]
        count = self.start_count
        self.points[:, :count] = torch.where(
            mask[..., None], points, self.points[:, :count]
        )
        self.arc[:, :count] = torch.where(mask, arc, self.arc[:, :count])
        self.count.masked_fill_(world_mask, count)
        self.nearest_piece.copy_(
            torch.where(mask, self.tail_point.clamp(max=count - 2), self.nearest_piece)
        )
        self.erased.copy_(torch.where(mask, arc[:, self.tail_point], self.erased))
        self.erased_point.copy_(torch.where(mask[..., None], tails, self.erased_point))
        self.view.masked_fill_(mask[..., None], 0.0)
        self.head.restart(world_mask, state)
        self.drift.masked_fill_(world_mask, 0.0)

    def _drawn_tip(self, state: PhysicalState) -> torch.Tensor:
        """Where the tip draws, ``(W, 2)``: the tip, moved to the left of the
        head's travel by the drift. Refreshes ``sideslip`` and ``drift``."""
        speed, going = self.head.update(state)
        tip = state.front_end_position[:, 0]
        if not self.drift_leak_s:
            return tip
        leak = ACTION_DURATION_S / self.drift_leak_s
        self.drift += speed * self.sideslip.sin() - leak * self.drift
        left = torch.stack((-going[:, 1], going[:, 0]), dim=-1)
        return tip + self.drift[:, None] * left

    def update(self, state: PhysicalState) -> tuple[torch.Tensor, torch.Tensor]:
        """Extend the lines with the tips, measure every tail, and return the
        line each tail erased on the step and its distance from the line it
        has not erased, both ``(W, N)`` in metres; ``view`` is refreshed."""
        worlds = torch.arange(self.count.shape[0], device=self.count.device)
        tip = self._drawn_tip(state)
        newest = self.count - 1
        step = (tip - self.points[worlds, newest]).norm(dim=-1)
        moved = step >= SPACING_M
        self.points[worlds, self.count] = torch.where(
            moved[:, None], tip, self.points[worlds, self.count]
        )
        self.arc[worlds, self.count] = torch.where(
            moved, self.arc[worlds, newest] + step, self.arc[worlds, self.count]
        )
        self.count += moved

        # The pieces around where each tail was last: (W, N, 2 x WINDOW).
        tails = self.tails(state)
        last_piece = (self.count - 2)[:, None, None]
        piece = (self.nearest_piece[..., None] + self.window).clamp(min=0)
        piece = torch.minimum(piece, last_piece)
        flat = piece.flatten(1)
        start = self.points.gather(1, flat[..., None].expand(-1, -1, 2)).view(
            *piece.shape, 2
        )
        end = self.points.gather(1, (flat + 1)[..., None].expand(-1, -1, 2)).view(
            *piece.shape, 2
        )
        start_arc = self.arc.gather(1, flat).view(piece.shape)
        along = end - start
        length = along.norm(dim=-1).clamp(min=1e-9)
        relative = tails[:, :, None] - start
        projected = (relative * along).sum(dim=-1) / length**2

        # The nearest point of the whole line: where the tail is along it.
        share = projected.clamp(0.0, 1.0)
        distance = (relative - share[..., None] * along).norm(dim=-1)
        best = distance.argmin(dim=-1, keepdim=True)
        self.nearest_piece.copy_(piece.gather(2, best).squeeze(2))
        share_best = share.gather(2, best).squeeze(2)
        length_best = length.gather(2, best).squeeze(2)
        tail_arc = start_arc.gather(2, best).squeeze(2) + share_best * length_best
        along_best = along.gather(2, best[..., None].expand(-1, -1, -1, 2)).squeeze(2)
        start_best = start.gather(2, best[..., None].expand(-1, -1, -1, 2)).squeeze(2)
        closest = start_best + share_best[..., None] * along_best
        side = (
            along_best[..., 0] * (tails - closest)[..., 1]
            - along_best[..., 1] * (tails - closest)[..., 0]
        )
        offset = torch.copysign(distance.gather(2, best).squeeze(2), side)
        angle = heading(state.body_quaternion) - torch.atan2(
            along_best[..., 1], along_best[..., 0]
        )
        angle = torch.remainder(angle + math.pi, 2 * math.pi) - math.pi

        erased = (tail_arc - self.erased).clamp(min=0.0)
        advanced = tail_arc > self.erased
        self.erased.copy_(torch.where(advanced, tail_arc, self.erased))
        self.erased_point.copy_(
            torch.where(advanced[..., None], closest, self.erased_point)
        )

        # The nearest point of the line not erased: from the erase point on.
        lowest = ((self.erased[..., None] - start_arc) / length).clamp(0.0, 1.0)
        remaining = torch.maximum(projected.clamp(max=1.0), lowest)
        gone = start_arc + length < self.erased[..., None]
        to_line = (relative - remaining[..., None] * along).norm(dim=-1)
        to_line = to_line.masked_fill(gone, math.inf).amin(dim=-1)
        to_end = (tails - self.erased_point).norm(dim=-1)
        off_line = torch.minimum(to_line, to_end)

        self.view.copy_(torch.stack((offset, angle, off_line), dim=-1))
        return erased, off_line


class StraightLine:
    """A straight line in every world and every segment's place on it
    (cycle 18): through the centre of the body's segments at each restart,
    along the head's heading there.

    After each ``update``, ``view`` ``(W, N, 3)`` holds for each segment's
    centre its signed distance from the line (m, positive to the left of the
    line's direction), the angle of its body from the line's direction (rad,
    positive to the left), and 0. ``sideslip`` is the head's
    (``HeadSideslip``) and ``drift`` stays zero, as HeadLine's without one.
    """

    def __init__(
        self, world_count: int, segment_count: int, device: torch.device | str
    ) -> None:
        self.origin = torch.zeros((world_count, 2), device=device)
        self.direction = torch.zeros((world_count, 2), device=device)
        # Each centre's place along the line after the last step.
        self.along = torch.zeros((world_count, segment_count), device=device)
        self.view = torch.zeros((world_count, segment_count, 3), device=device)
        self.head = HeadSideslip(world_count, device)
        self.sideslip = self.head.sideslip
        self.drift = torch.zeros(world_count, device=device)

    def _measure(self, state: PhysicalState) -> torch.Tensor:
        """Each centre's place along the line, then its view, ``(W, N, 4)``."""
        relative = state.body_planar_position - self.origin[:, None]
        x, y = self.direction[:, None].unbind(dim=-1)
        along = relative[..., 0] * x + relative[..., 1] * y
        left = x * relative[..., 1] - y * relative[..., 0]
        angle = heading(state.body_quaternion) - torch.atan2(y, x)
        angle = torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
        return torch.stack((along, left, angle, torch.zeros_like(along)), dim=-1)

    def restart(self, world_mask: torch.Tensor, state: PhysicalState) -> None:
        """Lay the masked worlds' lines through the body as it stands."""
        mask = world_mask[:, None]
        self.origin.copy_(
            torch.where(mask, state.body_planar_position.mean(dim=1), self.origin)
        )
        self.direction.copy_(
            torch.where(mask, forward_axis(state)[:, 0], self.direction)
        )
        measured = self._measure(state)
        self.along.copy_(torch.where(mask, measured[..., 0], self.along))
        self.view.copy_(torch.where(mask[..., None], measured[..., 1:], self.view))
        self.head.restart(world_mask, state)

    def update(self, state: PhysicalState) -> tuple[torch.Tensor, torch.Tensor]:
        """Measure every centre; return how far it went along the line on the
        step (m, negative backward) and its distance from the line (m), both
        ``(W, N)``; ``view`` is refreshed."""
        self.head.update(state)
        measured = self._measure(state)
        progress = measured[..., 0] - self.along
        self.along.copy_(measured[..., 0])
        self.view.copy_(measured[..., 1:])
        return progress, measured[..., 1].abs()
