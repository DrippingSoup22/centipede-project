"""The line the head's tip draws on the floor, and each segment's place on it
(docs/cooperators/reward.md, cycle 7).

Every segment has its own copy of the line, which its tail, its rear end,
erases as it passes: a copy runs from the furthest point its tail has
reached to the tip. A follower is paid for the line its tail erases and
charged for how far its tail lies from the line it has not erased yet,
beside it or behind its end. The line is kept as points, one each time the
tip has moved SPACING_M, with their arc lengths along the line; a world's
line starts at each restart as the body's own centreline.
"""

import math

import torch

from centipede.environment.simulation import PhysicalState

# The tip adds a point once it has moved this far from the last one, so a
# head standing still draws nothing.
SPACING_M = 0.0005
# A tail looks for itself this many pieces either way of where it was last.
WINDOW_PIECES = 64


def heading(body_quaternion: torch.Tensor) -> torch.Tensor:
    """The yaw of each segment's forward axis from its (w, x, y, z) quaternion."""
    w, x, y, z = body_quaternion.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def forward_axis(state: PhysicalState) -> torch.Tensor:
    """Each segment's forward axis in the plane, ``(W, N, 2)``."""
    yaw = heading(state.body_quaternion)
    return torch.stack((yaw.cos(), yaw.sin()), dim=-1)


class HeadLine:
    """Every world's line and every segment's place on it.

    After each ``update``, ``view`` ``(W, N, 3)`` holds for each segment's
    tail: its signed distance from the line (m, positive to the left of the
    line's direction), the angle of its body from the line's direction
    there (rad, positive to the left), and its distance from the line it has
    not erased (m). The head's own copy, erased by its neck, is measured too.
    """

    def __init__(
        self,
        world_count: int,
        segment_lengths_m: tuple[float, ...],
        episode_steps: int,
        device: torch.device | str,
    ) -> None:
        """Room for the starting centreline and one point per step of an
        episode, the most the tip can add before the world restarts."""
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

    def update(self, state: PhysicalState) -> tuple[torch.Tensor, torch.Tensor]:
        """Extend the lines with the tips, measure every tail, and return the
        line each tail erased on the step and its distance from the line it
        has not erased, both ``(W, N)`` in metres; ``view`` is refreshed."""
        worlds = torch.arange(self.count.shape[0], device=self.count.device)
        tip = state.front_end_position[:, 0]
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
