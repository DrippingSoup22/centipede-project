"""Each segment's trail and how far its body points off it (docs/cooperators/
reward.md, cycle 4).

A segment's trail is the path its front end traced: the head's tip, or the
spine joint it shares with the segment ahead. Its rear end is one segment
length behind, so the segment lies on its trail when its body points at
where its front end was one segment length ago. The offset is the angle
between the two; it needs no speed and is zero standing still. A segment's
offset is also the error of the joint in front of it: the bend that joint
should have to put the segment on the trail, less the bend it has.
"""

import math

import torch

from centipede.environment.simulation import PhysicalState

# Kept points per shortest segment length: the reference point lies at most
# this fraction of that length beyond one length back.
POINTS_PER_LENGTH = 8


def heading(body_quaternion: torch.Tensor) -> torch.Tensor:
    """The yaw of each segment's forward axis from its (w, x, y, z) quaternion."""
    w, x, y, z = body_quaternion.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def forward_axis(state: PhysicalState) -> torch.Tensor:
    """Each segment's forward axis in the plane, ``(W, N, 2)``."""
    yaw = heading(state.body_quaternion)
    return torch.stack((yaw.cos(), yaw.sin()), dim=-1)


class TrailMemory:
    """The last stretch of every segment's trail in every world, and the
    offsets of the latest step, ``offset`` ``(W, N)`` in radians, positive
    when the body points to the left of its trail.

    With an ``averaging_rate`` below 1 the front ends and the bodies' forward
    axes are first averaged through time, each step moving the average that
    share of the way to the new value (cycle 5), so that the sway within a
    stride leaves the offsets.
    """

    def __init__(
        self,
        world_count: int,
        segment_lengths_m: tuple[float, ...],
        device: torch.device | str,
        averaging_rate: float = 1.0,
    ) -> None:
        """Keep enough points per segment to reach one length back.

        A front end adds a point each time it has moved a spacing, an eighth
        of the shortest length, from its last: a body swaying in place adds
        none, and a trail never wears away while the body stands.
        """
        self.averaging_rate = averaging_rate
        self.lengths = torch.tensor(segment_lengths_m, device=device)
        self.spacing_m = min(segment_lengths_m) / POINTS_PER_LENGTH
        self.point_count = math.ceil(max(segment_lengths_m) / self.spacing_m) + 2
        segment_count = len(segment_lengths_m)
        # Newest first.
        self.points = torch.zeros(
            (world_count, segment_count, self.point_count, 2), device=device
        )
        self.offset = torch.zeros((world_count, segment_count), device=device)
        self.behind = self.spacing_m * torch.arange(self.point_count, device=device)
        self.front = torch.zeros((world_count, segment_count, 2), device=device)
        self.axis = torch.zeros((world_count, segment_count, 2), device=device)

    def restart(self, world_mask: torch.Tensor, state: PhysicalState) -> None:
        """Start the masked worlds' trails straight back along each segment,
        as if it had walked in on a line: every offset zero."""
        axis = forward_axis(state)
        straight = (
            state.front_end_position[:, :, None]
            - axis[:, :, None] * self.behind[:, None]
        )
        mask = world_mask[:, None, None, None]
        self.points.copy_(torch.where(mask, straight, self.points))
        self.offset.masked_fill_(world_mask[:, None], 0.0)
        mask = world_mask[:, None, None]
        self.front.copy_(torch.where(mask, state.front_end_position, self.front))
        self.axis.copy_(torch.where(mask, axis, self.axis))

    def update(self, state: PhysicalState) -> torch.Tensor:
        """Add the step's front ends to the trails and return the offsets."""
        rate = self.averaging_rate
        self.front += rate * (state.front_end_position - self.front)
        self.axis += rate * (forward_axis(state) - self.axis)
        front = self.front
        moved = (front - self.points[:, :, 0]).norm(dim=-1) >= self.spacing_m
        shifted = torch.cat((front[:, :, None], self.points[:, :, :-1]), dim=2)
        self.points.copy_(torch.where(moved[..., None, None], shifted, self.points))

        # The newest point at least a segment length away; the oldest kept
        # if the front end has not yet gone that far.
        distance = (front[:, :, None] - self.points).norm(dim=-1)
        far = distance >= self.lengths[:, None]
        index = torch.where(
            far.any(dim=-1), far.int().argmax(dim=-1), self.point_count - 1
        )
        reference = self.points.gather(
            2, index[..., None, None].expand(-1, -1, 1, 2)
        ).squeeze(2)
        chord = front - reference
        difference = torch.atan2(self.axis[..., 1], self.axis[..., 0]) - torch.atan2(
            chord[..., 1], chord[..., 0]
        )
        self.offset.copy_(torch.remainder(difference + math.pi, 2 * math.pi) - math.pi)
        return self.offset
