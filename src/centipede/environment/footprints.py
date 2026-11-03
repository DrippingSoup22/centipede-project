"""Prints and marks (docs/follow-footsteps/task.md, prints, marks and the
speed): where each foot landed, where each walker should step, and when a
walker has lost its marks. The environment feeds it each step's feet and
the leader's footholds; the reward and the observations read the marks."""

import torch


class Footprints:
    """Every foot's last two prints, unit 1's footholds, and every walker's
    marks, in world metres, sides left then right.

    A print is where a foot touched down: its centre on the step it touches
    the ground after at least ``swing_steps`` steps in the air, so that a
    foot that only flickers off the ground makes none. ``prints`` ``(W, N,
    2, 2, 3)`` holds each foot's older print, then its newer. ``footholds``
    ``(W, 2, 2, 3)`` holds unit 1's on each side: the one its foot went for,
    then the next, which the leader gives at that foot's touchdown and which
    stays put until the next. A walker's marks on a side are its leader's
    prints there, unit 1's its footholds.

    A walker has lost its marks when its feet stay farther from them, on
    average, than ``lost_distance_m`` for ``lost_steps`` steps in a row.
    """

    def __init__(
        self,
        world_count: int,
        segment_count: int,
        swing_steps: int,
        lost_distance_m: float,
        lost_steps: int,
        device: torch.device | str,
    ) -> None:
        self.swing_steps = swing_steps
        self.lost_distance_m = lost_distance_m
        self.lost_steps = lost_steps
        self.prints = torch.zeros(world_count, segment_count, 2, 2, 3, device=device)
        self.footholds = torch.zeros(world_count, 2, 2, 3, device=device)
        self.steps_in_air = torch.zeros(
            world_count, segment_count, 2, dtype=torch.long, device=device
        )
        self.far_steps = torch.zeros(
            world_count, segment_count, dtype=torch.long, device=device
        )

    def restart(
        self,
        world_mask: torch.Tensor,
        foot_position: torch.Tensor,
        first_footholds: torch.Tensor,
    ) -> None:
        """The ``(W,)`` masked worlds start with both prints of every foot
        where it stands, ``foot_position`` ``(W, N, 2, 3)``, and unit 1's
        footholds where its feet stand, then ``first_footholds`` ``(W, 2,
        3)``."""
        standing = foot_position[world_mask]
        self.prints[world_mask] = standing[..., None, :].expand(-1, -1, -1, 2, -1)
        self.footholds[world_mask, :, 0] = standing[:, 0]
        self.footholds[world_mask, :, 1] = first_footholds[world_mask]
        self.steps_in_air[world_mask] = 0
        self.far_steps[world_mask] = 0

    def update(
        self, foot_position: torch.Tensor, foot_down: torch.Tensor
    ) -> torch.Tensor:
        """Record the step's touchdowns from each foot's ``foot_position``
        ``(W, N, 2, 3)`` and whether it touches the ground, ``foot_down``
        ``(W, N, 2)``; returns the touchdowns, ``(W, N, 2)``."""
        touchdown = foot_down & (self.steps_in_air >= self.swing_steps)
        self.steps_in_air = torch.where(foot_down, 0, self.steps_in_air + 1)
        self.prints[touchdown] = torch.stack(
            (self.prints[touchdown][:, 1], foot_position[touchdown]), dim=1
        )
        return touchdown

    def give_footholds(self, given: torch.Tensor, next_footholds: torch.Tensor) -> None:
        """On each side whose foot of unit 1 touched down, ``given`` ``(W,
        2)``, the next foothold becomes the one the foot went for and
        ``next_footholds`` ``(W, 2, 3)`` the next."""
        self.footholds[given] = torch.stack(
            (self.footholds[given][:, 1], next_footholds[given]), dim=1
        )

    def marks(self) -> torch.Tensor:
        """Each walker's marks, ``(W, N, 2, 2, 3)``: unit 1's footholds, then
        each other walker's leader's prints."""
        return torch.cat((self.footholds[:, None], self.prints[:, :-1]), dim=1)

    def mark_distances(self, foot_position: torch.Tensor) -> torch.Tensor:
        """Each foot's distance across the ground to the nearer of its marks
        on its side, ``(W, N, 2)`` m, from ``foot_position`` ``(W, N, 2, 3)``."""
        across = self.marks()[..., :2] - foot_position[..., None, :2]
        return across.norm(dim=-1).min(dim=-1).values

    def lost(self, mark_distances: torch.Tensor) -> torch.Tensor:
        """Count each walker's steps far from its marks, from this step's
        ``mark_distances`` ``(W, N, 2)``; returns the worlds where a walker
        has lost them, ``(W,)``."""
        far = mark_distances.mean(dim=-1) > self.lost_distance_m
        self.far_steps = torch.where(far, self.far_steps + 1, 0)
        return (self.far_steps >= self.lost_steps).any(dim=1)
