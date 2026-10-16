"""The cooperators' task: targets placed by a level, a stopwatch on each target,
and the level that follows how many are reached in time.

The cooperators' environment owns one; see docs/cooperators/task.md.
"""

import math

import torch

from centipede.environment.head import head_forward_direction
from centipede.environment.settings import CooperatorsSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.constants import ACTION_DURATION_S


class CooperativeTask:
    """Every world's target, the steps since it appeared, and the level.

    The near end of the targets' distances grows in a straight line from the
    settings' start at level 0 to its final body lengths at level 1, and the
    far end is always ``far_to_near_ratio`` times it. Their bearing, either
    side of the head's forward direction, opens in a straight line from
    ``start_bearing_deg`` at level 0 to the full circle at
    ``full_circle_level``: where the near end clears one body length plus a
    turning circle, so that a target behind lies clear of the body.
    """

    def __init__(
        self,
        settings: CooperatorsSettings,
        body: BodyMeasures,
        world_count: int,
        episode_steps: int,
        device: torch.device | str,
    ) -> None:
        """Start at the settings' level with no targets yet, and an empty
        record of stopwatches for one episode length of steps."""
        self.settings = settings
        self.level = settings.start_level
        self.start_near_m = settings.start_near_distance_m
        self.final_near_m = settings.final_near_body_lengths * body.body_length_m
        clear_of_body_m = body.body_length_m + body.turning_circle_m
        self.full_circle_level = min(
            1.0,
            (clear_of_body_m - self.start_near_m)
            / (self.final_near_m - self.start_near_m),
        )
        self.stopwatch_steps = round(settings.stopwatch_s / ACTION_DURATION_S)

        self.target_position = torch.zeros(
            (world_count, 2), dtype=torch.float32, device=device
        )
        self.target_steps = torch.zeros(world_count, dtype=torch.int32, device=device)
        # Whether each target's stopwatch can run out before its world restarts.
        self.timed = torch.zeros(world_count, dtype=torch.bool, device=device)
        # The worlds whose stopwatches move the level: all but those the
        # experiment reserves to measure the mean action.
        self.counted_worlds = torch.ones(world_count, dtype=torch.bool, device=device)
        self.generator = torch.Generator(device=device)
        # For each of the last episode length of steps: the targets whose
        # stopwatch ended on it reached in time, and all of them.
        self._stopwatches = torch.zeros((episode_steps, 2), device=device)
        self._recorded_steps = 0
        self.in_time_share: float | None = None

    def distance_range_m(self) -> tuple[float, float]:
        """The near and far ends of the current level's target distances."""
        near = self.start_near_m + self.level * (self.final_near_m - self.start_near_m)
        return near, self.settings.far_to_near_ratio * near

    def half_bearing_rad(self) -> float:
        """How far either side of straight ahead a target may lie."""
        opening = min(1.0, self.level / self.full_circle_level)
        start = self.settings.start_bearing_deg
        return math.radians(start + opening * (180.0 - start))

    def place_targets(
        self,
        world_mask: torch.Tensor,
        physical_state: PhysicalState,
        steps_left: torch.Tensor,
    ) -> None:
        """Give the masked worlds a new target from the head's tip, at a
        distance and a bearing drawn uniformly from the level's ranges, and
        start its stopwatch. Values are drawn for every world and kept only
        where the mask is set, which avoids asking the GPU how many there are.

        ``steps_left`` is each world's steps until it restarts: a target
        whose stopwatch cannot run out before then is never counted, reached
        or not, since only the reached ones would be (a cut stopwatch)."""
        world_count = world_mask.shape[0]
        near, far = self.distance_range_m()
        half_bearing = self.half_bearing_rad()
        draws = torch.rand(
            (2, world_count), generator=self.generator, device=world_mask.device
        )
        distance = near + (far - near) * draws[0]
        bearing = half_bearing * (2 * draws[1] - 1)
        forward = head_forward_direction(physical_state.body_quaternion[:, 0])
        left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        direction = (
            torch.cos(bearing)[:, None] * forward + torch.sin(bearing)[:, None] * left
        )
        target = physical_state.head_tip_position[:, :2] + distance[:, None] * direction
        self.target_position.copy_(
            torch.where(world_mask[:, None], target, self.target_position)
        )
        self.target_steps.masked_fill_(world_mask, 0)
        self.timed.copy_(
            torch.where(world_mask, steps_left >= self.stopwatch_steps, self.timed)
        )

    def time_step(self, arrived: torch.Tensor) -> torch.Tensor:
        """Advance every target's stopwatch by a step and record the timed
        targets whose stopwatch ends: reached within it, or not reached when
        it runs out. A target reached later is not counted again. Returns the
        worlds whose target was reached in time, timed or not."""
        self.target_steps += 1
        in_time = arrived & (self.target_steps <= self.stopwatch_steps)
        missed = ~arrived & (self.target_steps == self.stopwatch_steps)
        row = self._recorded_steps % self._stopwatches.shape[0]
        counted = self.timed & self.counted_worlds
        self._stopwatches[row, 0] = (in_time & counted).sum()
        self._stopwatches[row, 1] = ((in_time | missed) & counted).sum()
        self._recorded_steps += 1
        return in_time

    def adjust_level(self) -> None:
        """Move the level toward the difficulty where the set-point share of
        targets is reached in time, once a whole episode length is recorded.

        ``in_time_share`` is that share over the last episode length, or None
        before one is recorded or while no stopwatch ended in it. This is the
        task's only wait for the GPU, once per update cycle.
        """
        if self._recorded_steps < self._stopwatches.shape[0]:
            return
        in_time, timed = self._stopwatches.sum(dim=0).tolist()
        if timed == 0:
            self.in_time_share = None
            return
        self.in_time_share = in_time / timed
        gap = self.in_time_share - self.settings.level_set_point
        self.level = min(1.0, max(0.0, self.level + self.settings.level_rate * gap))
