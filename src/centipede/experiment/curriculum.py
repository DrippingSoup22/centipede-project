"""The curriculum: targets that get harder as the agents arrive more often.

The experiment keeps one difficulty level between 0 and 1. Level 0 draws the
targets from [environment.target]'s ranges, level 1 from [curriculum]'s final
ranges, and a level between from ranges in between, each end moved in
proportion to the level. After each update cycle, ``update`` moves the level
by ``level_rate`` times the gap between the arrival share and the share the
curriculum holds: it rises while the agents arrive more often than that and
falls while they arrive less often, so that it settles at the difficulty where
they arrive about that often. Automatic domain randomization (Akkaya et al.,
2019, "Solving Rubik's Cube with a Robot Hand") widens its ranges on the same
principle.

The arrival share is that of the episodes that ended over the last episode
length of windows, as the training console shows it: over that span every
world ends at least one episode. A shorter span would be misleading at the
start of a session, when all worlds start together and only arrivals can end
an episode before the time limit, so the level waits until the span is full.
"""

import math
from collections import deque

from centipede.environment.settings import TargetSettings
from centipede.experiment.configuration import CurriculumSettings


class Curriculum:
    """The target's difficulty level during training, and its target ranges."""

    def __init__(
        self,
        settings: CurriculumSettings,
        target: TargetSettings,
        window_steps: int,
        episode_steps: int,
        level: float,
    ) -> None:
        """Start at ``level``: 0 for a new run, the saved level when continuing."""
        self.settings = settings
        self.level = level
        self._start_ranges = (target.distance_range_m, target.bearing_range_deg)
        self._final_ranges = (
            settings.final_distance_range_m,
            settings.final_bearing_range_deg,
        )
        # How the episodes of the last episode length of windows ended.
        self._recent_endings: deque[list[float]] = deque(
            maxlen=math.ceil(episode_steps / window_steps)
        )

    def target_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The distance (m) and bearing (degrees) ranges of the current level."""
        distance, bearing = (
            (
                low + self.level * (final_low - low),
                high + self.level * (final_high - high),
            )
            for (low, high), (final_low, final_high) in zip(
                self._start_ranges, self._final_ranges, strict=True
            )
        )
        return distance, bearing

    def update(self, ending_counts: list[float]) -> None:
        """Move the level after a window, from how its episodes ended.

        ``ending_counts`` counts the episodes that ended in the window by
        ending, arrivals first, as the environment's ending histogram does.
        """
        self._recent_endings.append(ending_counts)
        if len(self._recent_endings) < self._recent_endings.maxlen:
            return
        totals = [sum(column) for column in zip(*self._recent_endings, strict=True)]
        gap = totals[0] / sum(totals) - self.settings.arrival_share  # 0: arrived
        self.level = min(1.0, max(0.0, self.level + self.settings.level_rate * gap))
