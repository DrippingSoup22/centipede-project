"""The training's arrivals, which the curriculum and the plateau stop follow.

The arrival share is the share of arrivals among the episodes that ended over
the last episode length of windows, as the training console shows it: over
that span every world ends at least one episode. A shorter span would mislead
at the start of a session, when all worlds start together and only arrivals
can end an episode before the time limit, so there is no share until the span
is full. The targets reached per world and minute over the same span count
how many targets the agents collect, which a body walking on after each
arrival makes more than one per episode length.
"""

import math
from collections import deque

from centipede.experiment.report import STEP_SECONDS


def targets_per_minute(arrivals: float, world_count: int, steps: int) -> float:
    """Targets reached per world and minute, from the arrivals over ``steps``."""
    return arrivals / world_count / (steps * STEP_SECONDS / 60)


class ArrivalShare:
    """The arrival share of the last episode length of windows, or None, and the
    targets reached per world and minute over the same windows."""

    def __init__(self, window_steps: int, episode_steps: int, world_count: int) -> None:
        self.value: float | None = None
        self.targets_per_minute: float | None = None
        self._window_steps = window_steps
        self._world_count = world_count
        # How the episodes of the last episode length of windows ended.
        self._recent_endings: deque[list[float]] = deque(
            maxlen=math.ceil(episode_steps / window_steps)
        )

    def add(self, ending_counts: list[float]) -> None:
        """Count a window's episodes by ending, arrivals first, as the
        environment's ending histogram does."""
        self._recent_endings.append(ending_counts)
        windows = len(self._recent_endings)
        if windows == self._recent_endings.maxlen:
            totals = [sum(column) for column in zip(*self._recent_endings, strict=True)]
            self.value = totals[0] / sum(totals)
            self.targets_per_minute = targets_per_minute(
                totals[0], self._world_count, windows * self._window_steps
            )
