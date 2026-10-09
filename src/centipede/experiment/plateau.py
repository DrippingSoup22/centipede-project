"""The plateau stop: ending a run that has stopped improving.

A run's progress is its arrival share (arrivals.py), plus the curriculum's
level when it has one. It rises while the arrivals rise toward the share the
curriculum holds, then while the level rises, and at the top level while the
arrivals rise again; without a curriculum it is the arrival share alone.

The arrival share swings with the rhythm of the episode length: the worlds
that run out of time started together and end together, one wave per episode
length. So the stop watches the average of the last AVERAGED_EPISODE_LENGTHS
episode lengths of readings, in which the swings cancel. A run reaches a
plateau when that average has not risen by ``min_progress`` above its best for
``cycles`` cycles. The experiment then stops the run cleanly, with a
checkpoint, so that money is not spent on cycles that teach nothing, whether
the agents have learned all they can or their learning has failed.
"""

import math
from collections import deque

# How many episode lengths of readings the progress is averaged over.
AVERAGED_EPISODE_LENGTHS = 3


class Plateau:
    """The best averaged progress of a session, and whether it is too long ago."""

    def __init__(self, cycles: int, min_progress: float, episode_windows: int) -> None:
        """``episode_windows`` is the number of windows in an episode length."""
        self.cycles = cycles
        self.min_progress = min_progress
        self.average_cycles = AVERAGED_EPISODE_LENGTHS * episode_windows
        self.best = -math.inf
        self.best_cycle = 0
        self._readings: deque[float] = deque(maxlen=self.average_cycles)

    def update(self, cycle: int, progress: float) -> None:
        """Note a cycle's progress. Once there are enough readings, their
        average counts as new progress only if it beats the best by
        ``min_progress``."""
        self._readings.append(progress)
        if len(self._readings) < self.average_cycles:
            return
        average = sum(self._readings) / self.average_cycles
        if average >= self.best + self.min_progress:
            self.best, self.best_cycle = average, cycle

    def reached(self, cycle: int) -> bool:
        """Whether ``cycles`` cycles have passed without new progress, counted
        from the first average."""
        return self.best > -math.inf and cycle - self.best_cycle >= self.cycles
