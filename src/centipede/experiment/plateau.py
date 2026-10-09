"""The plateau stop: ending a run that has stopped improving.

A run's progress is its arrival share (arrivals.py), plus the curriculum's
level when it has one. It rises while the arrivals rise toward the share the
curriculum holds, then while the level rises, and at the top level while the
arrivals rise again; without a curriculum it is the arrival share alone. A run
reaches a plateau when its progress has not risen by ``min_progress`` above
its best for ``cycles`` cycles. The experiment then stops the run cleanly,
with a checkpoint, so that money is not spent on cycles that teach nothing,
whether the agents have learned all they can or their learning has failed.
"""

import math


class Plateau:
    """The best progress of a session so far, and whether it is too long ago."""

    def __init__(self, cycles: int, min_progress: float, first_cycle: int) -> None:
        """Watch a session that starts at ``first_cycle``."""
        self.cycles = cycles
        self.min_progress = min_progress
        self.best = -math.inf
        self.best_cycle = first_cycle - 1

    def update(self, cycle: int, progress: float) -> None:
        """Note a cycle's progress; it counts as new only if it beats the best
        by ``min_progress``."""
        if progress >= self.best + self.min_progress:
            self.best, self.best_cycle = progress, cycle

    def reached(self, cycle: int) -> bool:
        """Whether ``cycles`` cycles have passed without new progress."""
        return cycle - self.best_cycle >= self.cycles
