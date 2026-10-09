"""The plateau stop: ending a run that has stopped improving.

A run's progress is watched through one or two signals. The task is its
arrival share (arrivals.py), plus the curriculum's level when it has one: it
rises while the arrivals rise toward the share the curriculum holds, then
while the level rises, and at the top level while the arrivals rise again;
without a curriculum it is the arrival share alone. In a run whose segments
have clocks, the wave is the second: how alike the phase offsets between
neighbours are across every world and step of a window, averaged over the
pairs. Every clock starts at the same tempo, so within each world the offsets
hold from the start; across the worlds, which start from random phases, they
grow alike only as the segments steer them toward the same wave.

The arrival share swings with the rhythm of the episode length: the worlds
that run out of time started together and end together, one wave per episode
length. So each signal is watched through the average of its last
AVERAGED_EPISODE_LENGTHS episode lengths of readings, in which the swings
cancel, and keeps its own best. A run reaches a plateau when no signal's
average has risen by ``min_progress`` above its best for ``cycles`` cycles,
and never before ``minimum_cycles``. The experiment then stops the run
cleanly, with a checkpoint, so that money is not spent on cycles that teach
nothing, whether the agents have learned all they can or their learning has
failed.
"""

import math
from collections import deque

# How many episode lengths of readings the progress is averaged over.
AVERAGED_EPISODE_LENGTHS = 3


class Signal:
    """One signal's latest readings, and its best average so far."""

    def __init__(self, average_cycles: int) -> None:
        self.best = -math.inf
        self.best_cycle = 0
        self.readings: deque[float] = deque(maxlen=average_cycles)


class Plateau:
    """The best averaged progress of each signal in a session, and whether every
    one is too long ago."""

    def __init__(
        self,
        cycles: int,
        min_progress: float,
        episode_windows: int,
        minimum_cycles: int = 0,
    ) -> None:
        """``episode_windows`` is the number of windows in an episode length."""
        self.cycles = cycles
        self.min_progress = min_progress
        self.minimum_cycles = minimum_cycles
        self.average_cycles = AVERAGED_EPISODE_LENGTHS * episode_windows
        self.signals: dict[str, Signal] = {}

    def update(self, cycle: int, readings: dict[str, float]) -> None:
        """Note a cycle's reading of each signal measured so far. Once a signal
        has enough readings, their average counts as new progress only if it
        beats that signal's best by ``min_progress``."""
        for name, reading in readings.items():
            signal = self.signals.setdefault(name, Signal(self.average_cycles))
            signal.readings.append(reading)
            if len(signal.readings) < self.average_cycles:
                continue
            average = sum(signal.readings) / self.average_cycles
            if average >= signal.best + self.min_progress:
                signal.best, signal.best_cycle = average, cycle

    def reached(self, cycle: int) -> bool:
        """Whether ``cycles`` cycles have passed without new progress in any
        signal, each counted from its first average, and the run has trained
        its ``minimum_cycles``."""
        return (
            cycle >= self.minimum_cycles
            and bool(self.signals)
            and all(
                signal.best > -math.inf and cycle - signal.best_cycle >= self.cycles
                for signal in self.signals.values()
            )
        )
