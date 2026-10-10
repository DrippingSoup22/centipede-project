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
cancel, and keeps its own best. A run reaches a plateau when neither the
task's nor the wave's average has risen by ``min_progress`` above its best
for ``cycles`` cycles, and never before ``minimum_cycles``. The experiment
then stops the run cleanly, with a checkpoint, so that money is not spent on
cycles that teach nothing, whether the agents have learned all they can or
their learning has failed.

A run also stops once its task is solved: when the task's average reaches
``solved_task`` (the curriculum's final level plus the solved arrival
share) and the speed, the targets reached per world and minute, has not
risen for ``cycles`` cycles, nor the wave in a run with clocks, again
never before ``minimum_cycles``. A centipede that nearly always arrives
can still learn to arrive sooner, and its segments may still find a
common wave, so the run goes on while either improves. The speed is read
as its logarithm, so that ``min_progress`` counts as a relative rise: 0.02
is 2%.
"""

import math
from collections import deque

# How many episode lengths of readings the progress is averaged over.
AVERAGED_EPISODE_LENGTHS = 3
# The signal that only the solved stop watches.
SPEED = "speed"


class Signal:
    """One signal's latest readings, their latest average, and its best
    average so far."""

    def __init__(self, average_cycles: int) -> None:
        self.best = -math.inf
        self.best_cycle = 0
        self.average: float | None = None
        self.readings: deque[float] = deque(maxlen=average_cycles)

    def stalled(self, cycle: int, cycles: int) -> bool:
        """Whether ``cycles`` cycles have passed since its best, counted from
        its first average."""
        return self.best > -math.inf and cycle - self.best_cycle >= cycles


class Plateau:
    """The best averaged progress of each signal in a session, and whether every
    one is too long ago."""

    def __init__(
        self,
        cycles: int,
        min_progress: float,
        episode_windows: int,
        minimum_cycles: int = 0,
        solved_task: float | None = None,
    ) -> None:
        """``episode_windows`` is the number of windows in an episode length;
        without ``solved_task`` a run never counts as solved."""
        self.cycles = cycles
        self.min_progress = min_progress
        self.minimum_cycles = minimum_cycles
        self.solved_task = solved_task
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
            signal.average = sum(signal.readings) / self.average_cycles
            if signal.average >= signal.best + self.min_progress:
                signal.best, signal.best_cycle = signal.average, cycle

    def reached(self, cycle: int) -> bool:
        """Whether ``cycles`` cycles have passed without new progress in the
        task or the wave, each counted from its first average, and the run has
        trained its ``minimum_cycles``."""
        progress = [signal for name, signal in self.signals.items() if name != SPEED]
        return (
            cycle >= self.minimum_cycles
            and bool(progress)
            and all(signal.stalled(cycle, self.cycles) for signal in progress)
        )

    def solved(self, cycle: int) -> bool:
        """Whether the task's average has reached ``solved_task``, neither the
        speed nor the wave (in a run with clocks) has risen for ``cycles``
        cycles, and the run has trained its ``minimum_cycles``."""
        task, speed = self.signals.get("task"), self.signals.get(SPEED)
        wave = self.signals.get("wave")
        return (
            self.solved_task is not None
            and cycle >= self.minimum_cycles
            and task is not None
            and speed is not None
            and task.average is not None
            and task.average >= self.solved_task
            and speed.stalled(cycle, self.cycles)
            and (wave is None or wave.stalled(cycle, self.cycles))
        )
