"""The interaction loop's diagnostics: timing, and the environment's window summaries.

The front file uses three things. ``with collecting():`` around a window's
steps, both in training and in evaluation, and ``with learning():`` around the
update, time those parts; ``step_taken`` after every step adds the
environment's step facts and episode summaries to the window's summaries. Each
window starts empty. The clock waits for the GPU only when a timed part starts
and ends, once per window, so the times include the GPU's work. The values are
listed in docs/diagnostics.md.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from centipede.diagnostics_category import WindowSummary, measure


@dataclass(frozen=True)
class Timing:
    """Where the last window's time went; single values, on the CPU."""

    collecting_seconds: torch.Tensor = measure(
        "Time spent collecting the window: acting, stepping, recording", "s"
    )
    learning_seconds: torch.Tensor = measure(
        "Time spent on the update after the window; 0 in evaluation", "s"
    )
    transitions_per_second: torch.Tensor = measure(
        "World steps collected per second of collecting", "1/s"
    )


class LoopDiagnostics:
    """Times each window and summarises the environment's categories over it.

    ``timing``, ``step_window``, and ``episode_window`` are what the experiment
    reads; each window summary offers its category's ``descriptions`` and its
    ``result()``. Episode summaries count only the worlds whose episode ended.
    """

    def __init__(self, environment) -> None:
        """Prepare the summaries of the environment's two categories."""
        self.step_window = WindowSummary(environment.diagnostics.step)
        self.episode_window = WindowSummary(
            environment.diagnostics.episode, mask_field="episode_ended"
        )
        self.timing = Timing(
            collecting_seconds=torch.zeros(()),
            learning_seconds=torch.zeros(()),
            transitions_per_second=torch.zeros(()),
        )
        self._world_count = environment.world_count
        self._window_steps = 0

    @contextmanager
    def collecting(self) -> Iterator[None]:
        """Start a new window, and time the steps taken inside the block."""
        self.step_window.clear()
        self.episode_window.clear()
        self._window_steps = 0
        self.timing.learning_seconds.zero_()
        start = _synchronised_clock()
        yield
        seconds = _synchronised_clock() - start
        self.timing.collecting_seconds.fill_(seconds)
        self.timing.transitions_per_second.fill_(
            self._world_count * self._window_steps / seconds
        )

    @contextmanager
    def learning(self) -> Iterator[None]:
        """Time the update inside the block."""
        start = _synchronised_clock()
        yield
        self.timing.learning_seconds.fill_(_synchronised_clock() - start)

    def step_taken(self, counted_worlds: torch.Tensor | None = None) -> None:
        """Add the step just taken to the window's summaries.

        ``counted_worlds`` is a ``(W,)`` mask of the worlds whose values count,
        for evaluation; in training every world counts.
        """
        self.step_window.add(counted_worlds)
        self.episode_window.add(counted_worlds)
        self._window_steps += 1


def _synchronised_clock() -> float:
    """The time in seconds, once all GPU work queued so far has finished."""
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return time.perf_counter()
