"""Tests for the interaction loop's diagnostics.

How a category is summarised is tested with ``WindowSummary``; these tests
cover what the loop's diagnostics add: each window starts empty, every step
reaches both environment summaries, and the timing.
"""

from types import SimpleNamespace

import torch

from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.simulation.diagnostics import SimulationDiagnostics
from centipede.interaction_loop.diagnostics import LoopDiagnostics


def test_windows_start_empty_and_time_collecting_and_learning():
    simulation = SimulationDiagnostics.allocate(2, 3, "cpu")
    environment_diagnostics = EnvironmentDiagnostics(
        2, 3, ["arrival"], "cpu", simulation.facts
    )
    environment = SimpleNamespace(diagnostics=environment_diagnostics, world_count=2)
    diagnostics = LoopDiagnostics(environment)
    episode = environment_diagnostics.episode

    for window_steps in (3, 2):
        with diagnostics.collecting():
            for _ in range(window_steps):
                environment_diagnostics.step.body_height.fill_(float(window_steps))
                episode.episode_ended.fill_(True)
                diagnostics.step_taken()
        with diagnostics.learning():
            pass

    # Only the second window's two steps of two worlds are summarised.
    assert diagnostics.step_window.result()["body_height"].tolist() == [2.0] * 3
    assert list(diagnostics.simulation_window.result()) == [
        "contact_count",
        "constraint_rows",
        "solver_iterations",
    ]
    assert diagnostics.episode_window.result()["episode_ended"] == 4
    timing = diagnostics.timing
    assert timing.collecting_seconds > 0 and timing.learning_seconds > 0
    torch.testing.assert_close(
        timing.transitions_per_second, 4 / timing.collecting_seconds
    )

    # Evaluation counts only the worlds given, and leaves learning at zero.
    with diagnostics.collecting():
        diagnostics.step_taken(torch.tensor([True, False]))
    assert diagnostics.episode_window.result()["episode_ended"] == 1
    assert timing.learning_seconds == 0
