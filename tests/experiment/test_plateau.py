"""Tests for the plateau stop's rule.

How the experiment stops a run at a plateau is tested with the experiment.
"""

import math

from centipede.experiment.plateau import Plateau


def test_a_plateau_is_too_many_cycles_without_a_large_enough_rise_of_the_average():
    # One window per episode length: the average covers the last three cycles.
    plateau = Plateau(cycles=3, min_progress=0.02, episode_windows=1)
    plateau.update(1, {"task": 0.40})
    plateau.update(2, {"task": 0.60})
    assert not plateau.reached(10)  # the clock starts with the first average
    plateau.update(3, {"task": 0.50})  # the first average, 0.50, is the best so far
    plateau.update(4, {"task": 0.43})  # the average is 0.51: too small a rise to count
    plateau.update(5, {"task": 0.57})  # 0.50
    assert not plateau.reached(5)
    assert plateau.reached(6)
    plateau.update(6, {"task": 0.62})  # 0.54: new progress
    task = plateau.signals["task"]
    assert (round(task.best, 3), task.best_cycle) == (0.54, 6)
    assert not plateau.reached(8)


def test_a_plateau_needs_every_signal_stalled_and_the_minimum_cycles():
    """The task is flat; the wave rises until cycle 6, the agents' return until 8."""
    plateau = Plateau(cycles=2, min_progress=0.02, episode_windows=1, minimum_cycles=9)
    for cycle in range(1, 4):
        plateau.update(cycle, {"task": 0.9, "wave": 0.1 * cycle, "return": -0.5})
    for cycle in range(4, 9):
        rising = -0.5 + 0.1 * min(cycle - 3, 3)
        plateau.update(cycle, {"task": 0.9, "wave": 0.4, "return": rising})
    assert plateau.signals["task"].best_cycle == 3
    assert plateau.signals["wave"].best_cycle == 6
    assert plateau.signals["return"].best_cycle == 8
    assert not plateau.reached(8)  # the task and the wave stalled, not the return
    assert not plateau.reached(9)
    assert plateau.reached(10)  # all three stalled, and the minimum is past
    late = Plateau(cycles=2, min_progress=0.02, episode_windows=1, minimum_cycles=9)
    for cycle in range(1, 4):
        late.update(cycle, {"task": 0.9, "wave": 0.4, "return": -0.5})
    assert not late.reached(8)  # all three stalled, before the minimum
    assert late.reached(9)


def test_a_solved_task_stops_a_run_once_it_gets_no_faster():
    # Solved at a task average of 1.95: the final level, 1, plus 95% arrivals.
    def plateau():
        return Plateau(cycles=1, min_progress=0.02, episode_windows=1, solved_task=1.95)

    steady, faster, waving = plateau(), plateau(), plateau()
    for cycle, task in enumerate([1.9, 1.9, 1.9, 2.0], start=1):
        steady.update(cycle, {"task": task, "speed": math.log(7)})
        faster.update(cycle, {"task": task, "speed": math.log(4 + cycle)})
        waving.update(cycle, {"task": task, "speed": math.log(7), "wave": 0.1 * cycle})
    assert not steady.solved(4)  # no faster since cycle 3, but the task is 1.93
    steady.update(5, {"task": 2.0, "speed": math.log(7)})
    faster.update(5, {"task": 2.0, "speed": math.log(9)})
    waving.update(5, {"task": 2.0, "speed": math.log(7), "wave": 0.5})
    assert steady.solved(5)  # the task is 1.97
    assert not faster.solved(5)  # it still gets faster
    assert not waving.solved(5)  # its wave still forms
    assert not steady.reached(5)  # a plateau needs the task itself stalled
