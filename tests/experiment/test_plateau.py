"""Tests for the plateau stop's rule.

How the experiment stops a run at a plateau is tested with the experiment.
"""

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
    plateau = Plateau(cycles=2, min_progress=0.02, episode_windows=1, minimum_cycles=9)
    for cycle in range(1, 4):
        plateau.update(cycle, {"task": 0.9, "wave": 0.1 * cycle})
    for cycle in range(4, 7):
        plateau.update(cycle, {"task": 0.9, "wave": 0.4})  # the wave still rises
    assert plateau.signals["task"].best_cycle == 3
    assert plateau.signals["wave"].best_cycle == 6
    assert not plateau.reached(7)  # the task stalled, the wave not yet
    assert not plateau.reached(8)  # both stalled, before the minimum
    assert plateau.reached(9)
