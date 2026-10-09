"""Tests for the plateau stop's rule.

How the experiment stops a run at a plateau is tested with the experiment.
"""

from centipede.experiment.plateau import Plateau


def test_a_plateau_is_too_many_cycles_without_a_large_enough_rise_of_the_average():
    # One window per episode length: the average covers the last three cycles.
    plateau = Plateau(cycles=3, min_progress=0.02, episode_windows=1)
    plateau.update(1, 0.40)
    plateau.update(2, 0.60)
    assert not plateau.reached(10)  # the clock starts with the first average
    plateau.update(3, 0.50)  # the first average, 0.50, is the best so far
    plateau.update(4, 0.43)  # the average is 0.51: too small a rise to count
    plateau.update(5, 0.57)  # 0.50
    assert not plateau.reached(5)
    assert plateau.reached(6)
    plateau.update(6, 0.62)  # 0.54: new progress
    assert (round(plateau.best, 3), plateau.best_cycle) == (0.54, 6)
    assert not plateau.reached(8)
