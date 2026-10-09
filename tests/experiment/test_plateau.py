"""Tests for the plateau stop's rule.

How the experiment stops a run at a plateau is tested with the experiment.
"""

from centipede.experiment.plateau import Plateau


def test_a_plateau_is_too_many_cycles_without_a_large_enough_rise():
    plateau = Plateau(cycles=3, min_progress=0.02, first_cycle=1)
    plateau.update(1, 0.50)  # the first progress is the best so far
    plateau.update(2, 0.51)  # too small a rise to count
    plateau.update(3, 0.49)
    assert not plateau.reached(3)
    assert plateau.reached(4)
    plateau.update(4, 0.53)  # a rise of 0.03: new progress
    assert (plateau.best, plateau.best_cycle) == (0.53, 4)
    assert not plateau.reached(6)
