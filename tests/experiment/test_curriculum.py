"""Tests for the curriculum: how its level follows the arrivals, and its ranges.

How the experiment applies it during a run is tested with the experiment.
"""

import pytest

from centipede.environment.settings import TargetSettings
from centipede.experiment.configuration import CurriculumSettings
from centipede.experiment.curriculum import Curriculum

# Episode endings, in the environment's order: arrived, within a quarter,
# within half, closer, not closer, left the circle.
ALL_ARRIVED = [4, 0, 0, 0, 0, 0]
THREE_OF_FOUR_ARRIVED = [3, 0, 0, 0, 1, 0]
NONE_ARRIVED = [0, 0, 0, 0, 4, 0]


def make_curriculum() -> Curriculum:
    """From the default targets (30-60 mm, ±30°) to 30-200 mm all around,
    with two windows per episode length."""
    settings = CurriculumSettings.from_section(
        {"final_distance_range_m": [0.03, 0.2], "final_bearing_range_deg": [-180, 180]}
    )
    target = TargetSettings.from_section({})
    return Curriculum(settings, target, window_steps=4, episode_steps=8, level=0.0)


def test_the_level_follows_the_arrival_share_of_the_last_episode_length():
    curriculum = make_curriculum()
    curriculum.update(ALL_ARRIVED)  # one window is not yet an episode length
    assert curriculum.level == 0.0
    curriculum.update(THREE_OF_FOUR_ARRIVED)  # 7 arrivals of 8 over both windows
    assert curriculum.level == pytest.approx(0.02 * (7 / 8 - 0.5))

    for _ in range(200):
        curriculum.update(ALL_ARRIVED)
    assert curriculum.level == 1.0
    distance, bearing = curriculum.target_ranges()
    assert distance == pytest.approx((0.03, 0.2))
    assert bearing == pytest.approx((-180, 180))

    for _ in range(200):
        curriculum.update(NONE_ARRIVED)
    assert curriculum.level == 0.0
    assert curriculum.target_ranges() == ((0.03, 0.06), (-30, 30))


def test_a_level_between_moves_each_end_of_each_range_in_proportion():
    curriculum = make_curriculum()
    curriculum.level = 0.5
    distance, bearing = curriculum.target_ranges()
    assert distance == pytest.approx((0.03, 0.13))
    assert bearing == pytest.approx((-105, 105))
