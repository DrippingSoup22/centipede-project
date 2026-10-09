"""Tests for the curriculum and the arrival share it follows.

How the experiment applies them during a run is tested with the experiment.
"""

import pytest

from centipede.environment.settings import TargetSettings
from centipede.experiment.arrivals import ArrivalShare
from centipede.experiment.configuration import CurriculumSettings
from centipede.experiment.curriculum import Curriculum


def make_curriculum() -> Curriculum:
    """From the default targets (30-60 mm, ±30°) to 30-200 mm all around."""
    settings = CurriculumSettings.from_section(
        {"final_distance_range_m": [0.03, 0.2], "final_bearing_range_deg": [-180, 180]}
    )
    return Curriculum(settings, TargetSettings.from_section({}), level=0.0)


def test_the_arrival_share_waits_for_a_whole_episode_length_of_windows():
    # Two windows of two worlds; targets per world and minute over them.
    share = ArrivalShare(window_steps=4, episode_steps=8, world_count=2)
    # Endings, in the environment's order: arrived, within a quarter, within
    # half, closer, not closer, left the circle.
    share.add([4, 0, 0, 0, 0, 0])
    assert share.value is None
    share.add([3, 0, 0, 0, 1, 0])
    assert share.value == 7 / 8
    assert share.targets_per_minute == pytest.approx(7 / 2 / (8 * 0.02 / 60))
    share.add([0, 0, 0, 0, 4, 0])  # the first window has left the span
    assert share.value == 3 / 8


def test_the_level_follows_the_arrival_share_within_0_and_1():
    curriculum = make_curriculum()
    curriculum.update(0.75)
    assert curriculum.level == pytest.approx(0.02 * 0.25)

    for _ in range(200):
        curriculum.update(1.0)
    assert curriculum.level == 1.0
    distance, bearing = curriculum.target_ranges()
    assert distance == pytest.approx((0.03, 0.2))
    assert bearing == pytest.approx((-180, 180))

    for _ in range(200):
        curriculum.update(0.0)
    assert curriculum.level == 0.0
    assert curriculum.target_ranges() == ((0.03, 0.06), (-30, 30))


def test_a_level_between_moves_each_end_of_each_range_in_proportion():
    curriculum = make_curriculum()
    curriculum.level = 0.5
    distance, bearing = curriculum.target_ranges()
    assert distance == pytest.approx((0.03, 0.13))
    assert bearing == pytest.approx((-105, 105))
