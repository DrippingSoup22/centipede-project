"""Tests for the segments' clocks, without physics."""

import math

import pytest
import torch

from centipede.environment.clocks import Clocks
from centipede.environment.settings import ClockSettings

STEP_S = 0.02


def clocks(middle_tempo_hz=2.0, world_count=2, segment_count=3) -> Clocks:
    settings = ClockSettings(middle_tempo_hz=middle_tempo_hz, tempo_range_octaves=1.0)
    return Clocks(settings, world_count, segment_count, 1, STEP_S, "cpu")


def test_hands_turn_at_the_chosen_tempo_and_restart_at_random():
    clock = clocks()
    every_world = torch.ones(2, dtype=torch.bool)
    clock.reset(every_world, seed=1)
    start = clock.phase.clone()

    still = torch.zeros(2, 3, 6)
    clock.step(torch.tensor([[1.0, 0.0, -1.0]] * 2), still, still)

    assert clock.tempo_hz[0].tolist() == pytest.approx([4.0, 2.0, 1.0])
    turned = (clock.phase - start).remainder(2 * math.pi)
    assert turned[0].tolist() == pytest.approx(
        [2 * math.pi * tempo * STEP_S for tempo in (4, 2, 1)], rel=1e-5
    )
    assert clock.observation_values()[0, :, 2].tolist() == [1.0, 0.0, -1.0]

    # A restart draws new hands and the middle tempo, only where asked.
    before = clock.phase.clone()
    clock.reset(torch.tensor([True, False]))
    assert not torch.equal(clock.phase[0], before[0])
    assert torch.equal(clock.phase[1], before[1])
    assert clock.tempo_hz[0].tolist() == [2.0] * 3
    # A seed repeats the hands.
    clock.reset(every_world, seed=1)
    assert torch.equal(clock.phase, start)


def test_legs_are_compared_with_the_last_turn_and_tempos_with_the_neighbours():
    """At 1.5625 turns per second a turn takes exactly 32 steps, in which the
    hand passes two of the 64 remembered points; it starts a quarter of a point
    in, so that no point lies exactly where a step starts or ends. The legs
    turn steadily, 0.01 rad per step, so the angles between two steps are
    exact."""
    clock = clocks(middle_tempo_hz=1.5625, world_count=1)
    clock.reset(torch.ones(1, dtype=torch.bool), seed=0)
    clock.phase.fill_(2 * math.pi / 64 / 4)
    middle_tempo = torch.zeros(1, 3)

    def legs(step):
        return torch.full((1, 3, 6), 0.01 * step)

    for step in range(1, 33):
        result = clock.step(middle_tempo, legs(step - 1), legs(step))
        assert not result.leg_difference_known.any()  # the first turn
    for step in range(33, 36):
        result = clock.step(middle_tempo, legs(step - 1), legs(step))
        # Every point is compared with the last turn, 32 steps earlier.
        assert result.leg_difference_known.all()
        assert torch.allclose(
            result.leg_squared_difference, torch.full((1, 3), 0.32**2)
        )

    # The tempo is compared with the neighbours each segment sees, in octaves
    # over the largest difference: the head and segment 1 differ by one octave.
    result = clock.step(torch.tensor([[1.0, 0.0, 0.0]]), legs(35), legs(35))
    assert result.tempo_mismatch[0].tolist() == pytest.approx([0.5, 0.25, 0.0])
