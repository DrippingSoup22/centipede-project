"""Tests for the clocks, without physics."""

import math

import pytest
import torch

from centipede.environment.clocks import Clocks, leg_targets
from centipede.environment.settings import ClockSettings

STEP_S = 0.02


def clocks(
    middle_tempo_hz=2.0,
    world_count=2,
    segment_count=3,
    per_leg=False,
    feedback=0.0,
    centre_time_constant_s=0.0,
    amplitude_time_constant_s=0.0,
    coupling="both",
):
    settings = ClockSettings(
        middle_tempo_hz=middle_tempo_hz,
        tempo_range_octaves=1.0,
        load_feedback_rad_per_s=feedback,
        centre_time_constant_s=centre_time_constant_s,
        amplitude_time_constant_s=amplitude_time_constant_s,
        coupling=coupling,
    )
    return Clocks(
        settings, world_count, segment_count, 1, STEP_S, "cpu", per_leg=per_leg
    )


def test_hands_turn_at_the_chosen_tempo_and_restart_at_random():
    clock = clocks()
    every_world = torch.ones(2, dtype=torch.bool)
    clock.reset(every_world, seed=1)
    start = clock.phase.clone()

    still = torch.zeros(2, 3, 6)
    clock.step(torch.tensor([[1.0, 0.0, -1.0]] * 2)[..., None], still, still)

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
    middle_tempo = torch.zeros(1, 3, 1)

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
    head_faster = torch.tensor([[[1.0], [0.0], [0.0]]])
    result = clock.step(head_faster, legs(35), legs(35))
    assert result.tempo_mismatch[0].tolist() == pytest.approx([0.5, 0.25, 0.0])
    # With coupling ahead, each segment is compared only with the one ahead of
    # it: the pair's difference is segment 1's alone, and the head pays nothing.
    ahead = clocks(middle_tempo_hz=1.5625, world_count=1, coupling="ahead")
    result = ahead.step(head_faster, legs(35), legs(35))
    assert result.tempo_mismatch[0].tolist() == pytest.approx([0.0, 0.5, 0.0])


def test_each_segment_sees_its_neighbours_clocks_relative_to_its_own():
    clock = clocks(world_count=1)
    clock.phase.copy_(torch.tensor([[0.1, 0.5, 1.7]]))
    clock.tempo_octaves.copy_(torch.tensor([[1.0, 0.0, -0.5]]))

    values = clock.neighbour_values()

    def seen(phase_difference, tempo_difference):
        return [
            math.cos(phase_difference),
            math.sin(phase_difference),
            tempo_difference,
        ]

    # The segment ahead, then the one behind; tempo differences over 2 octaves.
    missing = [0.0, 0.0, 0.0]
    expected = [
        [missing, seen(0.4, -0.5)],
        [seen(-0.4, 0.5), seen(1.2, -0.25)],
        [seen(-1.2, 0.25), missing],
    ]
    assert values.shape == (1, 3, 2, 3)
    assert torch.allclose(values[0], torch.tensor(expected), atol=1e-6)

    # With coupling ahead, the segments behind are seen as zeros.
    ahead = clocks(world_count=1, coupling="ahead")
    ahead.phase.copy_(clock.phase)
    ahead.tempo_octaves.copy_(clock.tempo_octaves)
    only_ahead = torch.tensor(expected)
    only_ahead[:, 1] = 0.0
    assert torch.allclose(ahead.neighbour_values()[0], only_ahead, atol=1e-6)


def test_the_step_shape_pushes_back_on_the_ground_and_lifts_the_leg_coming_forward():
    """The left leg at four points of its turn; the right leg's shape is zero."""
    phase = torch.zeros(4, 1, 2)
    phase[:, 0, 0] = torch.tensor([0.0, 0.5, 1.0, 1.5]) * math.pi
    shape_actions = torch.zeros(4, 1, 2, 5)
    # Sweep and lift amplitudes, then the sweep, lift and knee centres.
    shape_actions[:, 0, 0] = torch.tensor([0.5, 0.4, 0.1, -0.2, -0.3])

    targets = leg_targets(phase, shape_actions[..., :2], shape_actions[..., 2:])

    expected_left = [
        [0.6, -0.2, -0.3],  # φ = 0: fully forward, down
        [0.1, -0.2, -0.3],  # π/2: mid-stance
        [-0.4, -0.2, -0.3],  # π: fully back
        [0.1, 0.2, -0.3],  # 3π/2: mid-swing, lifted
    ]
    assert torch.allclose(targets[:, 0, :3], torch.tensor(expected_left), atol=1e-6)
    assert not targets[:, 0, 3:].any()


def test_leg_clocks_turn_each_leg_and_compare_it_with_its_sibling_and_its_side():
    """Three segments, each seeing one neighbour on each side: tempo actions
    1, 0, 0 on the left and 0, 0, -1 on the right."""
    clock = clocks(world_count=1, per_leg=True)
    clock.reset(torch.ones(1, dtype=torch.bool), seed=1)
    assert not torch.equal(clock.clock_phase[..., 0], clock.clock_phase[..., 1])
    start = clock.clock_phase.clone()
    tempo_action = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, -1.0]]])

    result = clock.step(tempo_action)

    tempo = torch.tensor([[[4.0, 2.0], [2.0, 2.0], [2.0, 1.0]]])
    turned = (clock.clock_phase - start).remainder(2 * math.pi)
    assert torch.allclose(turned, 2 * math.pi * tempo * STEP_S, rtol=1e-5)
    assert torch.equal(clock.phase, clock.clock_phase[..., 0])  # the left legs'
    assert result.leg_difference_known is None
    # Each leg against its sibling and the same side's neighbours, in octaves
    # over two; the segment pays the mean of its legs. The head's left leg
    # differs by an octave from its sibling and from segment 1's left leg:
    # (0.5 + 0.5) / 2; its right leg from its sibling only: 0.5 / 2.
    assert result.tempo_mismatch[0].tolist() == pytest.approx(
        [(0.5 + 0.25) / 2, 1 / 6, (0.25 + 0.5) / 2]
    )

    # Each segment sees its legs' clocks, the left's first, and its
    # neighbours' clocks leg by leg, each relative to its own side.
    own = clock.observation_values()
    assert own.shape == (1, 3, 6)
    assert own[0, :, 2].tolist() == [1.0, 0.0, 0.0]
    assert own[0, :, 5].tolist() == [0.0, 0.0, -1.0]
    ahead_of_segment_1 = clock.neighbour_values()[0, 1, 0]
    difference = clock.clock_phase[0, 0] - clock.clock_phase[0, 1]
    expected = [
        *(math.cos(difference[0]), math.sin(difference[0]), 0.5),
        *(math.cos(difference[1]), math.sin(difference[1]), 0.0),
    ]
    assert torch.allclose(ahead_of_segment_1, torch.tensor(expected), atol=1e-6)


def test_load_feedback_holds_a_loaded_legs_clock_past_mid_stance():
    """At 2 Hz, omega = 4 pi rad/s; with sigma = 50 rad/s a loaded leg's hand
    settles where cos phi = -omega / sigma. The right feet are in the air."""
    held = clocks(world_count=1, per_leg=True, feedback=50.0)
    free = clocks(world_count=1, per_leg=True)
    for clock in (held, free):
        clock.reset(torch.ones(1, dtype=torch.bool), seed=3)
    start = held.clock_phase.clone()
    left_feet_down = torch.tensor([[[True, False]] * 3])
    middle_tempo = torch.zeros(1, 3, 2)

    for _ in range(50):
        held.step(middle_tempo, foot_contact=left_feet_down)
        free.step(middle_tempo, foot_contact=left_feet_down)

    def angle_between(first, second):
        return ((first - second + math.pi).remainder(2 * math.pi) - math.pi).abs()

    settled = torch.full((1, 3), math.acos(-4 * math.pi / 50))
    assert angle_between(held.clock_phase[..., 0], settled).max() < 1e-4
    # 50 steps at 2 Hz are two whole turns: an unloaded leg, or any leg without
    # load feedback, is back where it started.
    assert angle_between(held.clock_phase[..., 1], start[..., 1]).max() < 1e-4
    assert angle_between(free.clock_phase, start).max() < 1e-4


def test_the_step_shape_starts_at_the_actions_then_follows_with_its_time_constants():
    """The amplitudes follow with 0.5 s, the centres at once (τ = 0)."""
    clock = clocks(world_count=1, per_leg=True, amplitude_time_constant_s=0.5)
    every_world = torch.ones(1, dtype=torch.bool)
    clock.reset(every_world, seed=0)
    clock.follow_step_shape(torch.full((1, 3, 2, 5), 0.2))
    assert torch.allclose(clock.step_shape, torch.tensor(0.2))  # where it asks

    clock.follow_step_shape(torch.full((1, 3, 2, 5), 1.2))

    # One 20 ms step covers 1 - e^(-0.02 / 0.5) of the way.
    covered = 0.2 + 1 - math.exp(-0.02 / 0.5)
    assert torch.allclose(clock.amplitudes, torch.tensor(covered))
    assert torch.allclose(clock.centres, torch.tensor(1.2))
    clock.reset(every_world)
    assert not clock.step_shape.any()  # a restart forgets the shape ...
    clock.follow_step_shape(torch.full((1, 3, 2, 5), -0.5))
    assert torch.allclose(clock.step_shape, torch.tensor(-0.5))  # ... and starts afresh
