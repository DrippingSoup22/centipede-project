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
    clock_time_constant_s=0.0,
    coupling="both",
    follow_nudge=0.0,
):
    settings = ClockSettings(
        middle_tempo_hz=middle_tempo_hz,
        tempo_range_octaves=1.0,
        load_feedback_rad_per_s=feedback,
        load_feedback_source="contact",
        clock_time_constant_s=clock_time_constant_s,
        coupling=coupling,
        follow_nudge_rad_per_s=follow_nudge,
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
    clock_actions = torch.zeros(1, 3, 2, 6)
    clock_actions[..., 0] = tempo_action

    clock.follow_actions(clock_actions)
    result = clock.step()
    assert not result.change_size.any()  # the first step after a restart

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
    # Each follower's clocks in use against the same side's of the segment
    # ahead: one tempo of the twelve values differs by 1, (1 / 2) / 12.
    assert result.tracking_mismatch[0].tolist() == pytest.approx([0.0, 1 / 24, 1 / 24])

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


def test_head_synchrony_compares_the_followers_tempos_and_amplitudes_with_the_head():
    """Each follower's legs against the head's same side; the centres do not
    count."""
    clock = clocks(world_count=1, per_leg=True)
    clock.reset(torch.ones(1, dtype=torch.bool), seed=0)
    clock_actions = torch.zeros(1, 3, 2, 6)
    clock_actions[0, 1, 0, 0] = 1.0  # segment 1's left tempo, an octave up
    clock_actions[0, 2, 1, 2] = -0.5  # segment 2's right lift amplitude
    clock_actions[0, 1:, :, 3:] = 1.0  # the followers' centres

    clock.follow_actions(clock_actions)
    result = clock.step()

    # Over two legs and three values: 1 - (1 / 2) / 6 and 1 - (0.5 / 2) / 6.
    scores = [1 - 0.5 / 6, 1 - 0.25 / 6]
    assert result.head_synchrony.tolist() == pytest.approx([sum(scores) / 2])


def test_load_feedback_holds_a_loaded_legs_clock_past_mid_stance():
    """At 2 Hz, omega = 4 pi rad/s; with sigma = 50 rad/s a loaded leg's hand
    settles where cos phi = -omega / sigma. The right feet are in the air."""
    held = clocks(world_count=1, per_leg=True, feedback=50.0)
    free = clocks(world_count=1, per_leg=True)
    for clock in (held, free):
        clock.reset(torch.ones(1, dtype=torch.bool), seed=3)
    start = held.clock_phase.clone()
    left_feet_down = torch.tensor([[[1.0, 0.0]] * 3])

    for _ in range(50):
        held.step(foot_load=left_feet_down)
        free.step(foot_load=left_feet_down)

    def angle_between(first, second):
        return ((first - second + math.pi).remainder(2 * math.pi) - math.pi).abs()

    settled = torch.full((1, 3), math.acos(-4 * math.pi / 50))
    assert angle_between(held.clock_phase[..., 0], settled).max() < 1e-4
    # 50 steps at 2 Hz are two whole turns: an unloaded leg, or any leg without
    # load feedback, is back where it started.
    assert angle_between(held.clock_phase[..., 1], start[..., 1]).max() < 1e-4
    assert angle_between(free.clock_phase, start).max() < 1e-4


def test_a_half_load_pushes_a_hand_half_as_hard():
    """A foot that shares its segment's weight with the other bears 0.5:
    its hand turns by (omega + sigma / 2 cos phi) times the step."""
    clock = clocks(world_count=1, per_leg=True, feedback=6.0)
    clock.reset(torch.ones(1, dtype=torch.bool), seed=4)
    start = clock.clock_phase.clone()

    clock.step(foot_load=torch.full((1, 3, 2), 0.5))

    speed = 4 * math.pi + 6.0 / 2 * torch.cos(start)
    expected = (start + speed * STEP_S).remainder(2 * math.pi)
    assert torch.allclose(clock.clock_phase, expected, atol=1e-5)


def test_the_follow_nudge_holds_a_leg_in_late_stance_while_the_foot_ahead_is_down():
    """Every hand at 3π/4, late stance, where |cos φ| = √2 / 2. The left feet
    of the head and segment 1 are down, the right feet lifted: segment 1's
    and 2's left legs are held, their right legs hurried, the head's not
    nudged. A hand in early stance or in swing is never nudged."""
    clock = clocks(world_count=1, per_leg=True, follow_nudge=8.0)
    start = torch.full((1, 3, 2), 0.75 * math.pi)
    feet_down = torch.tensor([[[True, False]] * 3])
    for phase, nudged in ((start, True), (start - math.pi / 2, False)):
        clock.clock_phase.copy_(phase)
        clock.step(foot_down=feet_down)
        turned = clock.clock_phase - phase
        nudge = (
            8.0 * math.sqrt(0.5) * torch.tensor([[0.0, 0.0], [-1.0, 1.0], [-1.0, 1.0]])
        )
        expected = (4 * math.pi + nudge * nudged) * STEP_S
        assert torch.allclose(turned, expected, atol=1e-5)


def test_the_clock_in_use_starts_at_the_actions_then_follows_with_its_time_constant():
    """The tempo and the step shape alike follow with 0.5 s."""
    clock = clocks(world_count=1, per_leg=True, clock_time_constant_s=0.5)
    every_world = torch.ones(1, dtype=torch.bool)
    clock.reset(every_world, seed=0)
    clock.follow_actions(torch.full((1, 3, 2, 6), 0.2))
    assert torch.allclose(clock.step_shape, torch.tensor(0.2))  # where it asks
    assert torch.allclose(clock.clock_tempo_octaves, torch.tensor(0.2))

    clock.follow_actions(torch.full((1, 3, 2, 6), -0.8))
    start = clock.clock_phase.clone()
    result = clock.step()

    # One 20 ms step covers 1 - e^(-0.02 / 0.5) of the way, and the hand
    # turns at the tempo in use.
    covered = 0.2 - (1 - math.exp(-0.02 / 0.5))
    assert torch.allclose(clock.step_shape, torch.tensor(covered))
    assert torch.allclose(clock.clock_tempo_octaves, torch.tensor(covered))
    # A change of half the range, of which one step covers its share: half
    # the largest change the lag allows, (1 / 2)².
    assert torch.allclose(result.change_size, torch.tensor(0.25))
    turned = (clock.clock_phase - start).remainder(2 * math.pi)
    assert torch.allclose(turned, torch.tensor(2 * math.pi * 2 * 2**covered * STEP_S))
    clock.reset(every_world)
    assert not clock.step_shape.any()  # a restart forgets the shape ...
    clock.follow_actions(torch.full((1, 3, 2, 6), -0.5))
    assert torch.allclose(clock.step_shape, torch.tensor(-0.5))  # ... and starts afresh
    assert torch.allclose(clock.clock_tempo_octaves, torch.tensor(-0.5))


def test_a_heart_starts_at_a_random_tempo_and_its_action_changes_it():
    """Both legs share the tempo, which starts between 1 and 4 Hz and moves by
    the action times one octave per second; each follower's gap is to the
    segment ahead."""
    settings = ClockSettings(
        middle_tempo_hz=2.0,
        tempo_range_octaves=1.0,
        load_feedback_rad_per_s=0.0,
        load_feedback_source="contact",
        clock_time_constant_s=0.0,
        coupling="ahead",
        segment_tempo=True,
        tempo_rate_octaves_per_s=1.0,
        start_tempo_range_hz=(1.0, 4.0),
    )
    heart = Clocks(settings, 2, 3, 1, STEP_S, "cpu", per_leg=True)
    heart.reset(torch.ones(2, dtype=torch.bool), seed=0)
    start = heart.clock_tempo_octaves.clone()
    actions = torch.zeros(2, 3, 2, 6)
    actions[0, ..., 0] = 1.0
    actions[1, ..., 0] = -0.5

    heart.follow_actions(actions)
    step = heart.step()

    assert torch.equal(start[..., 0], start[..., 1])
    assert ((start >= -1) & (start <= 1)).all() and start[0, :, 0].unique().numel() == 3
    change = torch.tensor([1.0, -0.5])[:, None, None] * STEP_S
    assert torch.allclose(heart.clock_tempo_octaves, start + change)
    tempo = heart.clock_tempo_octaves[..., 0]
    assert torch.allclose(
        step.tempo_gap_ahead[:, 1:], (tempo[:, 1:] - tempo[:, :-1]).abs()
    )
    assert not step.tempo_gap_ahead[:, 0].any()
