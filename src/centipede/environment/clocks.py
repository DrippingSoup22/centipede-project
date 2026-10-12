"""The clocks: hands that turn at tempos the segments choose.

The clocks are part of the environment's state, like the targets. With
``clocks``, every segment has one: it sets the clock's tempo with one action
and observes where its hand is. With ``leg_clocks``, every leg has one, whose
hand drives the leg through a fixed step shape (``leg_targets``): the segment
sets each leg's tempo, the size of its steps, and its joints' centres, which
the clocks let follow slowly, as they let its tempo, and observes the step
shape in use. With
``neighbour_clocks``, a segment also observes its neighbours' clocks,
relative to its own; with ``neighbour_tempos``, only their tempos. See
docs/environment.md, "Clocks".
"""

import math
from dataclasses import dataclass

import torch

from centipede.environment.settings import ClockSettings

# How many points of a turn a segment's clock remembers the legs' angles at,
# at the centre of each of as many equal parts of the turn.
REMEMBERED_POINTS = 64
POINT_WIDTH = 2 * math.pi / REMEMBERED_POINTS
# A segment observes each of its clocks as its hand's (cos, sin) and its
# tempo action.
VALUES_PER_CLOCK = 3


@dataclass(frozen=True)
class ClockStep:
    """What the clock costs read after a step; created anew every step.

    ``leg_squared_difference`` is ``(W, N)``: how far the legs were, at the
    points of its turn the hand passed during the step, from where they were
    at the same points one turn earlier: the difference of each leg angle
    squared, in rad², averaged over the angles and the points.
    ``leg_difference_known`` ``(W, N)`` is False where the hand passed only
    points it had not passed since the world restarted: in the first turn.
    Both are None with clocks per leg, which do not remember the legs.
    ``tempo_mismatch`` ``(W, N)`` is the mean, over what each of the
    segment's clocks is compared with, of the tempo difference in octaves
    divided by the largest possible, from 0 to 1, averaged over the
    segment's clocks. A clock per segment is compared with the neighbours the
    segment sees; a leg's clock with its sibling leg's and with the same
    side's legs of those neighbours. With ``coupling = "ahead"``, only the
    neighbours ahead of the segment count.

    With clocks per leg, three values compare the legs' clocks in use
    (``Clocks.clock_in_use``, 3 + J values per leg, each from −1 to 1); all
    are None with a clock per segment. ``tracking_mismatch`` ``(W, N)`` is how
    far each segment's legs' clocks in use are from the same side's legs of
    the segment ahead: the mean, over the two legs and their values, of
    |difference| ÷ 2, from 0 to 1; 0 for the head, with nothing ahead. The
    phase is not compared. ``step_size_tracking_mismatch`` ``(W, N)`` is the
    same over the tempo and the sweep and lift amplitudes only, the centres
    left out. ``change_size`` ``(W, N)`` is how fast each
    segment's legs' clocks in use changed on the step: the mean, over the two
    legs and their values, of (change ÷ the largest change the lag allows
    in one step)², each at most 1; 0 on the first step after a restart.
    ``head_synchrony`` ``(W,)`` is how close the followers' legs' clocks in
    use are to the head's: each follower scores 1 − the mean, over its two
    legs and three values, the tempo and the sweep and lift amplitudes, of
    |difference from the head's same-side leg| ÷ 2; the mean of the
    followers' scores, from 0 to 1. The centres and the phase are not
    compared. ``tempo_gap_ahead`` ``(W, N)``, with clocks per leg, is how far
    each segment's tempo, its first clock's, is from the segment ahead's, in
    octaves; 0 for the head. ``tempo_gap_behind`` likewise from the segment
    behind's; 0 for the rear segment.
    """

    leg_squared_difference: torch.Tensor | None
    leg_difference_known: torch.Tensor | None
    tempo_mismatch: torch.Tensor
    tracking_mismatch: torch.Tensor | None = None
    change_size: torch.Tensor | None = None
    head_synchrony: torch.Tensor | None = None
    step_size_tracking_mismatch: torch.Tensor | None = None
    tempo_gap_ahead: torch.Tensor | None = None
    tempo_gap_behind: torch.Tensor | None = None


def leg_targets(
    phase: torch.Tensor, amplitudes: torch.Tensor, centres: torch.Tensor
) -> torch.Tensor:
    """Every leg joint's target on the step shape, ``(W, N, 2J)`` in action
    order, with ``J`` joints per leg.

    ``phase`` ``(W, N, 2)`` is each leg's clock, the left leg's first;
    ``amplitudes`` ``(W, N, 2, 2)`` holds each leg's sweep and lift
    amplitudes, and ``centres`` ``(W, N, 2, J)`` its joints' centres, sweep,
    lift, knee (and ankle), in action units:

        sweep = sweep centre + sweep amplitude × cos φ
        lift  = lift centre + lift amplitude × max(0, −sin φ)
        knee  = knee centre, and so the ankle

    Over the first half of a turn the foot sweeps back with the leg down
    (stance); over the second it comes forward, lifted (swing). Nothing is
    clipped: a negative sweep amplitude steps backward.
    """
    sweep_amplitude, lift_amplitude = amplitudes.unbind(dim=-1)
    sweep = centres[..., 0] + sweep_amplitude * torch.cos(phase)
    lift = centres[..., 1] + lift_amplitude * (-torch.sin(phase)).clamp(min=0.0)
    return torch.cat(
        (sweep[..., None], lift[..., None], centres[..., 2:]), dim=-1
    ).flatten(start_dim=2)


def lag_rate(time_constant_s: float, step_duration_s: float) -> float:
    """The share of the way a first-order lag covers in one step: ``1 −
    e^(−Δt/τ)``, its exact step for any τ; 1, the whole way, for τ = 0."""
    if not time_constant_s:
        return 1.0
    return 1 - math.exp(-step_duration_s / time_constant_s)


class Clocks:
    """Every world's clocks: one per segment, or with ``per_leg`` one per leg.

    ``clock_phase`` ``(W, N, C)`` is every clock's hand, in rad from 0 to 2π,
    with ``C`` clocks per segment: one, or with ``per_leg`` two, the left
    leg's then the right's. ``clock_tempo_hz`` is the turns per second each
    ran at on the last step, its tempo in use, and ``clock_tempo_octaves`` the
    same in octaves from the middle tempo: a clock per segment's tempo action
    times the range, or a leg's clock's, followed slowly (``follow_actions``).
    All three are overwritten in place by ``reset`` and ``step``. ``phase``,
    ``tempo_hz`` and ``tempo_octaves`` ``(W, N)`` are views of each segment's
    first clock, its own or its left leg's, which the rhythm diagnostics and
    the log read as the segment's clock. ``value_count`` is how many values
    describe a segment's own clocks in its observation, and each neighbour's.

    A clock per segment also compares the legs with its last turn: it
    remembers each segment's leg angles at REMEMBERED_POINTS points of the
    turn, the centres of as many equal parts; the hand passes each point once
    per turn, so every turn replaces what the last one left. Clocks per leg
    also keep ``step_shape`` ``(W, N, 2, 2 + J)``, each leg's step shape in
    use for its ``J`` joints (``leg_joint_count``: 3, or 4 with an ankle),
    which follows the shape actions slowly, as the tempo does
    (``follow_actions``): ``amplitudes`` ``(W, N, 2, 2)``, the sweep and lift
    amplitudes, and ``centres`` ``(W, N, 2, J)``, the joints' centres, are
    views of it. ``clock_change`` ``(W, N, 2, 3 + J)`` is how far each leg's
    clock in use (``clock_in_use``) moved on the last step.
    """

    def __init__(
        self,
        settings: ClockSettings,
        world_count: int,
        segment_count: int,
        observation_radius: int,
        step_duration_s: float,
        device: torch.device | str,
        per_leg: bool = False,
        leg_joint_count: int = 3,
    ) -> None:
        """Allocate the clocks and count what each clock is compared with."""
        self.settings = settings
        self.step_duration_s = step_duration_s
        self.per_leg = per_leg
        shape = (world_count, segment_count, 2 if per_leg else 1)
        self.clock_phase = torch.zeros(shape, dtype=torch.float32, device=device)
        self.clock_tempo_octaves = torch.zeros(
            shape, dtype=torch.float32, device=device
        )
        self.clock_tempo_hz = torch.full(
            shape, settings.middle_tempo_hz, dtype=torch.float32, device=device
        )
        self.phase = self.clock_phase[..., 0]
        self.tempo_octaves = self.clock_tempo_octaves[..., 0]
        self.tempo_hz = self.clock_tempo_hz[..., 0]
        self.value_count = VALUES_PER_CLOCK * shape[2]
        if per_leg:
            self.step_shape = torch.zeros(
                (world_count, segment_count, 2, 2 + leg_joint_count),
                dtype=torch.float32,
                device=device,
            )
            self.amplitudes = self.step_shape[..., :2]
            self.centres = self.step_shape[..., 2:]
            # The worlds whose next step starts the clocks in use at the actions.
            self.restarted = torch.ones(world_count, dtype=torch.bool, device=device)
            # The share of the way to its action the clock in use covers in a
            # step: tempo, amplitudes and centres alike.
            self.lag_rate = lag_rate(settings.clock_time_constant_s, step_duration_s)
            # With a heart, the most a tempo action changes the tempo in one
            # step, in octaves; 0 without one.
            self.heart_step_octaves = (
                settings.tempo_rate_octaves_per_s * step_duration_s
            )
            # How far each leg's clock in use moved on the last step, in the
            # units of ``clock_in_use``.
            self.clock_change = torch.zeros(
                (world_count, segment_count, 2, 3 + leg_joint_count),
                dtype=torch.float32,
                device=device,
            )
        else:
            # The hand passes at most this many points in one step, at the
            # fastest tempo; one more for rounding.
            fastest = settings.middle_tempo_hz * 2**settings.tempo_range_octaves
            most_points = math.ceil(REMEMBERED_POINTS * fastest * step_duration_s) + 1
            self.point_offsets = torch.arange(
                most_points, dtype=torch.float32, device=device
            )
            self.remembered_angles = torch.zeros(
                (world_count, segment_count, REMEMBERED_POINTS, 2 * leg_joint_count),
                dtype=torch.float32,
                device=device,
            )
            self.remembered = torch.zeros(
                (world_count, segment_count, REMEMBERED_POINTS),
                dtype=torch.bool,
                device=device,
            )
        # Starting hands have their own random sequence, so they never change
        # the starting poses or the targets.
        self.generator = torch.Generator(device=device)

        # The neighbours a segment sees, whose tempos its own are compared
        # with: those ahead, and with coupling both ways those behind; a leg's
        # clock is also compared with its sibling's.
        self.neighbour_offsets = range(
            1, min(observation_radius, segment_count - 1) + 1
        )
        self.both_ways = settings.coupling == "both"
        counts = torch.zeros(segment_count, dtype=torch.float32, device=device)
        for offset in self.neighbour_offsets:
            counts[offset:] += 1
            if self.both_ways:
                counts[:-offset] += 1
        self.comparison_counts = (counts + int(per_leg)).clamp(min=1)

    def reset(self, world_mask: torch.Tensor, seed: int | None = None) -> None:
        """Start the masked worlds' clocks at random hands and the middle tempo.

        A ``seed`` restarts the hands' random sequence. The legs' remembered
        angles are forgotten, so the first turn after a restart costs nothing,
        and the legs' step shapes are zero until the next step starts the
        clocks in use at its actions. With a heart, each segment's tempo
        starts instead at a random value, uniform in octaves over
        ``start_tempo_range_hz``, the same for both its legs, drawn after the
        hands.
        """
        if seed is not None:
            self.generator.manual_seed(seed)
        new_phase = (
            2
            * math.pi
            * torch.rand(
                self.clock_phase.shape,
                generator=self.generator,
                device=self.clock_phase.device,
            )
        )
        restarted = world_mask[:, None, None]
        self.clock_phase.copy_(torch.where(restarted, new_phase, self.clock_phase))
        start_octaves = torch.zeros_like(self.clock_tempo_octaves)
        if self.per_leg and self.heart_step_octaves:
            low, high = (
                math.log2(tempo / self.settings.middle_tempo_hz)
                for tempo in self.settings.start_tempo_range_hz
            )
            draw = torch.rand(
                self.clock_phase.shape[:2],
                generator=self.generator,
                device=self.clock_phase.device,
            )
            start_octaves[:] = (low + (high - low) * draw)[..., None]
        self.clock_tempo_octaves.copy_(
            torch.where(restarted, start_octaves, self.clock_tempo_octaves)
        )
        self.clock_tempo_hz.copy_(
            self.settings.middle_tempo_hz * torch.exp2(self.clock_tempo_octaves)
        )
        if self.per_leg:
            self.step_shape.masked_fill_(restarted[..., None], 0.0)
            self.restarted.logical_or_(world_mask)
        else:
            self.remembered.masked_fill_(restarted, False)

    def step(
        self,
        tempo_action: torch.Tensor | None = None,
        leg_angles_before: torch.Tensor | None = None,
        leg_angles: torch.Tensor | None = None,
        foot_load: torch.Tensor | None = None,
        foot_down: torch.Tensor | None = None,
    ) -> ClockStep:
        """Turn every hand for one step at its clock's tempo in use.

        A clock per segment takes its tempo at once from ``tempo_action``
        ``(W, N, 1)``, from −1 to 1; a leg's clock turns at the tempo that
        ``follow_actions`` left it. A clock per segment also compares the legs
        with its last turn, from ``leg_angles_before`` and ``leg_angles``
        ``(W, N, 2J)``, the leg angles before and after the step (see
        ``_compare_legs``). With load
        feedback σ, a leg's clock whose foot bears the load ``N``, from 0 to 1
        in ``foot_load`` ``(W, N, 2)``, turns at ``ω + σ N cos φ`` rad/s
        instead of its tempo's ``ω``: before mid-stance (φ = π/2) the hand
        hurries toward it, after it the hand is held back (Owaki and Ishiguro,
        2017, with stance on the first half of the turn). With σ N ≥ ω it
        would stop where ``cos φ = −ω / (σ N)`` until its load fell, for good
        with a contact flag, since the foot leaves the ground only in swing;
        the settings keep σ below the slowest tempo's ω, and N is at most 1.

        With the follow nudge ρ, a follower's leg in the second half of
        stance (sin φ ≥ 0, cos φ < 0) also turns ρ |cos φ| rad/s slower while
        the same-side foot of the segment ahead is down, in ``foot_down``
        ``(W, N, 2)`` from before the step, and as much faster while it is
        lifted, so that it lifts soon after the leg ahead. The head, with
        nothing ahead, is not nudged. A held hand may stop, or turn back a
        little, until the foot ahead lifts.
        """
        if not self.per_leg:
            self.clock_tempo_octaves.copy_(
                tempo_action * self.settings.tempo_range_octaves
            )
        self.clock_tempo_hz.copy_(
            self.settings.middle_tempo_hz * torch.exp2(self.clock_tempo_octaves)
        )
        speed = 2 * math.pi * self.clock_tempo_hz
        load_feedback = self.settings.load_feedback_rad_per_s
        if load_feedback:
            speed = speed + load_feedback * foot_load * torch.cos(self.clock_phase)
        follow_nudge = self.settings.follow_nudge_rad_per_s
        if follow_nudge:
            late_stance = (-torch.cos(self.clock_phase)).clamp(min=0.0) * (
                torch.sin(self.clock_phase) >= 0
            )
            ahead_lifted = 1 - 2 * foot_down[:, :-1].float()  # -1 down, +1 lifted
            nudge = torch.zeros_like(speed)
            nudge[:, 1:] = follow_nudge * late_stance[:, 1:] * ahead_lifted
            speed = speed + nudge
        turned = speed * self.step_duration_s
        leg_squared_difference = leg_difference_known = None
        if not self.per_leg:
            leg_squared_difference, leg_difference_known = self._compare_legs(
                turned[..., 0], leg_angles_before, leg_angles
            )
        self.clock_phase.copy_((self.clock_phase + turned).remainder(2 * math.pi))
        tracking_mismatch = change_size = head_synchrony = step_size_mismatch = None
        tempo_gap_ahead = tempo_gap_behind = None
        if self.per_leg:
            tracking_mismatch, change_size, head_synchrony, step_size_mismatch = (
                self._compare_clocks_in_use()
            )
            tempo = self.clock_tempo_octaves[..., 0]
            tempo_gap_ahead = torch.zeros_like(tempo)
            tempo_gap_ahead[:, 1:] = (tempo[:, 1:] - tempo[:, :-1]).abs()
            tempo_gap_behind = torch.zeros_like(tempo)
            tempo_gap_behind[:, :-1] = tempo_gap_ahead[:, 1:]
        return ClockStep(
            leg_squared_difference,
            leg_difference_known,
            self._tempo_mismatch(),
            tracking_mismatch,
            change_size,
            head_synchrony,
            step_size_mismatch,
            tempo_gap_ahead,
            tempo_gap_behind,
        )

    def clock_in_use(self) -> torch.Tensor:
        """Every leg's clock in use, ``(W, N, 2, 3 + J)``, each from −1 to 1.

        For each leg, the left leg's first: its tempo in use in octaves over
        ``tempo_range_octaves``, the tempo action it follows, then its step
        shape in use, in action units.
        """
        tempo = self.clock_tempo_octaves / self.settings.tempo_range_octaves
        return torch.cat((tempo[..., None], self.step_shape), dim=-1)

    def follow_actions(self, leg_clock_actions: torch.Tensor) -> None:
        """Move every leg's clock in use toward its actions, before ``step``.

        ``leg_clock_actions`` is ``(W, N, 2, 3 + J)``: for each leg, its tempo
        action, then its sweep and lift amplitudes and its joints' centres,
        in action units. The tempo in octaves and the step shape
        follow their actions as a first-order lag (``lag_rate``) with
        ``clock_time_constant_s``, which covers 1 − e^(−1), 63%, of a sudden
        change in one time constant. On the first step after a restart, the
        clock in use starts at the actions, and its change counts zero. With a
        heart, the tempo action instead changes the tempo from where it is, by
        up to ``heart_step_octaves`` per step, up to ``max_tempo_hz`` when set.
        """
        before = self.clock_in_use()
        restarted = self.restarted[:, None, None]
        if self.heart_step_octaves:
            self.clock_tempo_octaves.add_(
                leg_clock_actions[..., 0] * self.heart_step_octaves
            )
            if self.settings.max_tempo_hz:
                self.clock_tempo_octaves.clamp_(
                    max=math.log2(
                        self.settings.max_tempo_hz / self.settings.middle_tempo_hz
                    )
                )
        else:
            tempo_octaves = (
                leg_clock_actions[..., 0] * self.settings.tempo_range_octaves
            )
            self.clock_tempo_octaves.copy_(
                torch.where(
                    restarted,
                    tempo_octaves,
                    self.clock_tempo_octaves.lerp(tempo_octaves, self.lag_rate),
                )
            )
        shape_actions = leg_clock_actions[..., 1:]
        self.step_shape.copy_(
            torch.where(
                restarted[..., None],
                shape_actions,
                self.step_shape.lerp(shape_actions, self.lag_rate),
            )
        )
        self.clock_change.copy_(self.clock_in_use() - before)
        self.clock_change.masked_fill_(restarted[..., None], 0.0)
        self.restarted.fill_(False)

    def _compare_clocks_in_use(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``ClockStep.tracking_mismatch`` and ``change_size``, ``(W, N)``
        each, ``head_synchrony``, ``(W,)``, and
        ``step_size_tracking_mismatch``, ``(W, N)``.

        Every value of a clock in use lies from −1 to 1, so a difference of 2
        is the largest. In one step the lag moves a value by at most its rate
        times that range, the unit of the change.
        """
        clock = self.clock_in_use()
        tracking_mismatch = clock.new_zeros(clock.shape[:2])
        step_size_mismatch = clock.new_zeros(clock.shape[:2])
        difference = ((clock[:, 1:] - clock[:, :-1]) / 2).abs()
        tracking_mismatch[:, 1:] = difference.mean(dim=(-2, -1))
        # The tempo and the two amplitudes, the first three values.
        step_size_mismatch[:, 1:] = difference[..., :3].mean(dim=(-2, -1))
        largest_change = 2 * self.lag_rate
        change_size = (
            (self.clock_change / largest_change)
            .square()
            .clamp(max=1.0)
            .mean(dim=(-2, -1))
        )
        # The tempo and the two amplitudes, the first three values of a leg's
        # clock in use, of every follower against the head's.
        tempo_and_amplitudes = clock[..., :3]
        head_distance = (tempo_and_amplitudes[:, 1:] - tempo_and_amplitudes[:, :1]) / 2
        head_synchrony = 1 - head_distance.abs().mean(dim=(1, 2, 3))
        return tracking_mismatch, change_size, head_synchrony, step_size_mismatch

    def _compare_legs(
        self,
        turned: torch.Tensor,
        leg_angles_before: torch.Tensor,
        leg_angles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare the legs with the last turn as the hand turns by ``turned``.

        Returns ``ClockStep``'s two leg values, before the hand has moved. At
        each point the hand passes, the legs are taken to be between their
        angles before and after the step, in proportion to how far the hand
        had come; they are compared with the angles remembered there one turn
        earlier, then remembered in their place.
        """
        start = self.phase

        # The points the hand may pass in the step, (W, N, P): the points'
        # centres are at (k + 1/2) point widths, and the first is the first
        # after the hand's start. At each point passed, the legs have made the
        # share ``way`` of the step's movement.
        first_point = torch.floor(start / POINT_WIDTH - 0.5) + 1
        point = first_point[..., None] + self.point_offsets
        centre = (point + 0.5) * POINT_WIDTH
        passed = centre <= (start + turned)[..., None]
        way = (centre - start[..., None]) / turned[..., None]
        movement = leg_angles - leg_angles_before
        angles = leg_angles_before[:, :, None] + way[..., None] * movement[:, :, None]

        # Compare the legs with what was remembered at the points passed, then
        # remember them in its place.
        index = point.long().remainder(REMEMBERED_POINTS)
        angle_index = index[..., None].expand(*index.shape, leg_angles.shape[-1])
        remembered_angles = self.remembered_angles.gather(2, angle_index)
        remembered = self.remembered.gather(2, index)
        known = passed & remembered
        point_difference = (angles - remembered_angles).square().mean(dim=-1)
        compared = known.sum(dim=-1)
        squared_difference = (point_difference * known).sum(dim=-1)
        self.remembered_angles.scatter_(
            2, angle_index, torch.where(passed[..., None], angles, remembered_angles)
        )
        self.remembered.scatter_(2, index, remembered | passed)
        return squared_difference / compared.clamp(min=1), compared > 0

    def _tempo_mismatch(self) -> torch.Tensor:
        """``ClockStep.tempo_mismatch``, ``(W, N)``, from the step's tempos.

        A tempo difference of two octaves, the largest, counts 1.
        """
        largest = 2 * self.settings.tempo_range_octaves
        octaves = self.clock_tempo_octaves
        mismatch = torch.zeros_like(octaves)
        for offset in self.neighbour_offsets:
            # Each pair's difference, paid by the segment behind and, with
            # coupling both ways, by the one ahead.
            difference = (octaves[:, :-offset] - octaves[:, offset:]).abs() / largest
            mismatch[:, offset:] += difference
            if self.both_ways:
                mismatch[:, :-offset] += difference
        if self.per_leg:
            mismatch += (octaves[..., :1] - octaves[..., 1:]).abs() / largest
        return (mismatch / self.comparison_counts[:, None]).mean(dim=-1)

    def observation_values(self) -> torch.Tensor:
        """Each segment's own clocks as it sees them, ``(W, N, value_count)``.

        For each clock, the left leg's first: its hand as ``cos`` and ``sin``,
        so that the end of a turn and the start of the next look alike, and
        its tempo in use in action units, from −1 to 1.
        """
        return torch.stack(
            (
                torch.cos(self.clock_phase),
                torch.sin(self.clock_phase),
                self.clock_tempo_octaves / self.settings.tempo_range_octaves,
            ),
            dim=-1,
        ).flatten(start_dim=2)

    def neighbour_values(self) -> torch.Tensor:
        """Each neighbour's clocks as a segment sees them, ``(W, N, 2k, values)``.

        ``values`` is ``value_count``. The ``k`` segments ahead, nearest
        first, then the ``k`` behind, for the observation radius ``k``. Each
        clock relative to the segment's own, a leg's to the same side's:
        ``cos`` and ``sin`` of the neighbour's hand minus its own, and the
        neighbour's tempo minus its own in octaves, over the largest
        difference, from −1 to 1. A neighbour that does not exist gives zeros,
        and so do those behind with ``coupling = "ahead"``.
        """
        largest = 2 * self.settings.tempo_range_octaves
        world_count, segment_count, _ = self.clock_phase.shape
        k = len(self.neighbour_offsets)
        values = self.clock_phase.new_zeros(
            (world_count, segment_count, 2 * k, self.value_count)
        )

        def relative(neighbours: slice, segments: slice) -> torch.Tensor:
            """The neighbours' clocks relative to the segments', ``(W, n, values)``."""
            phase_difference = (
                self.clock_phase[:, neighbours] - self.clock_phase[:, segments]
            )
            tempo_difference = (
                self.clock_tempo_octaves[:, neighbours]
                - self.clock_tempo_octaves[:, segments]
            )
            return torch.stack(
                (
                    torch.cos(phase_difference),
                    torch.sin(phase_difference),
                    tempo_difference / largest,
                ),
                dim=-1,
            ).flatten(start_dim=2)

        for column, offset in enumerate(self.neighbour_offsets):
            # Segment i's neighbour ahead is i − offset, behind it i + offset.
            values[:, offset:, column] = relative(
                slice(None, -offset), slice(offset, None)
            )
            if self.both_ways:
                values[:, :-offset, k + column] = relative(
                    slice(offset, None), slice(None, -offset)
                )
        return values

    def neighbour_tempos(self) -> torch.Tensor:
        """Each neighbour's tempo as a segment hears it, ``(W, N, 2k)``.

        The ``k`` segments ahead, nearest first, then the ``k`` behind, for the
        observation radius ``k``, whatever the coupling: the neighbour's tempo,
        its first clock's, minus the segment's own, in octaves; 0 for a
        neighbour that does not exist. Only the tempo, never the hand: how fast
        a neighbour steps, not when, so the offsets between neighbours stay
        free.
        """
        tempo = self.clock_tempo_octaves[..., 0]
        k = len(self.neighbour_offsets)
        values = tempo.new_zeros((*tempo.shape, 2 * k))
        for column, offset in enumerate(self.neighbour_offsets):
            # Segment i's neighbour ahead is i − offset, behind it i + offset.
            values[:, offset:, column] = tempo[:, :-offset] - tempo[:, offset:]
            values[:, :-offset, k + column] = tempo[:, offset:] - tempo[:, :-offset]
        return values
