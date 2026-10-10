"""The clocks: hands that turn at tempos the segments choose.

The clocks are part of the environment's state, like the targets. With
``clocks``, every segment has one: it sets the clock's tempo with one action
and observes where its hand is. With ``leg_clocks``, every leg has one, whose
hand drives the leg through a fixed step shape (``leg_targets``): the segment
sets each leg's tempo, the size of its steps, and its joints' centres, which
the clocks let follow slowly. With ``neighbour_clocks``, a segment also
observes its neighbours' clocks, relative to its own. See
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
    squared, in rad², averaged over the six angles and the points.
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
    """

    leg_squared_difference: torch.Tensor | None
    leg_difference_known: torch.Tensor | None
    tempo_mismatch: torch.Tensor


def leg_targets(
    phase: torch.Tensor, amplitudes: torch.Tensor, centres: torch.Tensor
) -> torch.Tensor:
    """Every leg joint's target on the step shape, ``(W, N, 6)`` in action order.

    ``phase`` ``(W, N, 2)`` is each leg's clock, the left leg's first;
    ``amplitudes`` ``(W, N, 2, 2)`` holds each leg's sweep and lift
    amplitudes, and ``centres`` ``(W, N, 2, 3)`` its sweep, lift and knee
    centres, in action units:

        sweep = sweep centre + sweep amplitude × cos φ
        lift  = lift centre + lift amplitude × max(0, −sin φ)
        knee  = knee centre

    Over the first half of a turn the foot sweeps back with the leg down
    (stance); over the second it comes forward, lifted (swing). Nothing is
    clipped: a negative sweep amplitude steps backward.
    """
    sweep_amplitude, lift_amplitude = amplitudes.unbind(dim=-1)
    sweep_centre, lift_centre, knee_centre = centres.unbind(dim=-1)
    sweep = sweep_centre + sweep_amplitude * torch.cos(phase)
    lift = lift_centre + lift_amplitude * (-torch.sin(phase)).clamp(min=0.0)
    return torch.stack((sweep, lift, knee_centre), dim=-1).flatten(start_dim=2)


class Clocks:
    """Every world's clocks: one per segment, or with ``per_leg`` one per leg.

    ``clock_phase`` ``(W, N, C)`` is every clock's hand, in rad from 0 to 2π,
    with ``C`` clocks per segment: one, or with ``per_leg`` two, the left
    leg's then the right's. ``clock_tempo_hz`` is the turns per second each
    ran at on the last step, and ``clock_tempo_octaves`` the same in octaves
    from the middle tempo: the tempo action times the range. All three are
    overwritten in place by ``reset`` and ``step``. ``phase``, ``tempo_hz``
    and ``tempo_octaves`` ``(W, N)`` are views of each segment's first clock,
    its own or its left leg's, which the rhythm diagnostics and the log read
    as the segment's clock. ``value_count`` is how many values describe a
    segment's own clocks in its observation.

    A clock per segment also compares the legs with its last turn: it
    remembers each segment's six leg angles at REMEMBERED_POINTS points of
    the turn, the centres of as many equal parts; the hand passes each point
    once per turn, so every turn replaces what the last one left. Clocks per
    leg also keep ``centres`` ``(W, N, 2, 3)``, the centres of each leg's
    step shape, which follow the centre actions slowly (``follow_centres``).
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
            self.centres = torch.zeros(
                (world_count, segment_count, 2, 3), dtype=torch.float32, device=device
            )
            # The worlds whose next step starts the centres at the actions.
            self.centres_restarted = torch.ones(
                world_count, dtype=torch.bool, device=device
            )
            # The share of the way to the actions the centres cover in a
            # step: a first-order lag with the time constant, sampled exactly.
            time_constant = settings.centre_time_constant_s
            self.centre_rate = (
                1 - math.exp(-step_duration_s / time_constant) if time_constant else 1.0
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
                (world_count, segment_count, REMEMBERED_POINTS, 6),
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
        and the centres of the legs' step shapes start at the next step's
        actions.
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
        self.clock_tempo_octaves.masked_fill_(restarted, 0.0)
        self.clock_tempo_hz.masked_fill_(restarted, self.settings.middle_tempo_hz)
        if self.per_leg:
            self.centres_restarted.logical_or_(world_mask)
        else:
            self.remembered.masked_fill_(restarted, False)

    def step(
        self,
        tempo_action: torch.Tensor,
        leg_angles_before: torch.Tensor | None = None,
        leg_angles: torch.Tensor | None = None,
        foot_contact: torch.Tensor | None = None,
    ) -> ClockStep:
        """Turn every hand for one step at the tempo chosen for its clock.

        ``tempo_action`` ``(W, N, C)`` is each clock's tempo action, from −1
        to 1. A clock per segment also compares the legs with its last turn,
        from ``leg_angles_before`` and ``leg_angles`` ``(W, N, 6)``, the leg
        angles before and after the step (see ``_compare_legs``). With load
        feedback σ, a leg's clock whose foot touches the ground, as
        ``foot_contact`` ``(W, N, 2)`` says, turns at ``ω + σ cos φ`` rad/s
        instead of its tempo's ``ω``: before mid-stance (φ = π/2) the hand
        hurries toward it, after it the hand is held back (Owaki and Ishiguro,
        2017, with stance on the first half of the turn). With σ ≥ ω it would
        stop where ``cos φ = −ω / σ``, for good, since the foot leaves the
        ground only in swing; the settings keep σ below the slowest tempo's ω.
        """
        self.clock_tempo_octaves.copy_(tempo_action * self.settings.tempo_range_octaves)
        self.clock_tempo_hz.copy_(
            self.settings.middle_tempo_hz * torch.exp2(self.clock_tempo_octaves)
        )
        speed = 2 * math.pi * self.clock_tempo_hz
        load_feedback = self.settings.load_feedback_rad_per_s
        if load_feedback:
            speed = speed + load_feedback * foot_contact * torch.cos(self.clock_phase)
        turned = speed * self.step_duration_s
        leg_squared_difference = leg_difference_known = None
        if not self.per_leg:
            leg_squared_difference, leg_difference_known = self._compare_legs(
                turned[..., 0], leg_angles_before, leg_angles
            )
        self.clock_phase.copy_((self.clock_phase + turned).remainder(2 * math.pi))
        return ClockStep(
            leg_squared_difference, leg_difference_known, self._tempo_mismatch()
        )

    def follow_centres(self, centre_actions: torch.Tensor) -> torch.Tensor:
        """Move the legs' step-shape centres toward the actions; return them.

        ``centre_actions`` and the returned ``centres`` are ``(W, N, 2, 3)``:
        each leg's sweep, lift and knee centres, in action units. Each step
        covers ``1 − e^(−Δt/τ)`` of the way, for the step's duration Δt and
        the time constant τ (``centre_time_constant_s``): the exact step of a
        first-order lag, for any τ, which covers 1 − e^(−1), 63%, of a sudden
        change in τ; with τ = 0 the centres are the actions. On the first
        step after a restart, the centres start at the actions.
        """
        following = self.centres + self.centre_rate * (centre_actions - self.centres)
        restarted = self.centres_restarted[:, None, None, None]
        self.centres.copy_(torch.where(restarted, centre_actions, following))
        self.centres_restarted.fill_(False)
        return self.centres

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
        angle_index = index[..., None].expand(*index.shape, 6)
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
        its tempo action, from −1 to 1.
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
