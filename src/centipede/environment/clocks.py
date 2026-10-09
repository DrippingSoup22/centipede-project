"""Each segment's clock: a hand that turns at a tempo the segment chooses.

The clocks are part of the environment's state, like the targets: a segment
sets its clock's tempo with one action and observes where its hand is, and
only its own. See docs/environment.md, "Clocks".
"""

import math
from dataclasses import dataclass

import torch

from centipede.environment.settings import ClockSettings

# How many points of a turn each clock remembers the legs' angles at.
REMEMBERED_POINTS = 64
# A segment observes its hand as (cos, sin) and its tempo action.
CLOCK_VALUE_COUNT = 3


@dataclass(frozen=True)
class ClockStep:
    """What the clock costs read after a step; created anew every step.

    ``leg_difference`` is ``(W, N, 6)``: each leg angle now minus its angle
    when the clock's hand last passed the same point of its turn, in rad;
    ``leg_difference_known`` ``(W, N)`` is False where the hand had not passed
    that point since the world restarted. ``tempo_mismatch`` ``(W, N)`` is the
    mean, over the neighbours the segment sees, of the tempo difference in
    octaves divided by the largest possible, from 0 to 1.
    """

    leg_difference: torch.Tensor
    leg_difference_known: torch.Tensor
    tempo_mismatch: torch.Tensor


class Clocks:
    """One clock per world and segment.

    ``phase`` ``(W, N)`` is each hand, in rad from 0 to 2π; ``tempo_hz`` the
    turns per second it ran at on the last step, and ``tempo_octaves`` the
    same in octaves from the middle tempo: the tempo action times the range.
    All three are overwritten in place by ``reset`` and ``step``. To compare
    the legs with the clock's last turn, each segment's six leg angles are
    remembered at REMEMBERED_POINTS points of the turn, as the hand passes.
    """

    def __init__(
        self,
        settings: ClockSettings,
        world_count: int,
        segment_count: int,
        observation_radius: int,
        step_duration_s: float,
        device: torch.device | str,
    ) -> None:
        """Allocate the clocks and count each segment's visible neighbours."""
        self.settings = settings
        self.step_duration_s = step_duration_s
        shape = (world_count, segment_count)
        self.phase = torch.zeros(shape, dtype=torch.float32, device=device)
        self.tempo_octaves = torch.zeros(shape, dtype=torch.float32, device=device)
        self.tempo_hz = torch.full(
            shape, settings.middle_tempo_hz, dtype=torch.float32, device=device
        )
        self.remembered_angles = torch.zeros(
            (*shape, REMEMBERED_POINTS, 6), dtype=torch.float32, device=device
        )
        self.remembered = torch.zeros(
            (*shape, REMEMBERED_POINTS), dtype=torch.bool, device=device
        )
        # Starting hands have their own random sequence, so they never change
        # the starting poses or the targets.
        self.generator = torch.Generator(device=device)

        # The neighbours whose tempo a segment is compared with: those it sees.
        self.neighbour_offsets = range(
            1, min(observation_radius, segment_count - 1) + 1
        )
        counts = torch.zeros(segment_count, dtype=torch.float32, device=device)
        for offset in self.neighbour_offsets:
            counts[:-offset] += 1
            counts[offset:] += 1
        self.neighbour_counts = counts.clamp(min=1)

    def reset(self, world_mask: torch.Tensor, seed: int | None = None) -> None:
        """Start the masked worlds' clocks at random hands and the middle tempo.

        A ``seed`` restarts the hands' random sequence. The legs' remembered
        angles are forgotten, so the first turn after a restart costs nothing.
        """
        if seed is not None:
            self.generator.manual_seed(seed)
        new_phase = (
            2
            * math.pi
            * torch.rand(
                self.phase.shape, generator=self.generator, device=self.phase.device
            )
        )
        restarted = world_mask[:, None]
        self.phase.copy_(torch.where(restarted, new_phase, self.phase))
        self.tempo_octaves.masked_fill_(restarted, 0.0)
        self.tempo_hz.masked_fill_(restarted, self.settings.middle_tempo_hz)
        self.remembered.masked_fill_(restarted[..., None], False)

    def step(self, tempo_action: torch.Tensor, leg_angles: torch.Tensor) -> ClockStep:
        """Turn every hand for one step and compare the legs with the last turn.

        ``tempo_action`` ``(W, N)`` is each segment's tempo action, from −1 to
        1; ``leg_angles`` ``(W, N, 6)`` the leg angles after the step, which
        are remembered where the hand now is, for the next turn.
        """
        self.tempo_octaves.copy_(tempo_action * self.settings.tempo_range_octaves)
        self.tempo_hz.copy_(
            self.settings.middle_tempo_hz * torch.exp2(self.tempo_octaves)
        )
        turned = 2 * math.pi * self.tempo_hz * self.step_duration_s
        self.phase.add_(turned).remainder_(2 * math.pi)

        point = (self.phase * (REMEMBERED_POINTS / (2 * math.pi))).long()
        point.clamp_(max=REMEMBERED_POINTS - 1)
        angle_index = point[..., None, None].expand(*point.shape, 1, 6)
        remembered_angles = self.remembered_angles.gather(2, angle_index).squeeze(2)
        known = self.remembered.gather(2, point[..., None]).squeeze(2)
        self.remembered_angles.scatter_(2, angle_index, leg_angles[:, :, None, :])
        self.remembered.scatter_(2, point[..., None], torch.ones_like(known[..., None]))

        # A tempo difference of two octaves, the largest, counts 1.
        largest = 2 * self.settings.tempo_range_octaves
        mismatch = torch.zeros_like(self.tempo_octaves)
        for offset in self.neighbour_offsets:
            difference = (
                self.tempo_octaves[:, :-offset] - self.tempo_octaves[:, offset:]
            ).abs() / largest
            mismatch[:, :-offset] += difference
            mismatch[:, offset:] += difference
        return ClockStep(
            leg_difference=leg_angles - remembered_angles,
            leg_difference_known=known,
            tempo_mismatch=mismatch / self.neighbour_counts,
        )

    def observation_values(self) -> torch.Tensor:
        """Each segment's own clock as it sees it, ``(W, N, 3)``.

        Its hand as ``cos`` and ``sin``, so that the end of a turn and the
        start of the next look alike, and its tempo action, from −1 to 1.
        """
        return torch.stack(
            (
                torch.cos(self.phase),
                torch.sin(self.phase),
                self.tempo_octaves / self.settings.tempo_range_octaves,
            ),
            dim=-1,
        )
