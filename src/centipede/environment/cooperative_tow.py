"""The tow: training wheels that fade (docs/cooperators/tow.md, cycle 10).

A standing body moves only when most of its segments step at once: a planted
foot holds about 0.8 of its segment's weight back. In training every segment
is pushed forward along the floor by its world's tow level times its own
weight, so that a few stepping segments move the body and a segment's own
step pays. A constant push must be braked by some feet in any steady walk;
from cycle 12 the push can shrink with each segment's own forward speed, to
none at a help speed, as Yu, Turk and Liu's propelling assistance does.
Each training world draws its level, as a share of the tow's top, at every
episode start, so that the agents train at smaller tows too. The worlds the
experiment reserves act with the agents' mean action: half of them keep the
top, half have no tow, which is the honest measure. The top falls while the
chosen half walks and rises while it does not.

The cooperators' environment owns one, applies its forces before each step
and adds its push back into the own ground force price; the experiment
starts it in training, reserves the measuring worlds and calls ``adjust``
after each update cycle.
"""

import math
from dataclasses import dataclass

import torch

from centipede.diagnostics_category import measure
from centipede.environment.cooperative_diagnostics import StepFacts
from centipede.environment.cooperative_reward import PART_NAMES
from centipede.environment.settings import CooperatorsSettings
from centipede.environment.simulation.constants import ACTION_DURATION_S

GROUPS = ("at the tow", "without tow")
LINE_PART = PART_NAMES.index("line")


@dataclass(frozen=True)
class TowFacts:
    """The tow after an update cycle, and what the mean action did over the
    measuring worlds' last window; single values or one per group, on the
    CPU, NaN before the first window."""

    tow: torch.Tensor = measure(
        "The tow's top: each segment's push in its own weights; training"
        " worlds draw theirs under it"
    )
    line_erased: torch.Tensor = measure(
        "Line each follower's tail erased", "mm/s", parts=GROUPS
    )
    closing: torch.Tensor = measure(
        "How fast the head's tip closed on its target", "mm/s", parts=GROUPS
    )
    lift_offs: torch.Tensor = measure(
        "Each foot's lift-offs after a stance of at least 60 ms", "1/s", parts=GROUPS
    )
    targets: torch.Tensor = measure("Targets reached per episode", parts=GROUPS)
    head_speed: torch.Tensor = measure(
        "The head's speed along the ground", "mm/s", parts=GROUPS
    )
    push: torch.Tensor = measure(
        "The push applied to each segment of the training worlds, in unit"
        " weights: under a tow that tracks a speed it falls as the body walks"
    )


class Tow:
    """Every world's tow level and the top it is drawn under.

    ``segment_weights_n`` are each segment's own weight; ``unit_weight_n``
    the unit the price and the observations count forces in.
    """

    def __init__(
        self,
        settings: CooperatorsSettings,
        world_count: int,
        segment_weights_n: tuple[float, ...],
        unit_weight_n: float,
        episode_steps: int,
        device: torch.device | str,
    ) -> None:
        """No tow until ``start``: the top is 0 and no world is reserved."""
        self.settings = settings
        self.episode_s = episode_steps * ACTION_DURATION_S
        self.top = 0.0
        self.started = False
        self.weights_n = torch.tensor(segment_weights_n, device=device)
        self.weights = self.weights_n / unit_weight_n
        self.unit_weight_n = unit_weight_n
        # Each segment's push of the last step, in unit weights.
        self.pushes = torch.zeros((world_count, len(segment_weights_n)), device=device)
        # Each world's level as a share of the top, and the worlds that draw
        # theirs at every episode start rather than keep it.
        self.shares = torch.zeros(world_count, device=device)
        self.drawing = torch.ones(world_count, dtype=torch.bool, device=device)
        # The measuring worlds of each group, and their measures' sums since
        # the top last moved: line, closing, lift-offs, targets, head speed.
        self.groups = torch.zeros((len(GROUPS), world_count), device=device)
        self.worlds_per_group = 0
        self._sums = torch.zeros((len(GROUPS), 5), device=device)
        # The training worlds' share in the mean push, and its sum.
        self._training_share = torch.zeros(world_count, device=device)
        self._push_sum = torch.zeros((), device=device)
        self._steps = 0
        self.generator = torch.Generator(device=device)
        nothing = torch.full((len(GROUPS),), math.nan)
        self.facts = TowFacts(
            tow=torch.tensor(0.0),
            line_erased=nothing.clone(),
            closing=nothing.clone(),
            lift_offs=nothing.clone(),
            targets=nothing.clone(),
            head_speed=nothing.clone(),
            push=torch.tensor(math.nan),
        )

    def start(self, top: float) -> None:
        """Tow from now on, under ``top``: the settings' start, or a saved one."""
        self.top = top
        self.started = True
        self.facts.tow.fill_(top)

    def reserve(self, worlds_per_group: int) -> None:
        """The last ``2 x worlds_per_group`` worlds measure the mean action:
        the first half always at the top, the second never towed."""
        world_count = self.shares.shape[0]
        first = world_count - len(GROUPS) * worlds_per_group
        for group_index, share in enumerate((1.0, 0.0)):
            start = first + group_index * worlds_per_group
            self.groups[group_index, start : start + worlds_per_group] = 1.0
            self.shares[start : start + worlds_per_group] = share
        self.drawing[first:] = False
        self.worlds_per_group = worlds_per_group
        segment_count = self.weights.shape[0]
        self._training_share.copy_(self.drawing / (self.drawing.sum() * segment_count))

    def restart(self, world_mask: torch.Tensor) -> None:
        """Draw a new share for the masked worlds that draw theirs, from
        ``tow_draw_floor`` to 1."""
        floor = self.settings.tow_draw_floor
        draws = floor + (1.0 - floor) * torch.rand(
            self.shares.shape, generator=self.generator, device=self.shares.device
        )
        self.shares.copy_(torch.where(world_mask & self.drawing, draws, self.shares))

    def levels(self) -> torch.Tensor:
        """Each world's tow level, ``(W,)``: its push in each segment's weights."""
        return self.shares * self.top

    def unit_weights(self) -> torch.Tensor:
        """Each segment's push of the last step, ``(W, N)``, in unit weights."""
        return self.pushes

    def forces(
        self, body_quaternion: torch.Tensor, forward_speed: torch.Tensor
    ) -> torch.Tensor:
        """Each segment's push, ``(W, N, 3)`` newtons in world axes, along its
        forward axis laid flat on the floor. With ``tow_help_speed_mm_per_s``
        the push shrinks with ``forward_speed`` ``(W, N)``, each segment's
        speed along its forward axis over the step before in m/s, to none at
        and above the help speed; it never holds a segment back."""
        help_speed = self.settings.tow_help_speed_mm_per_s * 1e-3
        share = (
            (1.0 - forward_speed / help_speed).clamp(0.0, 1.0) if help_speed else 1.0
        )
        self.pushes.copy_(self.levels()[:, None] * self.weights * share)
        w, x, y, z = body_quaternion.unbind(dim=-1)
        # The first column of each rotation matrix, without its height.
        flat = torch.stack(
            (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), torch.zeros_like(w)),
            dim=-1,
        )
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        size = self.pushes * self.unit_weight_n
        return size[..., None] * flat

    def record(self, step: StepFacts) -> None:
        """Add the step just taken to the measuring worlds' sums."""
        if not self.worlds_per_group:
            return
        measures = torch.stack(
            (
                step.reward_parts[:, 1:, LINE_PART].mean(dim=1),
                step.segment_progress[:, 0] * 1e3,
                step.lift_off.float().mean(dim=(1, 2)),
                step.target_reached.float(),
                step.segment_moved[:, 0] * 1e3,
            ),
            dim=-1,
        )
        self._sums += self.groups @ measures
        self._push_sum += self._training_share @ self.pushes.sum(dim=1)
        self._steps += 1

    def adjust(self) -> None:
        """After an update cycle: once ``tow_window_steps`` steps have passed
        since the top last moved, lower it by ``tow_step`` if the measuring
        worlds ``tow_fade_group`` names walked, followers erasing line and
        feet lifting off above the settings' bars, else raise it, within 0
        and ``tow_start``;
        then start a new window. The measures become ``facts``. This waits
        for the GPU once a window."""
        settings = self.settings
        if not self.worlds_per_group or self._steps < settings.tow_window_steps:
            return
        per_second = self._sums / (
            self.worlds_per_group * self._steps * ACTION_DURATION_S
        )
        line, closing, lift_offs, targets, head_speed = per_second.cpu().unbind(dim=-1)
        group = GROUPS.index(
            "at the tow" if settings.tow_fade_group == "towed" else "without tow"
        )
        walked = (
            line[group] > settings.tow_walk_mm_per_s
            and lift_offs[group] > settings.tow_lift_off_hz
        )
        if self.started:
            step = -settings.tow_step if walked else settings.tow_step
            self.top = min(settings.tow_start, max(0.0, self.top + step))
        self.facts.tow.fill_(self.top)
        self.facts.line_erased.copy_(line)
        self.facts.closing.copy_(closing)
        self.facts.lift_offs.copy_(lift_offs)
        self.facts.targets.copy_(targets * self.episode_s)
        self.facts.head_speed.copy_(head_speed)
        self.facts.push.copy_(self._push_sum / self._steps)
        self._sums.zero_()
        self._push_sum.zero_()
        self._steps = 0
