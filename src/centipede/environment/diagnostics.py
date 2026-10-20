"""The environment's diagnostics: each step, each episode, and the gait.

Two categories, allocated once and refreshed in place. ``StepFacts`` says
what the last step did in every world: the reward's parts, the body's
contacts, loads and energy, its place on the line, and the gait.
``EpisodeSummary`` says how each episode went, from one restart to the next.
The front file fills them with two calls: ``record_step`` after each step's
rewards are known, and ``restart`` for the worlds that restart. Nothing here
waits for the GPU.

The gait is read from the feet. A foot stands while the floor carries at
least a tenth of a unit's weight on it, and its step starts when it lifts off
after a stance of at least 60 ms (a shorter touch is a bounce). Its phase
runs from 0 at one lift-off to a full turn at the next, in proportion to
time, using the time between its last two lift-offs. At every lift-off of a
foot, the phase of the foot it is compared with is its lag: about 0 when the
two lift together, half a turn when they alternate, a small positive lag
when the rear foot follows the front one, as in a wave from front to rear,
and negative when it leads. The window summary gives each lag's mean, its
consistency over every world and lift-off, and its lock within each world.

A leg's strokes are read from its lift joint instead, contacts aside, so that
a gait too fast for a 60 ms stance still counts: a stroke ends when the
joint, having risen 3 degrees above its average over the last second, falls
3 degrees below it.
"""

import math
from dataclasses import dataclass

import torch

from centipede.diagnostics_category import measure
from centipede.environment.line import StraightLine
from centipede.environment.reward_function import (
    CHANNEL_NAMES,
    ENERGY_PARTS,
    PART_NAMES,
)
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.constants import ACTION_DURATION_S
from centipede.environment.simulation.diagnostics import SimulationFacts

LEGS = ("left", "right")
# A foot stands while the floor carries at least this share of a unit's weight
# on it, so that a graze is not a stance.
STANCE_LOAD_SHARE = 0.1
# A stance shorter than this (60 ms) is a bounce, not a step.
MIN_STANCE_STEPS = 3
# A lift joint's stroke: its swing either way of its average over a second.
STROKE_BAND_RAD = math.radians(3.0)
STROKE_AVERAGING_S = 1.0
# Shares of an episode's steps, in 10% bins.
SHARE_EDGES = tuple(round(0.1 * index, 1) for index in range(11))


@dataclass(frozen=True)
class StepFacts:
    """What happened in the last 20 ms step, in every world."""

    reward_parts: torch.Tensor = measure(
        "Each reward part, (W, N, 9), in the order of reward_part_names: the line,"
        " squeeze, motion and force share in mm, the rest in mJ",
        parts=PART_NAMES,
    )
    contact_flags: torch.Tensor = measure(
        "Left foot, right foot and body touching the ground, and legs touching,"
        " (W, N, 4)",
        summary="share",
        parts=("left foot", "right foot", "body", "legs touching"),
    )
    feet_carry: torch.Tensor = measure(
        "The floor's push on the unit's two feet, in unit weights, (W, N):"
        " 1 when they carry the unit's own weight"
    )
    belly_carry: torch.Tensor = measure(
        "The floor's push on the unit's body, in unit weights, (W, N)"
    )
    feet_push: torch.Tensor = measure(
        "The floor's force on the unit's feet along the unit's own axis, as the"
        " unit feels it, in unit weights, (W, N): a unit pitched nose down reads"
        " part of its support as braking, so floor_push judges driving"
    )
    floor_push: torch.Tensor = measure(
        "The floor's push on everything of the unit that touches it (its feet,"
        " its legs' other parts, its body), along the unit laid flat on the"
        " floor, in unit weights, (W, N): positive drives it forward, negative"
        " brakes it"
    )
    segment_progress: torch.Tensor = measure(
        "How far the unit's centre went along the line, (W, N)", "m"
    )
    own_energy: torch.Tensor = measure(
        "The unit's own energy, priced as the cost charges it, (W, N, 3)",
        "mJ",
        parts=ENERGY_PARTS,
    )
    spine_bend: torch.Tensor = measure(
        "Angle of the spine joint behind the unit, either way; 0 for the rear, (W, N)",
        "rad",
    )
    off_line: torch.Tensor = measure(
        "Distance from the unit's centre to the line, (W, N)", "m"
    )
    line_angle: torch.Tensor = measure(
        "Angle between the unit's body and the line's direction, either way, (W, N)",
        "rad",
    )
    joint_forces: torch.Tensor = measure(
        "The averaged force of the unit ahead on the unit and the averaged net"
        " force of its two joints, along the body, in unit weights, (W, N, 2):"
        " positive pulls it forward",
        parts=("front joint", "net"),
    )
    body_height: torch.Tensor = measure("Height of the unit's centre, (W, N)", "m")
    uprightness: torch.Tensor = measure(
        "1 upright, 0 on its side, -1 upside down, (W, N)"
    )
    support: torch.Tensor = measure(
        "At least one left foot and one right foot standing, anywhere along the"
        " body, (W,)",
        summary="share",
    )
    foot_standing: torch.Tensor = measure(
        "The foot stands: the floor carries at least a tenth of a unit's weight"
        " on it, (W, N, 2); its share is the duty factor",
        summary="share",
        parts=LEGS,
    )
    lift_off: torch.Tensor = measure(
        "The foot lifted off after a stance of at least 60 ms: a step began,"
        " (W, N, 2); its share over 20 ms is the foot's step tempo",
        summary="share",
        parts=LEGS,
    )
    leg_stroke: torch.Tensor = measure(
        "The leg's lift joint ended a stroke: having risen 3 degrees above its"
        " average over the last second, it fell 3 degrees below it, (W, N, 2);"
        " its share over 20 ms is the leg's stroke tempo, contacts aside",
        summary="share",
        parts=LEGS,
    )
    neighbour_lag_measured: torch.Tensor = measure(
        "The rear foot of each pair of neighbours' same-side feet lifted off"
        " while the front foot's phase was known, (W, N - 1, 2)",
        summary="share",
        parts=LEGS,
    )
    neighbour_lag: torch.Tensor = measure(
        "At each lift-off of a foot, the phase of the same-side foot of the"
        " unit ahead: positive when the rear follows, as in a wave from front"
        " to rear, (W, N - 1, 2)",
        "rad",
        summary="angle",
        parts=LEGS,
        counted_where="neighbour_lag_measured",
    )
    left_right_lag_measured: torch.Tensor = measure(
        "The right foot lifted off while the left foot's phase was known, (W, N)",
        summary="share",
    )
    left_right_lag: torch.Tensor = measure(
        "At each lift-off of a right foot, the phase of the left foot of the"
        " same unit: half a turn when they alternate, 0 together, (W, N)",
        "rad",
        summary="angle",
        counted_where="left_right_lag_measured",
    )
    line_ends: torch.Tensor = measure(
        "The line from a body length behind the units' middle to a body length"
        " ahead, world x and y of both ends, (W, 2, 2); kept for recordings",
        "m",
        summary="recorded",
    )


@dataclass(frozen=True)
class EpisodeSummary:
    """How each episode went, written for the worlds whose episode just ended.

    ``episode_ended`` marks those worlds; the other rows still hold earlier
    episodes and are left out of every summary.
    """

    episode_ended: torch.Tensor = measure(
        "Worlds whose episode ended on the last step, (W,)", summary="count"
    )
    strayed: torch.Tensor = measure(
        "The episode ended because the units lost the line, (W,)", summary="share"
    )
    duration: torch.Tensor = measure("The episode's length, (W,)", "s")
    line_progress: torch.Tensor = measure(
        "How far the units' centres went along the line over the episode, their"
        " mean, (W,)",
        "m",
    )
    segment_return: torch.Tensor = measure(
        "Each unit's summed rewards by account, (W, N, 2), in the order of"
        " channel_names, in mm",
        parts=CHANNEL_NAMES,
    )
    energy: torch.Tensor = measure(
        "Every unit's own energy over the episode, added up, (W, 3)",
        "mJ",
        parts=ENERGY_PARTS,
    )
    upside_down_share: torch.Tensor = measure(
        "Share of steps with the front unit upside down, (W,)",
        histogram_edges=SHARE_EDGES,
    )


class FootCycles:
    """Each foot's stance and step cycle in every world, ``(W, N, 2)``."""

    def __init__(self, world_count: int, segment_count: int, device) -> None:
        def zeros(dtype=torch.int32) -> torch.Tensor:
            return torch.zeros(
                (world_count, segment_count, 2), dtype=dtype, device=device
            )

        self.stance_steps = zeros()
        self.steps_since_lift_off = zeros()
        # Steps between the foot's last two lift-offs; 0 while not known.
        self.period_steps = zeros()
        self.lifted_before = zeros(torch.bool)

    def restart(self, world_mask: torch.Tensor) -> None:
        """Forget the masked worlds' cycles, as their episodes restart."""
        for value in vars(self).values():
            value.masked_fill_(world_mask[:, None, None], 0)

    def step(
        self, standing: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance by a step with each foot ``standing`` or not.

        Returns the feet that lifted off, every foot's phase in rad, and where
        the phase is known: once a whole cycle was measured, and while the
        foot keeps stepping, within twice its last cycle.
        """
        lift_off = ~standing & (self.stance_steps >= MIN_STANCE_STEPS)
        self.stance_steps.copy_(torch.where(standing, self.stance_steps + 1, 0))
        self.steps_since_lift_off += 1
        self.period_steps.copy_(
            torch.where(
                lift_off & self.lifted_before,
                self.steps_since_lift_off,
                self.period_steps,
            )
        )
        self.steps_since_lift_off.masked_fill_(lift_off, 0)
        self.lifted_before |= lift_off
        since = self.steps_since_lift_off.float()
        period = self.period_steps.float()
        phase = 2 * math.pi * since / period.clamp(min=1)
        known = (self.period_steps > 0) & (since < 2 * period)
        return lift_off, phase, known


class LegStrokes:
    """Each leg's lift joint against its average over the last second, in
    every world, ``(W, N, 2)``."""

    def __init__(self, world_count: int, segment_count: int, device) -> None:
        shape = (world_count, segment_count, 2)
        self.average = torch.zeros(shape, device=device)
        self.raised = torch.zeros(shape, dtype=torch.bool, device=device)
        # Whether the average has started from the joint's angle yet.
        self.started = torch.zeros(shape, dtype=torch.bool, device=device)

    def restart(self, world_mask: torch.Tensor) -> None:
        """Forget the masked worlds' strokes, as their episodes restart."""
        for value in vars(self).values():
            value.masked_fill_(world_mask[:, None, None], 0)

    def step(self, lift_angle: torch.Tensor) -> torch.Tensor:
        """Advance by a step with each leg's lift joint at ``lift_angle``
        (rad); returns the legs that ended a stroke."""
        rate = ACTION_DURATION_S / STROKE_AVERAGING_S
        moved = self.average + rate * (lift_angle - self.average)
        self.average.copy_(torch.where(self.started, moved, lift_angle))
        self.started.fill_(True)
        above = lift_angle > self.average + STROKE_BAND_RAD
        below = lift_angle < self.average - STROKE_BAND_RAD
        stroke = self.raised & below
        self.raised.copy_((self.raised | above) & ~below)
        return stroke


class EnvironmentDiagnostics:
    """Fills the two categories and keeps per-episode totals.

    ``step`` and ``episode`` are the categories the experiment reads, and
    ``simulation`` is the physics simulation's. ``reward_part_names`` labels
    the reward's parts and ``channel_names`` its accounts; a recording scores
    each world by ``recording_score``, named ``score_name``, and draws the line
    ``body_length_m`` either way of the body's middle.
    """

    reward_part_names = list(PART_NAMES)
    channel_names = list(CHANNEL_NAMES)
    score_name = "line paid"

    def __init__(
        self,
        world_count: int,
        segment_count: int,
        unit_weight_n: float,
        device: torch.device | str,
        simulation_facts: SimulationFacts,
        body_length_m: float = 0.0,
    ) -> None:
        """Allocate both categories and the running totals, all zero."""
        self.simulation = simulation_facts
        self.unit_weight_n = unit_weight_n
        self.line_ends_m = torch.tensor([-body_length_m, body_length_m], device=device)
        self.feet = FootCycles(world_count, segment_count, device)
        self.legs = LegStrokes(world_count, segment_count, device)

        def zeros(*trailing_shape: int, dtype=torch.float32) -> torch.Tensor:
            return torch.zeros(
                (world_count, *trailing_shape), dtype=dtype, device=device
            )

        def flags(*trailing_shape: int) -> torch.Tensor:
            return zeros(*trailing_shape, dtype=torch.bool)

        self.step = StepFacts(
            reward_parts=zeros(segment_count, len(PART_NAMES)),
            contact_flags=flags(segment_count, 4),
            feet_carry=zeros(segment_count),
            belly_carry=zeros(segment_count),
            feet_push=zeros(segment_count),
            floor_push=zeros(segment_count),
            segment_progress=zeros(segment_count),
            own_energy=zeros(segment_count, len(ENERGY_PARTS)),
            spine_bend=zeros(segment_count),
            off_line=zeros(segment_count),
            line_angle=zeros(segment_count),
            joint_forces=zeros(segment_count, 2),
            body_height=zeros(segment_count),
            uprightness=zeros(segment_count),
            support=flags(),
            foot_standing=flags(segment_count, 2),
            lift_off=flags(segment_count, 2),
            leg_stroke=flags(segment_count, 2),
            neighbour_lag_measured=flags(segment_count - 1, 2),
            neighbour_lag=zeros(segment_count - 1, 2),
            left_right_lag_measured=flags(segment_count),
            left_right_lag=zeros(segment_count),
            line_ends=zeros(2, 2),
        )
        self.episode = EpisodeSummary(
            episode_ended=flags(),
            strayed=flags(),
            duration=zeros(),
            line_progress=zeros(),
            segment_return=zeros(segment_count, len(CHANNEL_NAMES)),
            energy=zeros(len(ENERGY_PARTS)),
            upside_down_share=zeros(),
        )
        # The current episode's totals in every world, named as the summary
        # values they become.
        self._totals = {
            name: zeros(*getattr(self.episode, name).shape[1:])
            for name in (
                "duration",
                "line_progress",
                "segment_return",
                "energy",
                "upside_down_share",
            )
        }
        self._steps = zeros()

    def recording_score(self) -> torch.Tensor:
        """Each world's score for the step just taken, ``(W,)``: the line paid,
        the units' mean."""
        return self.step.reward_parts[..., PART_NAMES.index("line")].mean(dim=1)

    def restart(self, world_mask: torch.Tensor) -> None:
        """Clear the masked worlds' totals, foot cycles and leg strokes, as
        they restart."""
        for total in (*self._totals.values(), self._steps):
            total.masked_fill_(_per_world(world_mask, total), 0)
        self.feet.restart(world_mask)
        self.legs.restart(world_mask)

    def record_step(
        self,
        physical_state: PhysicalState,
        rewards: torch.Tensor,
        reward_parts: torch.Tensor,
        own_energy: torch.Tensor,
        progress: torch.Tensor,
        line: StraightLine,
        averaged_forces: torch.Tensor,
        strayed: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Refresh the step facts, add to the totals, and publish ended episodes.

        Called once per step, after the rewards are known and before any
        world restarts; the arguments are values the environment already has:
        ``rewards`` by account, ``(W, N, 2)``, the reward's parts, each unit's
        own energy, its step along the line (m), the line with each unit's
        place on it, the averaged forces (``AveragedForces.values``), the
        worlds that lost the line and the worlds whose episode ends.
        """
        state = physical_state
        step = self.step
        unit = self.unit_weight_n
        quaternion = state.body_quaternion

        step.reward_parts.copy_(reward_parts)
        step.contact_flags.copy_(
            torch.stack(
                (
                    state.left_foot_ground_contact,
                    state.right_foot_ground_contact,
                    state.body_ground_contact,
                    state.leg_leg_contact,
                ),
                dim=-1,
            )
        )
        step.feet_carry.copy_(state.foot_normal_force.sum(dim=-1) / unit)
        step.belly_carry.copy_(state.belly_normal_force / unit)
        step.feet_push.copy_(state.foot_forward_force.sum(dim=-1) / unit)
        step.floor_push.copy_(state.floor_forward_force / unit)
        step.segment_progress.copy_(progress)
        step.own_energy.copy_(own_energy)
        step.spine_bend.copy_(state.spine_yaw_position.abs())
        step.off_line.copy_(line.distance)
        step.line_angle.copy_(line.angle.abs())
        step.joint_forces.copy_(averaged_forces[..., [2, 0]])
        step.body_height.copy_(state.body_height)
        # The z component of each unit's up axis: 1 - 2(x² + y²).
        step.uprightness.copy_(
            1 - 2 * (quaternion[..., 1] ** 2 + quaternion[..., 2] ** 2)
        )

        standing = state.foot_normal_force >= STANCE_LOAD_SHARE * unit
        lift_off, phase, known = self.feet.step(standing)
        step.foot_standing.copy_(standing)
        step.support.copy_(standing.any(dim=1).all(dim=-1))
        step.lift_off.copy_(lift_off)
        # Each leg's joints run from the body outward, the left leg's first;
        # the lift is a leg's second.
        motors = state.leg_joint_position.shape[-1]
        lift_angle = state.leg_joint_position[..., [1, motors // 2 + 1]]
        step.leg_stroke.copy_(self.legs.step(lift_angle))
        step.neighbour_lag_measured.copy_(lift_off[:, 1:] & known[:, :-1])
        step.neighbour_lag.copy_(phase[:, :-1])
        step.left_right_lag_measured.copy_(lift_off[..., 1] & known[..., 0])
        step.left_right_lag.copy_(phase[..., 0])
        middle = line.origin + line.along.mean(dim=1, keepdim=True) * line.direction
        step.line_ends.copy_(
            middle[:, None] + self.line_ends_m[:, None] * line.direction[:, None]
        )

        self._steps += 1
        totals = self._totals
        totals["duration"] += ACTION_DURATION_S
        totals["line_progress"] += progress.mean(dim=1)
        totals["segment_return"] += rewards
        totals["energy"] += own_energy.sum(dim=1)
        totals["upside_down_share"] += step.uprightness[:, 0] < 0

        self.episode.episode_ended.copy_(truncated)
        self.episode.strayed.copy_(
            torch.where(truncated, strayed, self.episode.strayed)
        )
        for name, total in totals.items():
            if name == "upside_down_share":
                total = total / self._steps
            summary = getattr(self.episode, name)
            summary.copy_(torch.where(_per_world(truncated, summary), total, summary))


def _per_world(world_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """A ``(W,)`` mask shaped to broadcast over a tensor whose rows are worlds."""
    return world_mask.reshape(world_mask.shape + (1,) * (like.dim() - 1))
