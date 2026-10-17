"""The cooperators' diagnostics: each step, each 20 s episode, and the gait.

Two categories, allocated once and refreshed in place, as in the main
environment's diagnostics. ``StepFacts`` says what the last step did in every
world: the reward's parts, the body's contacts and loads, its movement and
energy, the head's course, and the gait. ``EpisodeSummary`` says how each
episode went, the 20 s from one restart to the next: how many targets
the head reached, how far it walked, and what it spent. The cooperators'
environment fills them with two calls: ``record_step`` after each step's
rewards are known, and ``new_targets`` whenever targets appear, at an arrival
or a restart.

The gait is read from the feet, as the oscillators' gait analysis did. A foot
stands while the floor carries at least a tenth of a unit's weight on it, and
its step starts when it lifts off after a stance of at least 60 ms (a shorter
touch is a bounce). Its phase runs from 0 at one lift-off to a full turn at
the next, in proportion to time, using the time between its last two
lift-offs. At every lift-off of a foot, the phase of the foot it is compared
with is its lag: about 0 when the two lift together, half a turn when they
alternate, a small positive lag when the rear foot follows the front one, as
in a wave from head to tail, and negative when it leads. The window summary
gives each lag's mean, its consistency over every world and lift-off, and its
lock within each world. Nothing here waits for the GPU.
"""

import math
from dataclasses import dataclass

import torch
from torch.nn.functional import one_hot

from centipede.diagnostics_category import measure
from centipede.environment.cooperative_line import HeadLine
from centipede.environment.cooperative_reward import CHANNEL_NAMES, PART_NAMES
from centipede.environment.diagnostics import SHARE_EDGES
from centipede.environment.head import head_forward_direction, target_seen_from_head
from centipede.environment.reward_function import StepRewards
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.diagnostics import SimulationFacts
from centipede.environment.simulation.model_mapping import HeadOutline

LEGS = ("left", "right")
ENERGY_PARTS = ("pushing", "braking", "force")
# A foot stands while the floor carries at least this share of a unit's weight
# on it, so that a graze is not a stance.
STANCE_LOAD_SHARE = 0.1
# A stance shorter than this (60 ms) is a bounce, not a step.
MIN_STANCE_STEPS = 3
# The target's bearing from the head's forward direction when it appeared,
# either way, in four classes of 45 degrees.
BEARINGS = ("ahead", "side", "behind side", "behind")
BEARING_INNER_EDGES_RAD = (math.pi / 4, math.pi / 2, 3 * math.pi / 4)
# Targets reached in an episode; more count in the last bin.
TARGET_COUNT_EDGES = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)


@dataclass(frozen=True)
class StepFacts:
    """What happened in the last 20 ms step, in every world."""

    reward_parts: torch.Tensor = measure(
        "Each reward part, (W, N, 12), in the order of reward_part_names: the goal's"
        " in mm, health's in mJ",
        parts=PART_NAMES,
    )
    contact_flags: torch.Tensor = measure(
        "Left foot, right foot and body touching the ground, and legs touching,"
        " (W, N, 4)",
        summary="share",
        parts=("left foot", "right foot", "body", "legs touching"),
    )
    feet_carry: torch.Tensor = measure(
        "The floor's push on the segment's two feet, in unit weights, (W, N):"
        " 1 when they carry the segment's own weight"
    )
    belly_carry: torch.Tensor = measure(
        "The floor's push on the segment's body, in unit weights, (W, N)"
    )
    feet_push: torch.Tensor = measure(
        "The floor's push on the segment's feet along the segment, in unit"
        " weights, (W, N): positive drives it forward, negative brakes it"
    )
    segment_progress: torch.Tensor = measure(
        "Distance gained toward the segment's goal, (W, N)", "m"
    )
    segment_moved: torch.Tensor = measure(
        "Flat distance the segment's centre moved, (W, N)", "m"
    )
    own_energy: torch.Tensor = measure(
        "The segment's own energy, before sharing, priced as the cost charges"
        " it, (W, N, 3)",
        "mJ",
        parts=ENERGY_PARTS,
    )
    spine_bend: torch.Tensor = measure(
        "Angle of the spine joint behind the segment, either way; 0 for the"
        " rear, (W, N)",
        "rad",
    )
    off_line: torch.Tensor = measure(
        "Distance from the segment's tail to the head's line it has not erased, (W, N)",
        "m",
    )
    head_sideslip: torch.Tensor = measure(
        "The head's sideslip, either way: the angle between where it points"
        " and where its centre goes, each averaged over a step cycle, (W,)",
        "rad",
    )
    line_drift: torch.Tensor = measure(
        "How far to the side of the tip the head's line is drawn, either way"
        " (cycle 15; 0 without a drift), (W,)",
        "m",
    )
    joint_forces: torch.Tensor = measure(
        "The averaged force of the segment ahead on the segment and the"
        " averaged net force of its two joints, along the body, in unit"
        " weights, (W, N, 2): positive pulls it forward",
        parts=("front joint", "net"),
    )
    body_height: torch.Tensor = measure("Height of the segment's centre, (W, N)", "m")
    uprightness: torch.Tensor = measure(
        "1 upright, 0 on its side, -1 upside down, (W, N)"
    )
    head_distance: torch.Tensor = measure(
        "Flat distance from the head's tip to the target, (W,)", "m"
    )
    heading_error: torch.Tensor = measure(
        "Angle between the head's forward direction and the target, (W,)", "rad"
    )
    on_course: torch.Tensor = measure(
        "The target lies ahead of the head's tip, within the head's width of"
        " its forward line, (W,)",
        summary="share",
    )
    support: torch.Tensor = measure(
        "At least one left foot and one right foot standing, anywhere along the"
        " body, (W,)",
        summary="share",
    )
    target_reached: torch.Tensor = measure(
        "The head reached its target on the step, (W,)", summary="share"
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
    neighbour_lag_measured: torch.Tensor = measure(
        "The rear foot of each pair of neighbours' same-side feet lifted off"
        " while the front foot's phase was known, (W, N - 1, 2)",
        summary="share",
        parts=LEGS,
    )
    neighbour_lag: torch.Tensor = measure(
        "At each lift-off of a foot, the phase of the same-side foot of the"
        " segment ahead: positive when the rear follows, as in a wave from head"
        " to tail, (W, N - 1, 2)",
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
        " same segment: half a turn when they alternate, 0 together, (W, N)",
        "rad",
        summary="angle",
        counted_where="left_right_lag_measured",
    )
    target_position: torch.Tensor = measure(
        "Each world's target, world x and y, (W, 2); kept for recordings",
        "m",
        summary="recorded",
    )


@dataclass(frozen=True)
class EpisodeSummary:
    """How each episode went, written for the worlds whose episode just ended.

    Every episode lasts the time limit, 20 s, but a world's first one when
    first episodes are staggered. ``episode_ended`` marks those worlds; the
    other rows still hold earlier episodes and are left out of every summary.
    """

    episode_ended: torch.Tensor = measure(
        "Worlds whose episode ended on the last step, (W,)", summary="count"
    )
    targets_reached: torch.Tensor = measure(
        "Targets the head reached in the episode, (W,)",
        histogram_edges=TARGET_COUNT_EDGES,
    )
    targets_in_time: torch.Tensor = measure(
        "Targets the head reached within their stopwatch, (W,)"
    )
    head_progress: torch.Tensor = measure(
        "Distance the head's tip closed on its targets over the episode, (W,)", "m"
    )
    head_path_length: torch.Tensor = measure(
        "Total distance the head's tip travelled, (W,)", "m"
    )
    segment_return: torch.Tensor = measure(
        "Each segment's summed rewards by channel, (W, N, C), in the order of"
        " channel_names: the goal in mm, health in mJ",
        parts=CHANNEL_NAMES,
    )
    energy: torch.Tensor = measure(
        "Every segment's own energy over the episode, added up, (W, 3)",
        "mJ",
        parts=ENERGY_PARTS,
    )
    upside_down_share: torch.Tensor = measure(
        "Share of steps with the head upside down, (W,)", histogram_edges=SHARE_EDGES
    )
    targets_by_bearing: torch.Tensor = measure(
        "Targets that appeared in the episode, by their bearing from the head"
        " when they appeared, (W, 4)",
        parts=BEARINGS,
    )
    reached_by_bearing: torch.Tensor = measure(
        "Targets reached in the episode, by the bearing they appeared at, (W, 4):"
        " over targets_by_bearing, each class's share reached",
        parts=BEARINGS,
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


class CooperativeDiagnostics:
    """Fills the cooperators' two categories and keeps per-episode totals.

    ``step`` and ``episode`` are the categories the experiment reads, and
    ``simulation`` is the physics simulation's. ``reward_part_names`` labels
    the reward's parts and ``channel_names`` its channels; a recording scores
    each world by ``recording_score``, named ``score_name``.
    """

    reward_part_names = list(PART_NAMES)
    score_name = "targets reached"

    def __init__(
        self,
        world_count: int,
        segment_count: int,
        unit_weight_n: float,
        head_outline: HeadOutline,
        device: torch.device | str,
        simulation_facts: SimulationFacts,
        channel_names: tuple[str, ...] = CHANNEL_NAMES,
    ) -> None:
        """Allocate both categories and the running totals, all zero."""
        self.simulation = simulation_facts
        self.channel_names = list(channel_names)
        self.unit_weight_n = unit_weight_n
        self.head_outline = head_outline
        self.feet = FootCycles(world_count, segment_count, device)

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
            segment_progress=zeros(segment_count),
            segment_moved=zeros(segment_count),
            own_energy=zeros(segment_count, len(ENERGY_PARTS)),
            spine_bend=zeros(segment_count),
            off_line=zeros(segment_count),
            head_sideslip=zeros(),
            line_drift=zeros(),
            joint_forces=zeros(segment_count, 2),
            body_height=zeros(segment_count),
            uprightness=zeros(segment_count),
            head_distance=zeros(),
            heading_error=zeros(),
            on_course=flags(),
            support=flags(),
            target_reached=flags(),
            foot_standing=flags(segment_count, 2),
            lift_off=flags(segment_count, 2),
            neighbour_lag_measured=flags(segment_count - 1, 2),
            neighbour_lag=zeros(segment_count - 1, 2),
            left_right_lag_measured=flags(segment_count),
            left_right_lag=zeros(segment_count),
            target_position=zeros(2),
        )
        self.episode = EpisodeSummary(
            episode_ended=flags(),
            targets_reached=zeros(),
            targets_in_time=zeros(),
            head_progress=zeros(),
            head_path_length=zeros(),
            segment_return=zeros(segment_count, len(channel_names)),
            energy=zeros(len(ENERGY_PARTS)),
            upside_down_share=zeros(),
            targets_by_bearing=zeros(len(BEARINGS)),
            reached_by_bearing=zeros(len(BEARINGS)),
        )
        # The current episode's totals in every world, named as the summary
        # values they become, and each target's bearing class.
        self._totals = {
            name: zeros(*getattr(self.episode, name).shape[1:])
            for name in (
                "targets_reached",
                "targets_in_time",
                "head_progress",
                "head_path_length",
                "segment_return",
                "energy",
                "upside_down_share",
                "targets_by_bearing",
                "reached_by_bearing",
            )
        }
        self._steps = zeros()
        self._bearing_class = zeros(dtype=torch.long)
        self._bearing_edges = torch.tensor(BEARING_INNER_EDGES_RAD, device=device)

    def recording_score(self) -> torch.Tensor:
        """Each world's score for the step just taken, ``(W,)``."""
        return self.step.target_reached.float()

    def new_targets(
        self,
        placed: torch.Tensor,
        restarted: torch.Tensor,
        physical_state: PhysicalState,
        target_position: torch.Tensor,
    ) -> None:
        """Note the targets just placed in the ``placed`` worlds, after clearing
        the totals and foot cycles of the ``restarted`` ones."""
        for total in (*self._totals.values(), self._steps):
            total.masked_fill_(_per_world(restarted, total), 0)
        self.feet.restart(restarted)
        offset = target_seen_from_head(physical_state, target_position)
        bearing = torch.atan2(offset[:, 1].abs(), offset[:, 0])
        bearing_class = torch.bucketize(bearing, self._bearing_edges)
        self._bearing_class.copy_(
            torch.where(placed, bearing_class, self._bearing_class)
        )
        self._totals["targets_by_bearing"] += (
            one_hot(bearing_class, len(BEARINGS)) * placed[:, None]
        )
        self.step.target_position.copy_(
            torch.where(placed[:, None], target_position, self.step.target_position)
        )

    def record_step(
        self,
        physical_state: PhysicalState,
        previous_body_planar_position: torch.Tensor,
        previous_head_tip_position: torch.Tensor,
        target_position: torch.Tensor,
        rewards: torch.Tensor,
        step_rewards: StepRewards,
        own_energy: torch.Tensor,
        off_line: torch.Tensor,
        averaged_forces: torch.Tensor,
        line: HeadLine,
        arrived: torch.Tensor,
        in_time: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Refresh the step facts, add to the totals, and publish ended episodes.

        Called once per step, after the rewards are known and before any target
        is replaced or world restarted; the arguments are values the
        environment already has: ``rewards`` by channel, ``(W, N, C)``, the
        reward's parts, each segment's own energy, its tail's distance from
        the head's line it has not erased, the averaged forces
        (``AveragedForces.values``), the head's line (its sideslip and
        drift), and ``in_time`` marking the arrivals within their stopwatch.
        """
        state = physical_state
        step = self.step
        unit = self.unit_weight_n
        quaternion = state.body_quaternion
        head_tip = state.head_tip_position[:, :2]
        to_target = target_position - head_tip
        forward = head_forward_direction(quaternion[:, 0])
        seen = target_seen_from_head(state, target_position)

        segment_progress = step_rewards.segment_progress
        step.reward_parts.copy_(step_rewards.reward_parts)
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
        step.segment_progress.copy_(segment_progress)
        step.segment_moved.copy_(
            (state.body_planar_position - previous_body_planar_position).norm(dim=-1)
        )
        step.own_energy.copy_(own_energy)
        step.spine_bend.copy_(state.spine_yaw_position.abs())
        step.off_line.copy_(off_line)
        step.head_sideslip.copy_(line.sideslip.abs())
        step.line_drift.copy_(line.drift.abs())
        step.joint_forces.copy_(averaged_forces[..., [2, 0]])
        step.body_height.copy_(state.body_height)
        # The z component of each segment's up axis: 1 - 2(x² + y²).
        step.uprightness.copy_(
            1 - 2 * (quaternion[..., 1] ** 2 + quaternion[..., 2] ** 2)
        )
        step.head_distance.copy_(to_target.norm(dim=-1))
        step.heading_error.copy_(
            torch.atan2(
                forward[:, 0] * to_target[:, 1] - forward[:, 1] * to_target[:, 0],
                (forward * to_target).sum(dim=-1),
            ).abs()
        )
        step.on_course.copy_(
            (seen[:, 0] > 0) & (seen[:, 1].abs() <= self.head_outline.half_width_m)
        )
        step.target_reached.copy_(arrived)

        standing = state.foot_normal_force >= STANCE_LOAD_SHARE * unit
        lift_off, phase, known = self.feet.step(standing)
        step.foot_standing.copy_(standing)
        step.support.copy_(standing.any(dim=1).all(dim=-1))
        step.lift_off.copy_(lift_off)
        step.neighbour_lag_measured.copy_(lift_off[:, 1:] & known[:, :-1])
        step.neighbour_lag.copy_(phase[:, :-1])
        step.left_right_lag_measured.copy_(lift_off[..., 1] & known[..., 0])
        step.left_right_lag.copy_(phase[..., 0])

        self._steps += 1
        totals = self._totals
        totals["targets_reached"] += arrived
        totals["targets_in_time"] += in_time
        totals["head_progress"] += segment_progress[:, 0]
        totals["head_path_length"] += (head_tip - previous_head_tip_position).norm(
            dim=-1
        )
        totals["segment_return"] += rewards
        totals["energy"] += own_energy.sum(dim=1)
        totals["upside_down_share"] += step.uprightness[:, 0] < 0
        totals["reached_by_bearing"] += (
            one_hot(self._bearing_class, len(BEARINGS)) * arrived[:, None]
        )

        self.episode.episode_ended.copy_(truncated)
        for name, total in totals.items():
            if name == "upside_down_share":
                total = total / self._steps
            summary = getattr(self.episode, name)
            summary.copy_(torch.where(_per_world(truncated, summary), total, summary))


def _per_world(world_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """A ``(W,)`` mask shaped to broadcast over a tensor whose rows are worlds."""
    return world_mask.reshape(world_mask.shape + (1,) * (like.dim() - 1))
