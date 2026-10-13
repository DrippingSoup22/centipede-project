import math
from dataclasses import dataclass

import torch

from centipede.environment.clocks import ClockStep
from centipede.environment.settings import RewardSettings
from centipede.environment.simulation import PhysicalState


@dataclass(frozen=True)
class RewardContext:
    """What every reward term may read about the step just taken.

    Terms that need the body itself, such as joint angles or heights, read
    ``physical_state``. Values that several terms share and that must be
    computed first, such as distances, are computed once here.

    ``arrived`` is 1.0 for the worlds whose head reached the target on this step
    and 0.0 elsewhere, ``(W,)``. ``distance_before`` and ``distance_after`` are
    each segment's flat distance to its goal before and after the step,
    ``(W, N)``: the head's tip to the target for the head, and for every other
    segment its centre to where the segment ahead's centre was before the step.
    ``arrival_radius_m`` is the distance that counts as arrival.
    ``joint_movement`` is ``(W, N)``: the mean, over the joints each segment
    commands, of how far each moved on the step, squared, in rad².
    ``command_size`` is ``(W, N)``: the mean, over the same joints, of the
    segment's command for the step, squared, from 0 (none) to 1 (full).
    ``foot_slip_speed`` is ``(W, N, 2)``: how fast each foot slid along the
    ground during the step, m/s, counted only for a foot that touched the
    ground at both ends of it. ``clock`` is the clocks' step, or None without
    clocks. ``leg_push_unit_n`` is the leg-leg force that costs a whole leg
    pushing cost: one segment's weight.

    With the feet's physics (cycle 2), ``foot_load`` ``(W, N, 2)`` is each
    foot's normal force from the floor over the segment's own weight,
    averaged over the step, and ``stride_support`` ``(W, N)`` both feet's
    together averaged over the segment's last stride; None without it.
    ``foot_lift_load`` ``(W, N, 2)`` is, for each foot that left the ground
    on the step, the load it bore on the step before (0 for the others).
    ``health_scale`` ``(N,)`` multiplies each segment's health costs: the
    curriculum's ramp, times the followers' scale.
    """

    physical_state: PhysicalState
    arrived: torch.Tensor
    distance_before: torch.Tensor
    distance_after: torch.Tensor
    arrival_radius_m: float
    joint_movement: torch.Tensor
    command_size: torch.Tensor
    foot_slip_speed: torch.Tensor
    clock: ClockStep | None
    leg_push_unit_n: float
    foot_load: torch.Tensor | None = None
    stride_support: torch.Tensor | None = None
    health_scale: torch.Tensor | float = 1.0
    foot_lift_load: torch.Tensor | None = None

    @classmethod
    def from_step(
        cls,
        physical_state: PhysicalState,
        previous_body_planar_position: torch.Tensor,
        previous_head_tip_position: torch.Tensor,
        target_position: torch.Tensor,
        arrived: torch.Tensor,
        arrival_radius_m: float,
        previous_joint_position: torch.Tensor,
        commanded_joints: torch.Tensor,
        joint_action: torch.Tensor,
        foot_slip_speed: torch.Tensor,
        clock: ClockStep | None,
        leg_push_unit_n: float,
        foot_load: torch.Tensor | None = None,
        stride_support: torch.Tensor | None = None,
        health_scale: torch.Tensor | float = 1.0,
        foot_lift_load: torch.Tensor | None = None,
    ) -> "RewardContext":
        """Build the context from the state after the step and saved copies.

        ``previous_body_planar_position`` is ``(W, N, 2)`` and
        ``previous_head_tip_position`` and ``target_position`` are ``(W, 2)``,
        all flat on the ground; ``arrived`` is a boolean ``(W,)``.
        ``previous_joint_position`` is ``(W, N, M + 1)``, the ``joint_angles``
        before the step, with ``M`` leg motors per segment; ``commanded_joints``
        is ``(N, M + 1)``, 1.0 for the joints each segment commands and 0.0 for
        the others; ``joint_action`` is the step's ``(W, N, M or M + 1)`` motor
        commands: with leg clocks, the leg targets of the step shape.
        New tensors are built, so neither the physical state nor the saved
        copies change.
        """
        # Where each segment was and is: centres, except the head's tip.
        before = previous_body_planar_position.clone()
        before[:, 0] = previous_head_tip_position
        after = physical_state.body_planar_position.clone()
        after[:, 0] = physical_state.head_tip_position[:, :2]

        # Each segment's goal: the target for the head, and for the others
        # where the segment ahead was before the step.
        goal = torch.empty_like(before)
        goal[:, 0] = target_position
        goal[:, 1:] = previous_body_planar_position[:, :-1]

        squared_movement = (
            joint_angles(physical_state) - previous_joint_position
        ).square() * commanded_joints
        # Without spine control the action has no spine column.
        squared_command = (
            joint_action.square() * commanded_joints[:, : joint_action.shape[-1]]
        )
        joint_count = commanded_joints.sum(dim=-1)
        return cls(
            physical_state=physical_state,
            arrived=arrived.float(),
            distance_before=(before - goal).norm(dim=-1),
            distance_after=(after - goal).norm(dim=-1),
            arrival_radius_m=arrival_radius_m,
            joint_movement=squared_movement.sum(dim=-1) / joint_count,
            command_size=squared_command.sum(dim=-1) / joint_count,
            foot_slip_speed=foot_slip_speed,
            clock=clock,
            leg_push_unit_n=leg_push_unit_n,
            foot_load=foot_load,
            stride_support=stride_support,
            health_scale=health_scale,
            foot_lift_load=foot_lift_load,
        )


def joint_angles(physical_state: PhysicalState) -> torch.Tensor:
    """Each segment's leg angles and the spine joint behind it, ``(W, N, M + 1)``."""
    return torch.cat(
        (
            physical_state.leg_joint_position,
            physical_state.spine_yaw_position.unsqueeze(-1),
        ),
        dim=-1,
    )


# -- Reward terms --------------------------------------------------------------
# Each term takes the context and the reward settings and returns one value per
# world and segment, (W, N), already signed: positive is good, negative is a
# cost. The reward function multiplies each by its weight and adds them up.


def arrival(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """1 for the head of a world whose head arrived on this step, else 0; each
    follower receives ``follower_arrival_share`` of it (1: the same)."""
    rewards = context.arrived[:, None].repeat(1, context.distance_after.shape[1])
    rewards[:, 1:] *= settings.follower_arrival_share
    return rewards


def progress(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """How many times the head halved its distance to the target, for every
    segment: positive when it came closer, negative when it moved away.

    log2(distance before ÷ distance after) of the head, which every follower
    receives too, at ``follower_progress_share`` (1: the same). Distances
    closer than the arrival radius count as the radius, so landing nearer the
    target's centre earns nothing more. Over an episode the values add up to
    log2(start distance ÷ end distance), whatever the path: log2(start distance
    ÷ arrival radius) for an episode that arrives, and walking past the target
    gives back what was earned.

    An earlier form of the reward paid each follower instead for halving its
    own distance to the spot where the segment ahead had been, at
    ``follower_progress_ratio``. Those spots move on every step, so the
    followers' values never add up to anything bounded: they pay for speed.
    The term is kept so that runs configured with it read back with their
    reward.
    """
    radius = context.arrival_radius_m
    halvings = torch.log2(
        context.distance_before.clamp(min=radius)
        / context.distance_after.clamp(min=radius)
    )
    head = halvings[:, :1]
    rewards = settings.follower_progress_share * head.expand_as(halvings)
    if settings.follower_progress_ratio is not None:
        rewards = rewards + settings.follower_progress_ratio * halvings
    rewards[:, 0] = head[:, 0]
    return rewards


def step_cost(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−1 for every segment on every step: the price of time."""
    return -torch.ones_like(context.distance_after)


def efficiency(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−(ε + distance after) / (ε + distance before): −1 when not moving.

    The earlier efficiency reward held the step cost and the progress in this
    one term, under one weight; it is kept for the runs configured with it.
    """
    epsilon = settings.distance_ratio_epsilon_m
    return -(epsilon + context.distance_after) / (epsilon + context.distance_before)


def body_contact(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−1 for each segment whose body touches the ground, else 0."""
    return -context.physical_state.body_ground_contact.float()


def leg_contact(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−1 for each segment with a leg touching another leg, else 0."""
    return -context.physical_state.leg_leg_contact.float()


def leg_pushing(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−how hard the segment's legs press on other legs, from 0 to 1.

    The summed normal force of the contacts between one of its legs and
    another leg, in units of one segment's weight, at most 1. Both segments of
    a contact pay. Legs that brush past each other cost almost nothing; legs
    that push against each other cost. A health cost: times the context's
    ``health_scale``.
    """
    force = context.physical_state.leg_leg_force
    return -(force / context.leg_push_unit_n).clamp(max=1.0) * context.health_scale


def carry(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−the share of its own weight a segment's feet did not carry, 0 to 1.

    max(0, 1 − the feet's floor force over the segment's own weight,
    averaged over its last stride): a segment carried by its neighbours
    pays, one that carries more than its weight pays nothing (the user's
    choice, 2026-10-12: that is its neighbours' problem). A health cost:
    times the context's ``health_scale``.
    """
    shortfall = (1 - context.stride_support).clamp(min=0.0)
    return -shortfall * context.health_scale


def grip(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−how fast the segment's loaded feet slid on the floor, 0 to 1.

    Each foot that touched the ground at both ends of the step counts its
    sliding speed weighted by its load, its floor force over the segment's
    own weight; the two feet's sum in units of ``grip_unit_m_per_s`` (about
    a default stroke's foot speed), at most 1. A foot that slides while
    bearing no weight costs nothing; a loaded foot dragged along costs. A
    health cost: times the context's ``health_scale``.
    """
    loaded_sliding = (context.foot_load * context.foot_slip_speed).sum(dim=-1)
    unit = settings.grip_unit_m_per_s
    return -(loaded_sliding / unit).clamp(max=1.0) * context.health_scale


def foot_hold(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−how much the segment's loaded feet moved, 0 to 1: grip, and lifting.

    A foot that bears weight must not move (cycle 3, the user's load rule):
    grip's loaded sliding, plus, for each foot that left the ground on the
    step, the load it bore on the step before, in units of the segment's own
    weight. A foot lifts free once its weight has passed to other feet, its
    sibling's or its neighbours', so the cheapest walk lifts each leg while
    the legs around it stand: the load feedback of Owaki and Ishiguro (2017)
    judged on the outcome, never acting on the clocks; in arthropods, load
    signals keep a leg in stance until it is unloaded (Duysens, Clarac and
    Cruse, Physiol Rev 2000). At most 1. A health cost.
    """
    loaded_sliding = (context.foot_load * context.foot_slip_speed).sum(dim=-1)
    moved = loaded_sliding / settings.grip_unit_m_per_s
    moved = moved + context.foot_lift_load.sum(dim=-1)
    return -moved.clamp(max=1.0) * context.health_scale


def tempo_ahead(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−how far the segment's tempo is from the segment ahead's, 0 to 1.

    The difference in octaves, a whole octave (one tempo twice the other) or
    more costing 1, in proportion below (the absolute form, which works
    where the squared one did not; lesson 3 of the oscillators study). The
    head, with nothing ahead, pays nothing: each follower keeps the tempo of
    the one ahead, and so, down the chain, the head's.
    """
    return -context.clock.tempo_gap_ahead.clamp(max=1.0)


# The tempo gap at which each of the tempo chain's costs reaches its whole
# weight: four octaves, beyond any gap two segments reach (the heart's
# 12.5 Hz ceiling over a 1 Hz neighbour is 3.6), so that neither cost ever
# stops growing and their sum always has one cheapest tempo. Two octaves,
# the first scale, let the costs go flat under run 42's racing head (2.3).
TEMPO_CHAIN_OCTAVES = 4.0


def tempo_chain_ahead(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−the segment's tempo gap to the segment ahead over four octaves, 0 to 1.

    The tempo chain (the user's rule, 2026-10-13): each segment copies the
    tempo of the segment ahead, the cost growing in proportion to the gap so
    that small gaps still cost (lesson 3 of the oscillators study), and
    minds the segment behind (``tempo_chain_behind``). The head, with nothing
    ahead, pays nothing here.
    """
    gap = context.clock.tempo_gap_ahead / TEMPO_CHAIN_OCTAVES
    return -gap.clamp(max=1.0)


def tempo_chain_behind(
    context: RewardContext, settings: RewardSettings
) -> torch.Tensor:
    """−(the segment's tempo gap to the segment behind over four octaves)², 0 to 1.

    Squared, so that a small gap costs almost nothing and a large one a lot.
    With weights w_ahead and w_behind, a follower copies the segment ahead
    until it is 2 w_ahead / w_behind octaves from the segment behind, then
    waits for it (where the two costs' slopes meet); the head may lead the
    body but not run away from it. The rear segment, with nothing behind,
    pays nothing here.
    """
    gap = context.clock.tempo_gap_behind / TEMPO_CHAIN_OCTAVES
    return -gap.clamp(max=1.0).square()


def energy(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−the positive power of the segment's leg motors, 0 to 1.

    The work the motors put into the legs per second: each motor's force
    times its speed wherever it pushes the way its joint moves, summed over
    the segment's leg motors and averaged over the step's physics steps, in
    units of ``energy_reference_w``, at most 1. A motor holding a pose or
    pushed back spends nothing. Fu et al. (CoRL 2021) charged their robots
    the motors' power: without it, their legs tapped at high frequency.
    """
    power = context.physical_state.leg_motor_power
    return -(power / settings.energy_reference_w).clamp(max=1.0)


def foot_slip(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−how fast the segment's feet slid along the ground, from 0 to 1.

    Each foot that touched the ground at both ends of the step counts its
    sliding speed in units of ``foot_slip_unit_m_per_s``, at most 1; a foot in
    the air counts 0. The mean over the two feet. A foot standing still on a
    still body costs nothing; a foot dragged along by the body costs.
    """
    unit = settings.foot_slip_unit_m_per_s
    return -(context.foot_slip_speed / unit).clamp(max=1.0).mean(dim=-1)


def legs_off_tempo(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−how far the legs are from where they were at this point of the last turn.

    Each leg angle is compared with its angle at the same point of the
    segment's clock's last turn: the mean, over the six leg angles and the
    points the hand passed in the step, of the difference squared in units of
    ``legs_off_tempo_unit_deg``, at most 1, and 0 in the first turn after the
    world restarted. It asks the legs to repeat their movement with the
    clock, whatever the movement is: a leg standing still repeats too.
    """
    unit = math.radians(settings.legs_off_tempo_unit_deg)
    clock = context.clock
    error = (clock.leg_squared_difference / unit**2).clamp(max=1.0)
    return -error * clock.leg_difference_known


def out_of_tempo(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−how far the segment's tempo is from its neighbours', from 0 to 1.

    The mean, over the neighbours the segment sees, of the tempo difference
    in octaves divided by the largest possible; with leg clocks, each leg's
    against its sibling's and the same side's legs of those neighbours, and
    the mean of the two legs. Both segments of a pair pay; the head pays
    ``head_tempo_share`` of its own, so that the others follow its tempo
    more than it follows theirs. With the clocks' ``coupling = "ahead"``, a
    segment is compared only with the neighbours ahead of it, so only the
    rear segment of a pair pays and the share is None: the head pays nothing
    for its neighbours, only, with leg clocks, its own legs' mismatch.
    """
    mismatch = context.clock.tempo_mismatch.clone()
    if settings.head_tempo_share is not None:
        mismatch[:, 0] *= settings.head_tempo_share
    return -mismatch


def no_support(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−1 for each segment with neither foot on the ground after the step.

    A segment carried by its neighbours pays; one standing on its feet, even
    still, pays nothing.
    """
    state = context.physical_state
    carried = ~state.left_foot_ground_contact & ~state.right_foot_ground_contact
    return -carried.float()


def both_feet_down(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−1 for each segment with both feet on the ground after the step.

    The counterpart of ``no_support``: one foot down is free, both down cost
    a little, none down more. Both feet down is part of slow walking, so its
    weight is kept small.
    """
    state = context.physical_state
    standing = state.left_foot_ground_contact & state.right_foot_ground_contact
    return -standing.float()


def clock_tracking(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−how far the segment's legs' clocks are from the segment ahead's, 0 to 1.

    With leg clocks, each leg's clock in use, its tempo and step shape, is
    compared with the same side's leg of the segment ahead: the mean, over
    the two legs and the six values, of |difference| ÷ 2, so that a mismatch
    costs in proportion to its size. The phase is not
    compared, so the offset between neighbours stays free. The head, with
    nothing ahead, pays nothing: a follower's job is to track the clock
    ahead, so that the head's clock passes down the body. With
    ``clock_tracking_centres`` off, only the tempo and the two amplitudes
    are compared, so that the posture stays free.
    """
    if settings.clock_tracking_centres:
        return -context.clock.tracking_mismatch
    return -context.clock.step_size_tracking_mismatch


def head_synchrony(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """How well the whole body keeps the head's clock, 0 to 1, paid alike to
    every follower; the head receives 0.

    With leg clocks, each follower's legs' tempos and sweep and lift
    amplitudes in use are compared with the head's same-side leg: a
    follower scores 1 − the mean, over its two legs and the three values, of
    |difference| ÷ 2, and every follower receives the mean of the
    followers' scores. A shared reward: the followers cannot see the head,
    so they earn it only by passing its clock down faithfully. The centres
    and the phase are left out, so posture and the offset between
    neighbours stay free.
    """
    rewards = context.clock.head_synchrony[:, None].repeat(
        1, context.distance_after.shape[1]
    )
    rewards[:, 0] = 0.0
    return rewards


def clock_change(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−how fast the head's legs' clocks change, 0 to 1; the followers pay 0.

    The mean, over the head's two legs and the six values of their clocks in
    use, of the change on the step squared, in units of the largest change
    the clocks' lag allows in one step, each at most 1: a head that swings
    its clock from one extreme to the other pays 1, one that holds it pays 0.
    The head may change its clock, but not too fast.
    """
    head_change = torch.zeros_like(context.clock.change_size)
    head_change[:, 0] = context.clock.change_size[:, 0]
    return -head_change


def movement(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−(how far the segment's joints moved on the step)², in random-command units.

    The joints' mean squared movement divided by that of random commands
    (``random_command_movement_deg``, squared), at most 1: −1 for moving as
    much as random commands or more, almost 0 for a calm movement, 0 for none.
    Squaring makes a sudden movement cost more than the same distance in small
    steps: ten steps of 1° cost a tenth of one step of 10°.
    """
    unit = math.radians(settings.random_command_movement_deg) ** 2
    return -(context.joint_movement / unit).clamp(max=1.0)


def command(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−(the segment's commands)²: −1 for full commands, 0 for none.

    The mean, over the joints the segment commands, of its command squared,
    like the control cost of Gymnasium's Ant (half the sum of its squared
    commands). Unlike the movement cost, it charges what the segment asks for,
    exploration noise included, rather than how far its joints move.
    """
    return -context.command_size


@dataclass(frozen=True)
class StepRewards:
    """What the reward function returns for one step; created anew every step.

    ``rewards`` is ``(W, N)``. ``reward_parts`` is ``(W, N, T)``, one weighted
    term per position along the last dimension, in the order of the reward
    function's term names; the parts add up to ``rewards``.
    ``segment_progress`` is ``(W, N)``: metres each segment came closer to its
    goal on this step; ``joint_movement`` is the context's, in rad².
    ``foot_slip`` and ``legs_off_tempo`` are ``(W, N)``: the two walking costs
    before their weights, from 0 to 1, whether or not they are paid; None in
    the earlier efficiency reward, and ``legs_off_tempo`` also without a clock
    per segment, the only kind that compares the legs.
    """

    rewards: torch.Tensor
    reward_parts: torch.Tensor
    segment_progress: torch.Tensor
    joint_movement: torch.Tensor
    foot_slip: torch.Tensor | None = None
    legs_off_tempo: torch.Tensor | None = None


# The terms of each reward, in order: the current one, built from its rules,
# and the earlier efficiency reward, for the runs configured with it.
TERMS = {
    "arrival": arrival,
    "progress": progress,
    "step_cost": step_cost,
    "efficiency": efficiency,
    "body_contact": body_contact,
    "leg_contact": leg_contact,
    "foot_slip": foot_slip,
    "legs_off_tempo": legs_off_tempo,
    "out_of_tempo": out_of_tempo,
    "no_support": no_support,
    "both_feet_down": both_feet_down,
    "clock_tracking": clock_tracking,
    "head_synchrony": head_synchrony,
    "clock_change": clock_change,
    "movement": movement,
    "leg_pushing": leg_pushing,
    "command": command,
    "carry": carry,
    "grip": grip,
    "foot_hold": foot_hold,
    "tempo_ahead": tempo_ahead,
    "tempo_chain_ahead": tempo_chain_ahead,
    "tempo_chain_behind": tempo_chain_behind,
    "energy": energy,
}


class RewardFunction:
    """Each segment's reward: the weighted sum of the reward terms.

    The weights follow from [environment.rewards] and the episode length (see
    ``RewardSettings.weights``); a weight of zero switches its term off. To add
    a term, write its function above, add it to ``TERMS``, and give it a weight
    in ``RewardSettings.weights``.
    """

    def __init__(
        self,
        reward_settings: RewardSettings,
        max_episode_steps: int,
        arrival_radius_m: float,
        commanded_joints: torch.Tensor,
        leg_push_unit_n: float,
    ) -> None:
        """Pair every term with its name and weight, in a fixed order.

        ``commanded_joints`` is the ``(N, M + 1)`` mask of ``RewardContext``,
        and ``leg_push_unit_n`` its unit of leg pushing.
        """
        self.settings = reward_settings
        self.arrival_radius_m = arrival_radius_m
        self.leg_push_unit_n = leg_push_unit_n
        self.commanded_joints = commanded_joints
        self.weights = reward_settings.weights(max_episode_steps)
        # Each segment's health costs' multiplier: the ramp's start until the
        # curriculum sets a level, times the followers' scale.
        self.health_scale = torch.ones(
            commanded_joints.shape[0], device=commanded_joints.device
        )
        self.set_level(0.0)
        self.reward_terms = [
            (name, weight, TERMS[name])
            for name, weight in self.weights.per_step.items()
        ]
        self.term_names = [name for name, _, _ in self.reward_terms]

    def set_level(self, level: float) -> None:
        """Harden the health costs to the curriculum's ``level``, 0 to 1:
        ``health_ramp_start`` at 0, growing in proportion to 1 at 1."""
        start = self.settings.health_ramp_start
        self.health_scale.fill_(start + (1 - start) * level)
        self.health_scale[1:] *= self.settings.follower_health_scale

    def compute(
        self,
        physical_state: PhysicalState,
        previous_body_planar_position: torch.Tensor,
        previous_head_tip_position: torch.Tensor,
        target_position: torch.Tensor,
        arrived: torch.Tensor,
        previous_joint_position: torch.Tensor,
        joint_action: torch.Tensor,
        foot_slip_speed: torch.Tensor,
        clock: ClockStep | None = None,
        foot_load: torch.Tensor | None = None,
        stride_support: torch.Tensor | None = None,
        foot_lift_load: torch.Tensor | None = None,
    ) -> StepRewards:
        """Every segment's reward for the step just taken, with its parts.

        The arguments are those of ``RewardContext.from_step``.
        """
        context = RewardContext.from_step(
            physical_state,
            previous_body_planar_position,
            previous_head_tip_position,
            target_position,
            arrived,
            self.arrival_radius_m,
            previous_joint_position,
            self.commanded_joints,
            joint_action,
            foot_slip_speed,
            clock,
            self.leg_push_unit_n,
            foot_load,
            stride_support,
            self.health_scale,
            foot_lift_load,
        )
        reward_parts = torch.stack(
            [
                weight * term(context, self.settings)
                for _, weight, term in self.reward_terms
            ],
            dim=-1,
        )
        rules = not self.settings.uses_per_step_weights
        compares_legs = clock is not None and clock.leg_difference_known is not None
        return StepRewards(
            rewards=reward_parts.sum(dim=-1),
            reward_parts=reward_parts,
            segment_progress=context.distance_before - context.distance_after,
            joint_movement=context.joint_movement,
            foot_slip=-foot_slip(context, self.settings) if rules else None,
            legs_off_tempo=(
                -legs_off_tempo(context, self.settings)
                if rules and compares_legs
                else None
            ),
        )
