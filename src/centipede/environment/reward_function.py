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
    clocks.
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
    ) -> "RewardContext":
        """Build the context from the state after the step and saved copies.

        ``previous_body_planar_position`` is ``(W, N, 2)`` and
        ``previous_head_tip_position`` and ``target_position`` are ``(W, 2)``,
        all flat on the ground; ``arrived`` is a boolean ``(W,)``.
        ``previous_joint_position`` is ``(W, N, 7)``, the ``joint_angles``
        before the step; ``commanded_joints`` is ``(N, 7)``, 1.0 for the joints
        each segment commands and 0.0 for the others; ``joint_action`` is the
        step's ``(W, N, 6 or 7)`` motor commands: with leg clocks, the leg
        targets of the step shape.
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
        # Without spine control the action has no seventh column.
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
        )


def joint_angles(physical_state: PhysicalState) -> torch.Tensor:
    """Each segment's six leg angles and the spine joint behind it, ``(W, N, 7)``."""
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
    "movement": movement,
    "command": command,
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
    ) -> None:
        """Pair every term with its name and weight, in a fixed order.

        ``commanded_joints`` is the ``(N, 7)`` mask of ``RewardContext``.
        """
        self.settings = reward_settings
        self.arrival_radius_m = arrival_radius_m
        self.commanded_joints = commanded_joints
        self.weights = reward_settings.weights(max_episode_steps)
        self.reward_terms = [
            (name, weight, TERMS[name])
            for name, weight in self.weights.per_step.items()
        ]
        self.term_names = [name for name, _, _ in self.reward_terms]

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
