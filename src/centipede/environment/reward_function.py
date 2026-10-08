from dataclasses import dataclass

import torch

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
    """

    physical_state: PhysicalState
    arrived: torch.Tensor
    distance_before: torch.Tensor
    distance_after: torch.Tensor
    arrival_radius_m: float

    @classmethod
    def from_step(
        cls,
        physical_state: PhysicalState,
        previous_body_planar_position: torch.Tensor,
        previous_head_tip_position: torch.Tensor,
        target_position: torch.Tensor,
        arrived: torch.Tensor,
        arrival_radius_m: float,
    ) -> "RewardContext":
        """Build the context from the state after the step and saved copies.

        ``previous_body_planar_position`` is ``(W, N, 2)`` and
        ``previous_head_tip_position`` and ``target_position`` are ``(W, 2)``,
        all flat on the ground; ``arrived`` is a boolean ``(W,)``. New tensors
        are built, so neither the physical state nor the saved copies change.
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

        return cls(
            physical_state=physical_state,
            arrived=arrived.float(),
            distance_before=(before - goal).norm(dim=-1),
            distance_after=(after - goal).norm(dim=-1),
            arrival_radius_m=arrival_radius_m,
        )


# -- Reward terms --------------------------------------------------------------
# Each term takes the context and the reward settings and returns one value per
# world and segment, (W, N), already signed: positive is good, negative is a
# cost. The reward function multiplies each by its weight and adds them up.


def arrival(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """1 for every segment of a world whose head arrived on this step, else 0."""
    return context.arrived[:, None].expand_as(context.distance_after)


def progress(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """How many times each segment halved its distance to its goal: positive
    when it came closer, negative when it moved away, 0 when it did not move.

    log2(distance before ÷ distance after), counted in full for the head and at
    ``follower_progress_ratio`` for the followers, whose goals are much nearer.
    Distances closer than the arrival radius count as the radius, so landing
    nearer the target's centre earns nothing more. Over an episode the head's
    values add up to log2(start distance ÷ end distance), whatever its path:
    log2(start distance ÷ arrival radius) for an episode that arrives.
    """
    radius = context.arrival_radius_m
    halvings = torch.log2(
        context.distance_before.clamp(min=radius)
        / context.distance_after.clamp(min=radius)
    )
    share = torch.full_like(halvings[0], settings.follower_progress_ratio)
    share[0] = 1.0
    return halvings * share


def step_cost(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−1 for every segment on every step: the price of time."""
    return -torch.ones_like(context.distance_after)


def efficiency(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−(ε + distance after) / (ε + distance before): −1 when not moving.

    The reward used before 2026-10-08 held the step cost and the progress in
    this one term, under one weight; it is kept for the runs saved then.
    """
    epsilon = settings.distance_ratio_epsilon_m
    return -(epsilon + context.distance_after) / (epsilon + context.distance_before)


def body_contact(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−1 for each segment whose body touches the ground, else 0."""
    return -context.physical_state.body_ground_contact.float()


def leg_contact(context: RewardContext, settings: RewardSettings) -> torch.Tensor:
    """−1 for each segment with a leg touching another leg, else 0."""
    return -context.physical_state.leg_leg_contact.float()


@dataclass(frozen=True)
class StepRewards:
    """What the reward function returns for one step; created anew every step.

    ``rewards`` is ``(W, N)``. ``reward_parts`` is ``(W, N, T)``, one weighted
    term per position along the last dimension, in the order of the reward
    function's term names; the parts add up to ``rewards``.
    ``segment_progress`` is ``(W, N)``: metres each segment came closer to its
    goal on this step.
    """

    rewards: torch.Tensor
    reward_parts: torch.Tensor
    segment_progress: torch.Tensor


# The terms of each reward, in order: the current one, built from its rules,
# and the one used before 2026-10-08, for the runs saved then.
TERMS = {
    "arrival": arrival,
    "progress": progress,
    "step_cost": step_cost,
    "efficiency": efficiency,
    "body_contact": body_contact,
    "leg_contact": leg_contact,
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
    ) -> None:
        """Pair every term with its name and weight, in a fixed order."""
        self.settings = reward_settings
        self.arrival_radius_m = arrival_radius_m
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
        )
        reward_parts = torch.stack(
            [
                weight * term(context, self.settings)
                for _, weight, term in self.reward_terms
            ],
            dim=-1,
        )
        return StepRewards(
            rewards=reward_parts.sum(dim=-1),
            reward_parts=reward_parts,
            segment_progress=context.distance_before - context.distance_after,
        )
