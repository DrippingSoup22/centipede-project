"""The environment's diagnostics: step facts and episode summaries.

Both categories are allocated once and refreshed in place. The front file fills
them with two calls: ``record_step`` after each step's rewards and episode ends
are known, and ``start_episodes`` whenever worlds start a new episode. Nothing
here waits for the GPU. The values are listed in docs/diagnostics.md.
"""

from dataclasses import dataclass

import torch

from centipede.diagnostics_category import measure
from centipede.environment.observation_builder import head_forward_direction
from centipede.environment.reward_function import StepRewards
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.diagnostics import SimulationFacts

# Histogram bins of four episode values, fixed so that every run, CPU or GPU,
# counts in the same bins. Episode lengths, in steps, grow in steps of about
# 1.5x and include every power of two; a time limit on an edge, such as 1,024
# or 8,192 steps, starts its own bin, so time-outs are not mixed with arrivals.
# Longer episodes count in the last bin.
LENGTH_EDGES_STEPS = (0,) + tuple(
    sorted(
        {2**power for power in range(4, 15)} | {3 * 2**power for power in range(3, 13)}
    )
)
# Distance closed from -100% (twice as far as at the start) to 100%, in 10% bins.
DISTANCE_CLOSED_EDGES = tuple(round(-1 + 0.1 * index, 1) for index in range(21))
# Shares of an episode's steps, in 10% bins.
SHARE_EDGES = tuple(round(0.1 * index, 1) for index in range(11))
# The head's distance to its target at the end of an episode, in metres: the
# first bin, below 1 mm, holds the arrivals when the arrival radius is 1 mm;
# the others tell how far the episodes that ran out of time ended, finely near
# the targets (placed 10 to 20 mm away) and coarsely far from them. Farther
# endings count in the last bin.
FINAL_DISTANCE_EDGES_M = (0.0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2)
# How each episode ended, one bin each, measured against its own start
# distance so that the classes hold for any target distance: arrived; ran out
# of time within a quarter of the start distance, within half, closer than at
# the start, or not closer; left the range circle.
ENDINGS = (
    "arrived",
    "within a quarter",
    "within half",
    "closer",
    "not closer",
    "left the circle",
)
ENDING_EDGES = tuple(range(len(ENDINGS) + 1))


@dataclass(frozen=True)
class StepFacts:
    """What happened in the last 20 ms step, in every world."""

    reward_parts: torch.Tensor = measure(
        "Each reward term's weighted contribution, (W, N, T), in the order of "
        "reward_part_names; the parts add up to the reward"
    )
    contact_flags: torch.Tensor = measure(
        "Left foot, right foot and body on the ground, and legs touching, (W, N, 4)",
        summary="share",
        parts=("left foot", "right foot", "body", "legs touching"),
    )
    segment_progress: torch.Tensor = measure(
        "Distance gained toward the segment's goal, (W, N)", "m"
    )
    segment_moved: torch.Tensor = measure(
        "Flat distance the segment's centre moved, (W, N)", "m"
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
    target_position: torch.Tensor = measure(
        "Each world's target, world x and y, (W, 2); kept for recordings",
        "m",
        summary="recorded",
    )
    range_radius: torch.Tensor = measure(
        "Radius of each world's range circle around its target, infinite without"
        " one, (W,); kept for recordings",
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
    arrived: torch.Tensor = measure(
        "The head reached the target rather than running out of time, (W,)",
        summary="share",
    )
    left_range: torch.Tensor = measure(
        "The head's tip left the range circle, which cut the episode, (W,)",
        summary="share",
    )
    ending: torch.Tensor = measure(
        "How the episode ended, as the bin of ENDINGS, (W,): arrived; ran out of"
        " time within a quarter of the start distance, within half, closer, or not"
        " closer; left the range circle",
        histogram_edges=ENDING_EDGES,
    )
    length_steps: torch.Tensor = measure(
        "Episode length, (W,)", "steps of 20 ms", histogram_edges=LENGTH_EDGES_STEPS
    )
    segment_return: torch.Tensor = measure(
        "Sum of each segment's rewards over the episode, (W, N)"
    )
    start_distance: torch.Tensor = measure(
        "Head's distance to the target at the start, (W,)", "m"
    )
    final_distance: torch.Tensor = measure(
        "Head's distance to the target at the end, (W,)",
        "m",
        histogram_edges=FINAL_DISTANCE_EDGES_M,
    )
    distance_closed: torch.Tensor = measure(
        "Share of the start distance closed by the end: 1 - final / start, (W,)",
        histogram_edges=DISTANCE_CLOSED_EDGES,
    )
    head_path_length: torch.Tensor = measure(
        "Total distance the head's tip travelled, (W,)", "m"
    )
    segment_total_progress: torch.Tensor = measure(
        "Sum of each segment's progress over the episode, (W, N)", "m"
    )
    body_contact_share: torch.Tensor = measure(
        "Share of steps with the body on the ground, (W, N)"
    )
    leg_contact_share: torch.Tensor = measure(
        "Share of steps with legs touching, (W, N)"
    )
    foot_contact_share: torch.Tensor = measure(
        "Share of steps each foot (left, right) was on the ground, (W, N, 2)",
        parts=("left foot", "right foot"),
    )
    upside_down_share: torch.Tensor = measure(
        "Share of steps with the head upside down, (W,)", histogram_edges=SHARE_EDGES
    )


class EnvironmentDiagnostics:
    """Fills the environment's two categories, and keeps per-episode totals.

    ``step`` and ``episode`` are the categories the experiment reads, and
    ``simulation`` is the physics simulation's, offered here because only the
    environment can see the simulation. ``reward_part_names`` labels the last
    dimension of ``reward_parts``.
    """

    def __init__(
        self,
        world_count: int,
        segment_count: int,
        reward_part_names: list[str],
        device: torch.device | str,
        simulation_facts: SimulationFacts,
    ) -> None:
        """Allocate both categories and the running totals, all zero."""
        self.reward_part_names = list(reward_part_names)
        self.simulation = simulation_facts

        def zeros(*trailing_shape: int, dtype=torch.float32) -> torch.Tensor:
            return torch.zeros(
                (world_count, *trailing_shape), dtype=dtype, device=device
            )

        self.step = StepFacts(
            reward_parts=zeros(segment_count, len(reward_part_names)),
            contact_flags=zeros(segment_count, 4, dtype=torch.bool),
            segment_progress=zeros(segment_count),
            segment_moved=zeros(segment_count),
            body_height=zeros(segment_count),
            uprightness=zeros(segment_count),
            head_distance=zeros(),
            heading_error=zeros(),
            target_position=zeros(2),
            range_radius=zeros(),
        )
        self.episode = EpisodeSummary(
            episode_ended=zeros(dtype=torch.bool),
            arrived=zeros(dtype=torch.bool),
            left_range=zeros(dtype=torch.bool),
            ending=zeros(),
            length_steps=zeros(),
            segment_return=zeros(segment_count),
            start_distance=zeros(),
            final_distance=zeros(),
            distance_closed=zeros(),
            head_path_length=zeros(),
            segment_total_progress=zeros(segment_count),
            body_contact_share=zeros(segment_count),
            leg_contact_share=zeros(segment_count),
            foot_contact_share=zeros(segment_count, 2),
            upside_down_share=zeros(),
        )
        # Running totals of the current episode in every world.
        self._steps = zeros()
        self._segment_return = zeros(segment_count)
        self._head_path_length = zeros()
        self._segment_progress = zeros(segment_count)
        self._contact_steps = zeros(
            segment_count, 4
        )  # left foot, right foot, body, legs
        self._upside_down_steps = zeros()
        self._start_distance = zeros()
        # Remaining shares of the start distance that separate the endings
        # "within a quarter", "within half", "closer", and "not closer".
        self._remaining_edges = torch.tensor([0.25, 0.5, 1.0], device=device)

    def start_episodes(
        self,
        world_mask: torch.Tensor,
        physical_state: PhysicalState,
        target_position: torch.Tensor,
        range_radius: torch.Tensor,
    ) -> None:
        """Clear the totals of the masked worlds and note their start distance.

        Called after those worlds were reset and given new targets, with every
        world's range circle radius.
        """
        for total in (
            self._steps,
            self._segment_return,
            self._head_path_length,
            self._segment_progress,
            self._contact_steps,
            self._upside_down_steps,
        ):
            total.masked_fill_(_per_world(world_mask, total), 0.0)
        start = _head_distance(physical_state, target_position)
        self._start_distance.copy_(torch.where(world_mask, start, self._start_distance))
        # The new targets of the reset worlds, so that after a step every
        # world's target matches its pose.
        self.step.target_position.copy_(
            torch.where(world_mask[:, None], target_position, self.step.target_position)
        )
        self.step.range_radius.copy_(
            torch.where(world_mask, range_radius, self.step.range_radius)
        )

    def record_step(
        self,
        physical_state: PhysicalState,
        previous_body_planar_position: torch.Tensor,
        previous_head_tip_position: torch.Tensor,
        target_position: torch.Tensor,
        step_rewards: StepRewards,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        left_range: torch.Tensor,
    ) -> None:
        """Refresh the step facts, add to the totals, and publish ended episodes.

        Called once per step, after rewards and episode ends are known and
        before any world is reset; the arguments are values the front file
        already has. ``left_range`` marks the cut episodes whose head left the
        range circle.
        """
        step = self.step
        quaternion = physical_state.body_quaternion
        contact_flags = torch.stack(
            (
                physical_state.left_foot_ground_contact,
                physical_state.right_foot_ground_contact,
                physical_state.body_ground_contact,
                physical_state.leg_leg_contact,
            ),
            dim=-1,
        )
        head_tip = physical_state.head_tip_position[:, :2]
        to_target = target_position - head_tip
        forward = head_forward_direction(quaternion[:, 0])

        step.reward_parts.copy_(step_rewards.reward_parts)
        step.contact_flags.copy_(contact_flags)
        step.segment_progress.copy_(step_rewards.segment_progress)
        step.segment_moved.copy_(
            (physical_state.body_planar_position - previous_body_planar_position).norm(
                dim=-1
            )
        )
        step.body_height.copy_(physical_state.body_height)
        # The z component of each segment's up axis: 1 - 2(x² + y²).
        step.uprightness.copy_(
            1 - 2 * (quaternion[..., 1] ** 2 + quaternion[..., 2] ** 2)
        )
        step.head_distance.copy_(to_target.norm(dim=-1))
        step.target_position.copy_(target_position)
        step.heading_error.copy_(
            torch.atan2(
                forward[:, 0] * to_target[:, 1] - forward[:, 1] * to_target[:, 0],
                (forward * to_target).sum(dim=-1),
            ).abs()
        )

        self._steps += 1
        self._segment_return += step_rewards.rewards
        self._head_path_length += (head_tip - previous_head_tip_position).norm(dim=-1)
        self._segment_progress += step_rewards.segment_progress
        self._contact_steps += contact_flags.float()
        self._upside_down_steps += (step.uprightness[:, 0] < 0).float()

        ended = terminated | truncated
        steps = self._steps[:, None]
        episode = self.episode
        episode.episode_ended.copy_(ended)
        remaining = step.head_distance / self._start_distance
        ending = torch.where(
            terminated,
            0,
            torch.where(
                left_range,
                5,
                torch.bucketize(remaining, self._remaining_edges, right=True) + 1,
            ),
        )
        for summary, value in (
            (episode.arrived, terminated),
            (episode.left_range, left_range),
            # The middle of the ending's bin, so that it counts in that bin.
            (episode.ending, ending + 0.5),
            (episode.length_steps, self._steps),
            (episode.segment_return, self._segment_return),
            (episode.start_distance, self._start_distance),
            (episode.final_distance, step.head_distance),
            (episode.distance_closed, 1 - step.head_distance / self._start_distance),
            (episode.head_path_length, self._head_path_length),
            (episode.segment_total_progress, self._segment_progress),
            (episode.body_contact_share, self._contact_steps[..., 2] / steps),
            (episode.leg_contact_share, self._contact_steps[..., 3] / steps),
            (
                episode.foot_contact_share,
                self._contact_steps[..., :2] / steps[..., None],
            ),
            (episode.upside_down_share, self._upside_down_steps / self._steps),
        ):
            summary.copy_(torch.where(_per_world(ended, summary), value, summary))


def _per_world(world_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """A ``(W,)`` mask shaped to broadcast over a tensor whose rows are worlds."""
    return world_mask.reshape(world_mask.shape + (1,) * (like.dim() - 1))


def _head_distance(physical_state: PhysicalState, target_position: torch.Tensor):
    """Flat distance from each head's tip to its target, ``(W,)``."""
    return (target_position - physical_state.head_tip_position[:, :2]).norm(dim=-1)
