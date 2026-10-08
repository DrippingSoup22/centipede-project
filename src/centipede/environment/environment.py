"""The environment's front file: the learning task built on the physics simulation.

It keeps each world's episode state (target, step count, previous positions)
and coordinates the observation builder, the reward function, the diagnostics,
and the physics simulation. The interaction loop uses only ``reset`` and
``step``; the experiment reads ``segment_count`` and ``observation_size`` to
create the agents. See docs/environment.md.
"""

import torch

from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.observation_builder import (
    ObservationBuilder,
    head_forward_direction,
)
from centipede.environment.reward_function import RewardFunction
from centipede.environment.settings import EnvironmentSettings
from centipede.environment.simulation import PhysicsSimulation
from centipede.settings_section import SettingsError


class Environment:
    """Batches of centipede worlds, each with a target for the head.

    Every tensor passed in or out has the worlds as its first dimension. When an
    episode ends in a world, ``step`` resets that world by itself.
    """

    def __init__(self, settings: EnvironmentSettings) -> None:
        """Create the simulation and the parts, and allocate the episode state."""
        self.settings = settings
        self.simulation = PhysicsSimulation(settings.simulation)
        self.world_count = self.simulation.world_count
        self.segment_count = self.simulation.segment_count
        self.device = self.simulation.physical_state.body_height.device

        # The radius comes from the configuration and the segment count from
        # the model, so they can only be checked together, here.
        if settings.observation_radius >= self.segment_count:
            raise SettingsError(
                f"[environment] observation_radius must be less than the model's "
                f"{self.segment_count} segments, got {settings.observation_radius}"
            )

        self.observation_builder = ObservationBuilder(
            self.segment_count,
            settings.observation_radius,
            self.world_count,
            self.device,
        )
        self.observation_size = self.observation_builder.observation_size
        # Arrival under the head counts progress no closer than half the head's
        # width: within the head's reach, nothing pays more.
        self.head_outline = self.simulation.head_outline
        target = settings.target
        self.reward_function = RewardFunction(
            settings.rewards,
            settings.max_episode_steps,
            self.head_outline.half_width_m
            if target.arrival == "head"
            else target.arrival_radius_m,
        )
        self.diagnostics = EnvironmentDiagnostics(
            self.world_count,
            self.segment_count,
            self.reward_function.term_names,
            self.device,
            self.simulation.diagnostics.facts,
        )

        # Episode state, overwritten in place.
        self.target_position = torch.zeros(
            (self.world_count, 2), dtype=torch.float32, device=self.device
        )
        self.episode_steps = torch.zeros(
            self.world_count, dtype=torch.int32, device=self.device
        )
        self.previous_body_planar_position = torch.zeros(
            (self.world_count, self.segment_count, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.previous_head_tip_position = torch.zeros(
            (self.world_count, 2), dtype=torch.float32, device=self.device
        )
        # The radius of each world's range circle, around its target; infinite
        # when there is no circle.
        self.range_radius = torch.full(
            (self.world_count,), torch.inf, dtype=torch.float32, device=self.device
        )

        # Targets have their own random sequence, separate from the starting
        # poses, so changing how targets are drawn never changes the poses.
        self.target_generator = torch.Generator(device=self.device)

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Start a new episode in every world; return the observations.

        A ``seed`` restarts both the starting poses' and the targets' random
        sequences, so the same seed repeats the same episodes' starts.
        """
        self.simulation.reset(seed=seed)
        if seed is not None:
            self.target_generator.manual_seed(seed)

        every_world = torch.ones(self.world_count, dtype=torch.bool, device=self.device)
        self._place_targets(every_world)
        self.episode_steps.zero_()
        state = self.simulation.physical_state
        self.diagnostics.start_episodes(
            every_world, state, self.target_position, self.range_radius
        )
        return self.observation_builder.build(state, self.target_position)

    def step(
        self, joint_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the ``(W, N, 6)`` joint action for one 20 ms step.

        Returns the observations, the rewards ``(W, N)``, terminated and
        truncated ``(W,)``, and the final observations. Worlds whose episode
        ended are already reset: their observations start the new episode, and
        their final observations are the ones the old episode ended with.
        """
        state = self.simulation.physical_state

        # 1. Copy the positions before the step: the physical state is
        # overwritten in place by the step.
        self.previous_body_planar_position.copy_(state.body_planar_position)
        self.previous_head_tip_position.copy_(state.head_tip_position[:, :2])

        # 2. Move the body.
        self.simulation.step(joint_action)
        self.episode_steps += 1

        # 3. Arrival, or a cut: the time limit or leaving the range circle,
        # which end the episode alike. Arrival wins when both happen.
        head_distance = (self.target_position - state.head_tip_position[:, :2]).norm(
            dim=-1
        )
        if self.settings.target.arrival == "head":
            terminated = self._target_under_head()
        else:
            terminated = head_distance <= self.settings.target.arrival_radius_m
        left_range = (head_distance > self.range_radius) & ~terminated
        time_is_up = self.episode_steps >= self.settings.max_episode_steps
        truncated = (time_is_up | left_range) & ~terminated

        # 4. and 5. Rewards, observations, diagnostics.
        step_rewards = self.reward_function.compute(
            state,
            self.previous_body_planar_position,
            self.previous_head_tip_position,
            self.target_position,
            terminated,
        )
        observations = self.observation_builder.build(state, self.target_position)
        self.diagnostics.record_step(
            state,
            self.previous_body_planar_position,
            self.previous_head_tip_position,
            self.target_position,
            step_rewards,
            terminated,
            truncated,
            left_range,
        )

        # 6. Reset the worlds whose episode ended. ``build`` returns a new
        # tensor, so the final observations keep the values from before.
        ended = terminated | truncated
        final_observations = observations
        if ended.any():
            self.simulation.reset(ended)
            self._place_targets(ended)
            self.episode_steps.masked_fill_(ended, 0)
            self.diagnostics.start_episodes(
                ended, state, self.target_position, self.range_radius
            )
            observations = self.observation_builder.build(state, self.target_position)

        return (
            observations,
            step_rewards.rewards,
            terminated,
            truncated,
            final_observations,
        )

    def _place_targets(self, world_mask: torch.Tensor) -> None:
        """Give the masked worlds a new target in front of the head's tip.

        Distance and bearing are drawn uniformly from ``[environment.target]``;
        a positive bearing is to the head's left. Values are drawn for every
        world and kept only where the mask is set, which avoids asking the GPU
        how many worlds are masked.
        """
        state = self.simulation.physical_state
        target_settings = self.settings.target
        head_tip = state.head_tip_position[:, :2]
        forward = head_forward_direction(state.body_quaternion[:, 0])
        left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)

        def uniform(low: float, high: float) -> torch.Tensor:
            """One value per world, uniform between ``low`` and ``high``."""
            draw = torch.rand(
                self.world_count, generator=self.target_generator, device=self.device
            )
            return low + (high - low) * draw

        distance = uniform(*target_settings.distance_range_m)
        bearing = torch.deg2rad(uniform(*target_settings.bearing_range_deg))
        direction = (
            torch.cos(bearing)[:, None] * forward + torch.sin(bearing)[:, None] * left
        )
        new_target = head_tip + distance[:, None] * direction
        self.target_position.copy_(
            torch.where(world_mask[:, None], new_target, self.target_position)
        )
        if target_settings.range_circle_ratio:
            self.range_radius.copy_(
                torch.where(
                    world_mask,
                    target_settings.range_circle_ratio * distance,
                    self.range_radius,
                )
            )

    def _target_under_head(self) -> torch.Tensor:
        """Worlds whose target lies under the head's outline, seen from above.

        The target is measured from the head's centre along the head's forward
        direction and to its side, and compared with the rectangle that encloses
        the head's body shape.
        """
        state = self.simulation.physical_state
        outline = self.head_outline
        offset = self.target_position - state.body_planar_position[:, 0]
        forward = head_forward_direction(state.body_quaternion[:, 0])
        along = (offset * forward).sum(dim=-1)
        across = forward[:, 0] * offset[:, 1] - forward[:, 1] * offset[:, 0]
        return (
            (along >= outline.rear_m)
            & (along <= outline.front_m)
            & (across.abs() <= outline.half_width_m)
        )
