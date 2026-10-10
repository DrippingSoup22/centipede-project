"""The environment's front file: the learning task built on the physics simulation.

It keeps each world's episode state (target, step count, previous positions,
and with clocks each segment's clock) and coordinates the observation builder,
the reward function, the diagnostics, and the physics simulation. The
interaction loop uses only ``reset`` and ``step``; the experiment reads
``segment_count``, ``observation_size``, and the action sizes to create the
agents, and its curriculum changes the target ranges between update cycles
with ``set_target_ranges``. See docs/environment.md.
"""

import torch

from centipede.environment.clocks import Clocks
from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.observation_builder import (
    ObservationBuilder,
    head_forward_direction,
)
from centipede.environment.reward_function import RewardFunction, joint_angles
from centipede.environment.settings import EnvironmentSettings
from centipede.environment.simulation import PhysicsSimulation
from centipede.settings_section import SettingsError


class Environment:
    """Batches of centipede worlds, each with a target for the head.

    Every tensor passed in or out has the worlds as its first dimension. When an
    episode ends in a world, ``step`` starts the next one by itself: it resets
    the world, or after an arrival may only give it a new target.

    ``segment_action_sizes`` lists how many actions each segment takes: six for
    its legs; with spine control, the spine joint behind it, for every segment
    but the rear (only for the head with a passive follower spine); and with
    clocks, last, its clock's tempo. ``action_size`` is the width of the joint
    action, the largest of them; a segment with fewer actions is padded.
    ``clocks`` holds the segments' clocks, or None without them.
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
            spine_observed=settings.spine_control,
            clock_observed=settings.clocks,
            neighbour_clock_observed=settings.neighbour_clocks,
        )
        self.observation_size = self.observation_builder.observation_size
        # Which segments command the spine joint behind them: with spine
        # control every segment but the rear, or only the head.
        spine_count = self.segment_count - 1
        commanders = 1 if settings.passive_follower_spine else spine_count
        commands_spine = [
            settings.spine_control and segment_index < commanders
            for segment_index in range(self.segment_count)
        ]
        self.segment_action_sizes = [
            6 + int(commands) + int(settings.clocks) for commands in commands_spine
        ]
        self.action_size = max(self.segment_action_sizes)
        # Each spine command, column 6 of the segment ahead of its joint, is
        # kept or replaced by zero; a follower's column 6 may be its tempo.
        self.spine_command_mask = torch.tensor(
            commands_spine[:spine_count], dtype=torch.float32, device=self.device
        )
        # Each segment's tempo action is its last.
        self.tempo_columns = torch.tensor(
            [size - 1 for size in self.segment_action_sizes], device=self.device
        )
        # The joints each segment commands, among its six leg joints and the
        # spine joint behind it.
        commanded_joints = torch.ones(
            (self.segment_count, 7), dtype=torch.float32, device=self.device
        )
        commanded_joints[:, 6] = torch.tensor(commands_spine, dtype=torch.float32)
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
            commanded_joints,
        )
        self.clocks = (
            Clocks(
                settings.clock,
                self.world_count,
                self.segment_count,
                settings.observation_radius,
                self.simulation.step_duration_s,
                self.device,
            )
            if settings.clocks
            else None
        )
        self.diagnostics = EnvironmentDiagnostics(
            self.world_count,
            self.segment_count,
            self.reward_function.term_names,
            self.device,
            self.simulation.diagnostics.facts,
            clocks=self.clocks,
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
        self.previous_joint_position = torch.zeros(
            (self.world_count, self.segment_count, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self.previous_foot_planar_position = torch.zeros(
            (self.world_count, self.segment_count, 2, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.previous_foot_contact = torch.zeros(
            (self.world_count, self.segment_count, 2),
            dtype=torch.bool,
            device=self.device,
        )
        # The radius of each world's range circle, around its target; infinite
        # when there is no circle.
        self.range_radius = torch.full(
            (self.world_count,), torch.inf, dtype=torch.float32, device=self.device
        )

        # Targets have their own random sequence, separate from the starting
        # poses, so changing how targets are drawn never changes the poses.
        self.target_generator = torch.Generator(device=self.device)
        # The ranges new targets are drawn from: [environment.target]'s, until
        # set_target_ranges changes them.
        self.distance_range_m = settings.target.distance_range_m
        self.bearing_range_deg = settings.target.bearing_range_deg

    def set_target_ranges(
        self,
        distance_range_m: tuple[float, float],
        bearing_range_deg: tuple[float, float],
    ) -> None:
        """Draw every new target from these ranges from now on.

        Targets already placed stay until their episode ends. The experiment's
        curriculum calls this between update cycles.
        """
        self.distance_range_m = distance_range_m
        self.bearing_range_deg = bearing_range_deg

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Start a new episode in every world; return the observations.

        A ``seed`` restarts the starting poses', the targets' and the clocks'
        random sequences, so the same seed repeats the same episodes' starts.
        """
        self.simulation.reset(seed=seed)
        if seed is not None:
            self.target_generator.manual_seed(seed)

        every_world = torch.ones(self.world_count, dtype=torch.bool, device=self.device)
        if self.clocks is not None:
            self.clocks.reset(every_world, seed)
        self._place_targets(every_world)
        self.episode_steps.zero_()
        state = self.simulation.physical_state
        self.diagnostics.start_episodes(
            every_world, state, self.target_position, self.range_radius
        )
        return self._observations()

    def step(
        self, joint_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the ``(W, N, action_size)`` joint action for one 20 ms step.

        Returns the observations, the rewards ``(W, N)``, terminated and
        truncated ``(W,)``, and the final observations. Worlds whose episode
        ended have already started a new one: their observations start the new
        episode, and their final observations are the ones the old episode
        ended with.
        """
        state = self.simulation.physical_state

        # 1. Copy the positions before the step: the physical state is
        # overwritten in place by the step.
        self.previous_body_planar_position.copy_(state.body_planar_position)
        self.previous_head_tip_position.copy_(state.head_tip_position[:, :2])
        self.previous_joint_position.copy_(joint_angles(state))
        self.previous_foot_planar_position.copy_(state.foot_planar_position)
        self.previous_foot_contact.copy_(_foot_contact(state))

        # 2. Move the body. With spine control, column 6 holds the command for
        # the spine joint behind each segment that commands one; without, the
        # spine is passive.
        if self.settings.spine_control:
            self.simulation.step(
                joint_action[..., :6],
                joint_action[:, :-1, 6] * self.spine_command_mask,
            )
        else:
            self.simulation.step(joint_action[..., :6])
        self.episode_steps += 1

        # The clocks turn at the tempo each segment chose, and the feet that
        # touched the ground before and after the step are measured for slip.
        clock_step = None
        if self.clocks is not None:
            tempo_columns = self.tempo_columns.expand(self.world_count, -1)
            tempo_action = joint_action.gather(2, tempo_columns[..., None])
            clock_step = self.clocks.step(
                tempo_action.squeeze(-1),
                self.previous_joint_position[..., :6],
                state.leg_joint_position,
            )
        planted = self.previous_foot_contact & _foot_contact(state)
        foot_slip_speed = (
            (state.foot_planar_position - self.previous_foot_planar_position).norm(
                dim=-1
            )
            / self.simulation.step_duration_s
            * planted
        )

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
            self.previous_joint_position,
            joint_action[..., :7],
            foot_slip_speed,
            clock_step,
        )
        observations = self._observations()
        self.diagnostics.record_step(
            state,
            self.previous_body_planar_position,
            self.previous_head_tip_position,
            self.target_position,
            step_rewards,
            terminated,
            truncated,
            left_range,
            clock_step,
        )

        # 6. Start a new episode in the worlds whose episode ended: restart
        # them, except, with after_arrival = "new_target", those that arrived,
        # whose body walks on toward a new target. ``build`` returns a new
        # tensor, so the final observations keep the values from before.
        ended = terminated | truncated
        final_observations = observations
        if ended.any():
            if self.settings.target.after_arrival == "new_target":
                restarted = truncated
            else:
                restarted = ended
            self.simulation.reset(restarted)
            if self.clocks is not None:
                self.clocks.reset(restarted)
            self._place_targets(ended)
            self.episode_steps.masked_fill_(ended, 0)
            self.diagnostics.start_episodes(
                ended, state, self.target_position, self.range_radius
            )
            observations = self._observations()

        return (
            observations,
            step_rewards.rewards,
            terminated,
            truncated,
            final_observations,
        )

    def _observations(self) -> torch.Tensor:
        """Every segment's observation of the current state."""
        clocks = self.clocks
        return self.observation_builder.build(
            self.simulation.physical_state,
            self.target_position,
            None if clocks is None else clocks.observation_values(),
            clocks.neighbour_values() if self.settings.neighbour_clocks else None,
        )

    def _place_targets(self, world_mask: torch.Tensor) -> None:
        """Give the masked worlds a new target around the head's tip.

        Distance and bearing are drawn uniformly from the current target
        ranges; a positive bearing is to the head's left. Values are drawn for every
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

        distance = uniform(*self.distance_range_m)
        bearing = torch.deg2rad(uniform(*self.bearing_range_deg))
        direction = (
            torch.cos(bearing)[:, None] * forward + torch.sin(bearing)[:, None] * left
        )
        new_target = head_tip + distance[:, None] * direction
        self.target_position.copy_(
            torch.where(world_mask[:, None], new_target, self.target_position)
        )
        if target_settings.range_circle_ratio:
            radius = (
                target_settings.range_circle_ratio * distance
                + target_settings.range_circle_margin_m
            )
            self.range_radius.copy_(torch.where(world_mask, radius, self.range_radius))

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


def _foot_contact(state) -> torch.Tensor:
    """Each segment's left and right foot on the ground, ``(W, N, 2)``."""
    return torch.stack(
        (state.left_foot_ground_contact, state.right_foot_ground_contact), dim=-1
    )
