"""The cooperators' environment: the same interface as the main environment,
with the task, observations and reward of docs/cooperators/.

The interaction loop uses ``reset`` and ``step``; the experiment reads the
same sizes as the main environment's, plus ``neighbour_table`` for the
agents' critics, and after each update cycle calls ``adjust_level`` and
reads ``task``. In training with a tow it calls ``start_tow`` and
``reserve_measuring_worlds`` first, ``adjust_tow`` after each update cycle,
and reads ``tow.facts``.
"""

from dataclasses import dataclass

import torch

from centipede.diagnostics_category import measure
from centipede.environment.cooperative_diagnostics import CooperativeDiagnostics
from centipede.environment.cooperative_line import HeadLine
from centipede.environment.cooperative_observations import (
    CooperativeObservationBuilder,
)
from centipede.environment.cooperative_reward import (
    CHANNEL_NAMES,
    AveragedForces,
    CooperativeReward,
)
from centipede.environment.cooperative_task import CooperativeTask
from centipede.environment.cooperative_tow import Tow
from centipede.environment.head import target_under_head
from centipede.environment.reward_function import joint_angles
from centipede.environment.settings import EnvironmentSettings
from centipede.environment.simulation import PhysicsSimulation
from centipede.environment.simulation.constants import ACTION_DURATION_S
from centipede.settings_section import SettingsError


@dataclass(frozen=True)
class TaskFacts:
    """The task's state after an update cycle; single values, on the CPU."""

    level: torch.Tensor = measure("The targets' difficulty, from 0 to 1")
    in_time_share: torch.Tensor = measure(
        "Share of the targets whose stopwatch ended over the last episode"
        " length that were reached within it; NaN before one is recorded"
    )


class CooperativeEnvironment:
    """Batches of centipede worlds walking on from target to target.

    Every tensor passed in or out has the worlds as its first dimension. Every
    segment commands its leg motors and, but the rear, the spine joint behind
    it; ``segment_action_sizes`` and ``action_size`` are as in the main
    environment with spine control; each spine motor receives
    ``spine_action_scale`` times its command. Rewards have two channels, goal
    and health, ``(W, N, 2)``, and every segment's critics learn from both
    (``segment_reward_channels``). An arrival
    never ends an episode: the next target appears at once. The only end is
    the time limit, a cut that the agents bootstrap from; the world then
    restarts. ``diagnostics`` summarise each step and each episode. A
    started ``tow`` pushes every segment forward (cooperative_tow.py).
    """

    def __init__(self, settings: EnvironmentSettings) -> None:
        """Create the simulation, measuring effort and the joint forces, and
        the design's parts."""
        self.settings = settings
        cooperators = settings.cooperators
        self.simulation = PhysicsSimulation(
            settings.simulation, measure_effort=True, measure_joint_forces=True
        )
        self.world_count = self.simulation.world_count
        self.segment_count = segment_count = self.simulation.segment_count
        self.leg_motor_count = legs = self.simulation.leg_motor_count
        self.device = self.simulation.physical_state.body_height.device
        radius = settings.observation_radius
        if radius >= segment_count:
            raise SettingsError(
                f"[environment] observation_radius must be less than the model's "
                f"{segment_count} segments, got {radius}"
            )
        body = self.simulation.body_measures

        self.segment_action_sizes = [legs + 1] * (segment_count - 1) + [legs]
        self.action_size = legs + 1
        commanded_joints = torch.ones(
            (segment_count, legs + 1), dtype=torch.float32, device=self.device
        )
        commanded_joints[-1, -1] = 0.0
        # What each motor receives for a command of 1: the spine's scaled.
        self.motor_scale = torch.ones(legs + 1, device=self.device)
        self.motor_scale[-1] = cooperators.spine_action_scale
        self.head_outline = self.simulation.head_outline

        self.observation_builder = CooperativeObservationBuilder(
            segment_count,
            radius,
            self.world_count,
            legs,
            body.unit_weight_n,
            self.device,
            instant_joint_forces=cooperators.observe_instant_joint_forces,
            tow_input=cooperators.observe_tow,
        )
        self.observation_size = self.observation_builder.observation_size
        self.neighbour_table = self.observation_builder.neighbour_table
        self.task = CooperativeTask(
            cooperators, body, self.world_count, settings.max_episode_steps, self.device
        )
        # Each step moves an average this share of the way (1: none).
        averaging_rate = (
            min(1.0, ACTION_DURATION_S / cooperators.averaging_s)
            if cooperators.averaging_s > 0
            else 1.0
        )
        self.line = HeadLine(
            self.world_count,
            body.segment_lengths_m,
            settings.max_episode_steps,
            self.device,
        )
        self.forces = AveragedForces(
            self.world_count,
            segment_count,
            body.unit_weight_n,
            averaging_rate,
            self.device,
        )
        self.reward = CooperativeReward(
            cooperators, segment_count, radius, body, commanded_joints
        )
        self.tow = Tow(
            cooperators,
            self.world_count,
            body.segment_weights_n,
            body.unit_weight_n,
            settings.max_episode_steps,
            self.device,
        )
        self.reward_channel_count = len(CHANNEL_NAMES)
        self.segment_reward_channels = [(0, 1)] * segment_count
        self.diagnostics = CooperativeDiagnostics(
            self.world_count,
            segment_count,
            body.unit_weight_n,
            self.head_outline,
            self.device,
            self.simulation.diagnostics.facts,
            CHANNEL_NAMES,
        )
        self.task_facts = TaskFacts(
            level=torch.tensor(self.task.level), in_time_share=torch.tensor(torch.nan)
        )

        # Episode state, overwritten in place.
        state = self.simulation.physical_state
        self.episode_steps = torch.zeros(
            self.world_count, dtype=torch.int32, device=self.device
        )
        self.last_action = torch.zeros(
            (self.world_count, segment_count, legs + 1), device=self.device
        )
        self.previous_body_planar_position = torch.zeros_like(
            state.body_planar_position
        )
        self.previous_head_tip_position = torch.zeros_like(
            state.head_tip_position[:, :2]
        )
        self.previous_joint_position = torch.zeros_like(self.last_action)

    @property
    def level(self) -> float:
        return self.task.level

    def set_level(self, level: float) -> None:
        """Place the following targets at ``level``, as a continued run does."""
        self.task.level = level
        self.task_facts.level.fill_(level)

    def adjust_level(self) -> None:
        """After an update cycle: move the level by the share of targets reached
        in time, and refresh ``task_facts``."""
        self.task.adjust_level()
        share = self.task.in_time_share
        self.task_facts.level.fill_(self.task.level)
        self.task_facts.in_time_share.fill_(torch.nan if share is None else share)

    def start_tow(self, top: float | None = None) -> None:
        """Tow every segment from the next step on, under the settings'
        ``tow_start`` or a saved ``top``; only training does."""
        self.tow.start(self.settings.cooperators.tow_start if top is None else top)

    def reserve_measuring_worlds(self, worlds_per_group: int) -> None:
        """The last ``2 x worlds_per_group`` worlds measure the agents' mean
        action, half of them at the tow's top and half without tow; the
        level no longer counts them."""
        self.tow.reserve(worlds_per_group)
        self.task.counted_worlds[self.world_count - 2 * worlds_per_group :] = False

    def adjust_tow(self) -> None:
        """After an update cycle: move the tow's top by what the towed
        measuring worlds did, and refresh ``tow.facts``."""
        self.tow.adjust()

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Start a new episode in every world; return the observations.

        A ``seed`` restarts the starting poses' and the targets' random
        sequences. With ``stagger_first_episodes`` every world starts its
        episode at a random step of it, drawn before the targets.
        """
        self.simulation.reset(seed=seed)
        if seed is not None:
            self.task.generator.manual_seed(seed)
            self.tow.generator.manual_seed(seed)
        state = self.simulation.physical_state
        every_world = torch.ones(self.world_count, dtype=torch.bool, device=self.device)
        self._restart_measures(every_world)
        self.episode_steps.zero_()
        if self.settings.cooperators.stagger_first_episodes:
            self.episode_steps.random_(
                0, self.settings.max_episode_steps, generator=self.task.generator
            )
        self.task.place_targets(every_world, state, self._steps_left())
        self.tow.restart(every_world)
        self.last_action.zero_()
        self.diagnostics.new_targets(
            every_world, every_world, state, self.task.target_position
        )
        return self._observations()

    def step(
        self, joint_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the ``(W, N, M + 1)`` joint action for one 20 ms step.

        Returns the observations, the rewards ``(W, N, C)``, terminated (never)
        and truncated ``(W,)``, and the final observations. A world that
        arrived already sees its next target in both; a world whose time ran
        out is already reset, and its final observation is the one its episode
        ended with.
        """
        state = self.simulation.physical_state
        task = self.task
        self.previous_body_planar_position.copy_(state.body_planar_position)
        self.previous_head_tip_position.copy_(state.head_tip_position[:, :2])
        self.previous_joint_position.copy_(joint_angles(state))

        legs = self.leg_motor_count
        motor_command = joint_action * self.motor_scale
        leg_commands = motor_command[..., :legs]
        spine_commands = motor_command[:, :-1, legs]
        if self.tow.started:
            self.simulation.step(
                leg_commands,
                spine_commands,
                body_forces=self.tow.forces(state.body_quaternion),
            )
        else:
            self.simulation.step(leg_commands, spine_commands)
        self.episode_steps += 1
        line_erased, off_line = self.line.update(state)
        averaged_forces = self.forces.update(state)

        arrived = target_under_head(state, task.target_position, self.head_outline)
        in_time = task.time_step(arrived)
        near, far = task.distance_range_m()
        rewards, step_rewards, own_energy = self.reward.compute(
            state,
            self.previous_body_planar_position,
            self.previous_head_tip_position,
            task.target_position,
            arrived,
            self.settings.cooperators.arrival_approaches * (near + far) / 2,
            self.previous_joint_position,
            motor_command,
            line_erased,
            off_line,
            averaged_forces,
            self.tow.unit_weights(),
        )
        truncated = self.episode_steps >= self.settings.max_episode_steps
        self.diagnostics.record_step(
            state,
            self.previous_body_planar_position,
            self.previous_head_tip_position,
            task.target_position,
            rewards,
            step_rewards,
            own_energy,
            off_line,
            averaged_forces,
            arrived,
            in_time,
            truncated,
        )
        self.tow.record(self.diagnostics.step)

        # Walking on: a reached target is replaced where the head stands.
        self.last_action.copy_(joint_action)
        task.place_targets(arrived & ~truncated, state, self._steps_left())
        final_observations = observations = self._observations()
        if truncated.any():
            self.simulation.reset(truncated)
            self._restart_measures(truncated)
            self.episode_steps.masked_fill_(truncated, 0)
            task.place_targets(truncated, state, self._steps_left())
            self.tow.restart(truncated)
            self.last_action.masked_fill_(truncated[:, None, None], 0.0)
            observations = self._observations()
        self.diagnostics.new_targets(
            arrived | truncated, truncated, state, task.target_position
        )
        never = torch.zeros_like(truncated)
        return observations, rewards, never, truncated, final_observations

    def _restart_measures(self, world_mask: torch.Tensor) -> None:
        """Start the masked worlds' lines and averaged forces afresh."""
        state = self.simulation.physical_state
        self.line.restart(world_mask, state)
        self.forces.restart(world_mask, state)

    def _steps_left(self) -> torch.Tensor:
        """Each world's steps until its episode's time limit."""
        return self.settings.max_episode_steps - self.episode_steps

    def _observations(self) -> torch.Tensor:
        return self.observation_builder.build(
            self.simulation.physical_state,
            self.task.target_position,
            self.last_action,
            self.line.view,
            self.forces.values,
            self.tow.levels(),
        )
