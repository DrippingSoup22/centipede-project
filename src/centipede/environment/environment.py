"""The environment's front file: the follow-line task built on the physics
simulation (docs/follow-line/).

It keeps each world's episode state and coordinates the line, the averaged
forces, the reward, the observations, the tow, the diagnostics and the
physics simulation. The interaction loop uses only ``reset`` and ``step``;
the experiment reads ``segment_count``, ``observation_size``, the action
sizes and ``leg_motor_count`` to create the agents, sets the ramp's weight
before each update cycle of training with ``set_ramp_weight``, and in
training with a tow calls ``start_tow`` and ``reserve_measuring_worlds``
first, ``adjust_tow`` after each update cycle, and reads ``tow.facts``,
``untowed_line`` and ``untowed_walk``.
"""

import math

import torch

from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.line import StraightLine, forward_axis
from centipede.environment.observation_builder import ObservationBuilder
from centipede.environment.reward_function import (
    CHANNEL_NAMES,
    AveragedForces,
    RewardFunction,
)
from centipede.environment.settings import EnvironmentSettings
from centipede.environment.simulation import PhysicsSimulation
from centipede.environment.simulation.constants import ACTION_DURATION_S
from centipede.environment.tow import GROUPS, Tow


class Environment:
    """Batches of centipede worlds, each following its straight line.

    Every tensor passed in or out has the worlds as its first dimension.
    Every unit commands its leg motors and, but the rear, the spine joint
    behind it: ``segment_action_sizes`` lists each unit's actions and
    ``action_size`` is the width of the joint action, the rear's last column
    going to no motor. Each spine motor receives ``spine_action_scale`` times
    its command. Rewards have two accounts, team and own, ``(W, N, 2)``. An
    episode never terminates: it is cut, and the agents value what would have
    followed, at the time limit, when the units' mean distance from the line
    passes their sight radius, or when the physics restarted the world; the
    world then restarts on a new line.
    """

    def __init__(self, settings: EnvironmentSettings) -> None:
        """Create the simulation, measuring effort and the joint forces, and
        the task's parts."""
        self.settings = settings
        self.simulation = PhysicsSimulation(
            settings.simulation, measure_effort=True, measure_joint_forces=True
        )
        self.world_count = world_count = self.simulation.world_count
        self.segment_count = segment_count = self.simulation.segment_count
        self.leg_motor_count = legs = self.simulation.leg_motor_count
        self.device = self.simulation.physical_state.body_height.device
        self.body = body = self.simulation.body_measures
        self.sight_radius_m = settings.sight_radius_spacings * body.unit_spacing_m

        self.segment_action_sizes = [legs + 1] * (segment_count - 1) + [legs]
        self.action_size = legs + 1
        commanded_joints = torch.ones((segment_count, legs + 1), device=self.device)
        commanded_joints[-1, -1] = 0.0
        # What each motor receives for a command of 1: the spine's scaled.
        self.motor_scale = torch.ones(legs + 1, device=self.device)
        self.motor_scale[-1] = settings.spine_action_scale

        self.line = StraightLine(
            world_count,
            segment_count,
            body.unit_spacing_m,
            self.sight_radius_m,
            self.device,
        )
        self.forces = AveragedForces(
            world_count,
            segment_count,
            body.unit_weight_n,
            min(1.0, ACTION_DURATION_S / settings.reward.averaging_s),
            self.device,
        )
        self.reward = RewardFunction(
            settings.reward, body, commanded_joints, self.sight_radius_m
        )
        # The accounts the agents learn from: the team's and each unit's own,
        # or the team's alone, which then holds every part.
        self.reward_channel_count = (
            len(CHANNEL_NAMES) if settings.reward.own_account else 1
        )
        self.observation_builder = ObservationBuilder(
            segment_count,
            settings.observation_radius,
            world_count,
            legs,
            body.unit_weight_n,
            self.device,
            settings.observe_priced_forces,
            settings.observe_carry,
        )
        self.observation_size = self.observation_builder.observation_size
        self.tow = Tow(
            settings.tow,
            world_count,
            body.segment_weights_n,
            body.unit_weight_n,
            self.device,
        )
        self.diagnostics = EnvironmentDiagnostics(
            world_count,
            segment_count,
            body.unit_weight_n,
            self.device,
            self.simulation.diagnostics.facts,
            body.body_length_m,
        )

        # Episode state, overwritten in place. The lines' angles and the first
        # episodes' staggering have their own random sequence.
        self.generator = torch.Generator(device=self.device)
        self.episode_steps = torch.zeros(
            world_count, dtype=torch.int32, device=self.device
        )
        self.last_action = torch.zeros(
            (world_count, segment_count, legs + 1), device=self.device
        )
        # Each unit's speed along its forward axis over the last step, m/s,
        # which the tow reads.
        self.forward_speed = torch.zeros(
            (world_count, segment_count), device=self.device
        )
        # Each unit's speed along the line, m/s, averaged as the forces are,
        # whose cap the line pays with ``line_cap_averaged``. An episode
        # starts it at the cap: the pay reads only how far above the cap it
        # is, so a slow start is paid as from 0, and a fast one gains nothing.
        self.line_cap_m_per_s = settings.reward.line_speed_cap_mm_per_s * 1e-3
        self.averaged_line_speed = torch.zeros(
            (world_count, segment_count), device=self.device
        )
        # Each unit's joints' speeds at the end of the last step, rad/s: its
        # legs' and the spine joint's behind it, for the accelerations.
        self.joint_velocity = torch.zeros(
            (world_count, segment_count, legs + 1), device=self.device
        )

    def set_ramp_weight(self, weight: float) -> None:
        """Weigh the ramped costs (energy, the squeeze, pain, the motion and
        the accelerations) by ``weight`` from the next step on, as the ramp of
        the coming update cycle says."""
        self.reward.ramp_weight = weight

    def start_tow(self, top: float | None = None) -> None:
        """Tow every unit from the next step on, under the settings' start or
        a saved ``top``; only training does."""
        self.tow.start(self.settings.tow.start if top is None else top)

    def reserve_measuring_worlds(self, worlds_per_group: int) -> None:
        """The last ``2 x worlds_per_group`` worlds measure the agents' mean
        action, half of them at the tow's top and half without tow."""
        self.tow.reserve(worlds_per_group)

    def adjust_tow(self) -> bool:
        """After an update cycle: move the tow's top by what the untowed
        measuring worlds did, and refresh ``tow.facts``, once a window;
        returns whether a window closed."""
        return self.tow.adjust()

    def untowed_line(self) -> float:
        """The line paid to the untowed measuring worlds over the tow's last
        window, mm/s, the units' mean; NaN before the first."""
        return self.tow.facts.line[GROUPS.index("without tow")].item()

    def untowed_walk(self) -> tuple[float, bool]:
        """How fast the untowed measuring worlds went along the line over the
        tow's last window, mm/s, the units' mean, and whether their feet
        stepped, lifting off above the tow's bar."""
        untowed = GROUPS.index("without tow")
        facts = self.tow.facts
        stepping = facts.lift_offs[untowed].item() > self.settings.tow.lift_off_hz
        return facts.speed[untowed].item(), stepping

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Start a new episode in every world; return the observations.

        A ``seed`` restarts the starting poses', the lines' and the tow's
        random sequences. With ``stagger_first_episodes`` every world starts
        its episode at a random step of it.
        """
        self.simulation.reset(seed=seed)
        if seed is not None:
            self.generator.manual_seed(seed)
            self.tow.generator.manual_seed(seed)
        every_world = torch.ones(self.world_count, dtype=torch.bool, device=self.device)
        self.episode_steps.zero_()
        if self.settings.stagger_first_episodes:
            self.episode_steps.random_(
                0, self.settings.max_episode_steps, generator=self.generator
            )
        self._restart(every_world)
        return self._observations()

    def step(
        self, joint_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the ``(W, N, M + 1)`` joint action for one 20 ms step.

        Returns the observations, the rewards ``(W, N, 2)`` (``(W, N)`` with
        one account), terminated
        (never) and truncated ``(W,)``, and the final observations. A world
        whose episode was cut is already restarted: its observations start
        the new episode, and its final observations are the ones the old
        episode ended with.
        """
        state = self.simulation.physical_state
        before = state.body_planar_position.clone()
        legs = self.leg_motor_count
        motor_command = joint_action * self.motor_scale
        body_forces = (
            self.tow.forces(state.body_quaternion, self.forward_speed)
            if self.tow.started
            else None
        )
        self.simulation.step(
            motor_command[..., :legs], motor_command[:, :-1, legs], body_forces
        )
        if self.tow.started:
            moved = state.body_planar_position - before
            self.forward_speed.copy_(
                (moved * forward_axis(state)).sum(dim=-1) / ACTION_DURATION_S
            )
        self.episode_steps += 1

        progress = self.line.update(state)
        averaged_forces = self.forces.update(state)
        velocity = self._joint_velocity()
        joint_acceleration = (velocity - self.joint_velocity) / ACTION_DURATION_S
        self.joint_velocity.copy_(velocity)
        self.averaged_line_speed += self.forces.averaging_rate * (
            progress / ACTION_DURATION_S - self.averaged_line_speed
        )
        rewards, reward_parts, own_energy = self.reward.compute(
            state,
            motor_command,
            progress,
            self.averaged_line_speed,
            self.line.distance,
            self.line.angle,
            averaged_forces,
            self.tow.pushes,
            joint_acceleration,
        )
        strayed = self.line.distance.mean(dim=1) > self.sight_radius_m
        truncated = (
            (self.episode_steps >= self.settings.max_episode_steps)
            | strayed
            | self.simulation.restarted_worlds
        )
        self.diagnostics.record_step(
            state,
            rewards,
            reward_parts,
            own_energy,
            progress,
            self.line,
            averaged_forces,
            strayed,
            truncated,
        )
        self.tow.record(self.diagnostics.step)

        self.last_action.copy_(joint_action)
        final_observations = observations = self._observations()
        if truncated.any():
            self.simulation.reset(truncated)
            self.episode_steps.masked_fill_(truncated, 0)
            self._restart(truncated)
            observations = self._observations()
        return (
            observations,
            rewards if self.reward_channel_count > 1 else rewards[..., 0],
            torch.zeros_like(truncated),
            truncated,
            final_observations,
        )

    def _restart(self, world_mask: torch.Tensor) -> None:
        """Start the masked worlds' episodes afresh, as they stand after the
        simulation's reset: a new line, averages, tow level and gait."""
        state = self.simulation.physical_state
        draw = torch.rand(
            self.world_count, generator=self.generator, device=self.device
        )
        turn = (2 * draw - 1) * math.radians(self.settings.line_angle_range_deg)
        self.line.restart(world_mask, state, turn)
        self.forces.restart(world_mask, state)
        self.tow.restart(world_mask)
        self.diagnostics.restart(world_mask)
        self.forward_speed.masked_fill_(world_mask[:, None], 0.0)
        self.averaged_line_speed.masked_fill_(
            world_mask[:, None], self.line_cap_m_per_s
        )
        self.last_action.masked_fill_(world_mask[:, None, None], 0.0)
        self.joint_velocity.copy_(
            torch.where(
                world_mask[:, None, None], self._joint_velocity(), self.joint_velocity
            )
        )

    def _joint_velocity(self) -> torch.Tensor:
        """Each unit's leg joints' speeds and the spine joint's behind it,
        ``(W, N, M + 1)`` rad/s."""
        state = self.simulation.physical_state
        return torch.cat(
            (state.leg_joint_velocity, state.spine_yaw_velocity[..., None]), dim=-1
        )

    def _observations(self) -> torch.Tensor:
        return self.observation_builder.build(
            self.simulation.physical_state,
            self.last_action,
            self.forces.values,
            self.line.trail,
            self.tow.levels(),
        )
