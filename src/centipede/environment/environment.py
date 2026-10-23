"""The environment's front file: the follow-line task built on the physics
simulation (docs/follow-line/).

It keeps each world's episode state and coordinates the line, the averaged
forces, the reward, the observations, the tow, the diagnostics and the
physics simulation. The interaction loop uses only ``reset`` and ``step``;
the experiment reads ``segment_count``, ``observation_size``,
``behind_observation_entries``, the action sizes and ``leg_motor_count`` to
create the agents, sets the ramp's weight
before each update cycle of training with ``set_ramp_weight``, and in
training with a tow calls ``start_tow`` and ``reserve_measuring_worlds``
first, ``adjust_tow`` after each update cycle, and reads ``tow.facts``,
``untowed_line`` and ``untowed_walk``. With curves it calls
``adjust_curve_level`` after each update cycle of training and keeps the
level in checkpoints (``line.level``, ``line.set_level``); with the speeds'
levels likewise ``adjust_speed_level``, reads ``speed_judgement`` and keeps
``speeds.level``.
"""

import math

import torch

from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.line import Line, forward_axis
from centipede.environment.observation_builder import ObservationBuilder
from centipede.environment.reward_function import (
    CHANNEL_NAMES,
    AveragedForces,
    RewardFunction,
)
from centipede.environment.settings import EnvironmentSettings
from centipede.environment.simulation import PhysicsSimulation
from centipede.environment.simulation.constants import ACTION_DURATION_S
from centipede.environment.speeds import CHANGE_WINDOW_S, Speeds
from centipede.environment.tow import GROUPS, Tow


class Environment:
    """Batches of centipede worlds, each following its line.

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

        self.line = Line(
            world_count,
            segment_count,
            body.unit_spacing_m,
            self.sight_radius_m,
            self.device,
            settings.curves if settings.line_shape == "curved" else None,
        )
        # With a start, the lines stay straight until the wave starts forming.
        if settings.line_shape == "curved" and settings.curves.start_speed_mm_per_s:
            self.line.set_level(-1)
        self.forces = AveragedForces(
            world_count,
            segment_count,
            body.unit_weight_n,
            min(1.0, ACTION_DURATION_S / settings.reward.averaging_s),
            self.device,
        )
        self.reward = RewardFunction(
            settings.reward,
            body,
            commanded_joints,
            self.sight_radius_m,
            settings.observation_radius,
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
            settings.observe_energy,
            settings.observe_line_speed,
            settings.reward.carry_per_leg,
            settings.observe_command,
            settings.observe_current_command,
            settings.observe_pain,
            settings.reward.pain_behind_only,
        )
        self.observation_size = self.observation_builder.observation_size
        self.behind_observation_entries = self.observation_builder.behind_entries
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
        # Each unit's forward line as a speed, mm/s, averaged as the forces
        # are, from 0 at an episode's start (the body at rest), which the
        # commanded speed reads.
        self.averaged_forward_line = torch.zeros(
            (world_count, segment_count), device=self.device
        )
        # Each unit's joints' speeds at the end of the last step, rad/s: its
        # legs' and the spine joint's behind it, for the accelerations.
        self.joint_velocity = torch.zeros(
            (world_count, segment_count, legs + 1), device=self.device
        )
        # Each unit's squared accelerations, averaged as the forces are, from
        # 0 at an episode's start (the body at rest).
        self.averaged_accelerations = torch.zeros(
            (world_count, segment_count), device=self.device
        )
        # Each unit's energy in mJ a second, averaged as the forces are, from
        # 0 at an episode's start.
        self.averaged_energy = torch.zeros(
            (world_count, segment_count), device=self.device
        )
        # Each motor's force on the last step, from 0 at an episode's start
        # (no force measured yet), for the cost of switching force.
        self.last_motor_force = torch.zeros(
            (world_count, segment_count, legs + 1), device=self.device
        )
        # Each world's commanded speed, mm/s, averaged as the forward line is,
        # from the straight command at an episode's start (the body on its
        # straight part), or with the speeds' schedule from 0, as the body's
        # speed; with slow_in_bends it falls in bends.
        self.commanded_speed = settings.reward.commanded_speed_mm_per_s
        self.averaged_command = torch.full(
            (world_count,), self.commanded_speed, device=self.device
        )
        # Stage 3: each world's command follows the speeds' schedule, whose
        # levels begin at once or once the curves reach their last.
        self.speeds = None
        if settings.speeds.level_hold_s:
            self.speeds = Speeds(
                settings.speeds, self.commanded_speed, world_count, self.device
            )
            if not settings.speeds.start_after_curves:
                self.speeds.set_level(0)
        # The untowed measuring worlds drawn at the speeds' level reached,
        # since it was last judged: their speed error's sum and steps, the
        # same within CHANGE_WINDOW_S of a change, their energy in mJ and
        # distance in mm while the command is above 0, and their lost lines;
        # the update cycles since; the last judgement's facts (None before).
        self.speed_sums = torch.zeros(7, device=self.device)
        self.speed_cycles_at_level = 0
        self.speed_judgement: dict[str, float] | None = None
        # The untowed measuring worlds' lost lines and steps, those laid at
        # the level reached alone, and the update cycles, since the curves'
        # level was last judged; the share of their episodes they kept the
        # line through then (NaN before).
        self.lost_lines = torch.zeros((), device=self.device)
        self.judged_world_steps = torch.zeros((), device=self.device)
        self.cycles_at_level = 0
        self.kept_line_share = math.nan

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

    def adjust_curve_level(self) -> bool:
        """After an update cycle of training: once ``level_judged_cycles``
        cycles have passed since the curves' level was last judged, judge
        it: raise it by one, up to the last, if the untowed measuring worlds
        kept the line through ``level_up_kept_share`` of their episodes over
        those cycles (lost lines per episode's worth of their steps, the worlds
        laid at that level alone), and start counting again; before the curves
        begin, nothing. Returns whether it rose; waits for the GPU once each
        judgement."""
        line = self.line
        curves = self.settings.curves
        if (
            line.level < 0
            or line.level >= line.level_count - 1
            or not self.tow.worlds_per_group
        ):
            return False
        self.cycles_at_level += 1
        if self.cycles_at_level < curves.level_judged_cycles:
            return False
        episodes = self.judged_world_steps.item() / self.settings.max_episode_steps
        if not episodes:
            return False
        self.kept_line_share = 1 - self.lost_lines.item() / episodes
        self._restart_judgement()
        if self.kept_line_share < curves.level_up_kept_share:
            return False
        line.set_level(line.level + 1)
        return True

    def start_curves(self) -> None:
        """Begin the curves at their first level from the next restarts on,
        once the wave starts forming (``CurveSettings``)."""
        self.line.set_level(0)
        self._restart_judgement()

    def _restart_judgement(self) -> None:
        self.lost_lines.zero_()
        self.judged_world_steps.zero_()
        self.cycles_at_level = 0

    def adjust_speed_level(self) -> bool:
        """After an update cycle of training, with the speeds' levels: before
        they begin, begin them once the curves reach their last level (with
        ``start_after_curves``); then, once ``level_judged_cycles`` cycles
        have passed since the level was last judged, judge it on the untowed
        measuring worlds drawn at it (``SpeedSettings``), keep the facts in
        ``speed_judgement``, raise the level by one, up to the last, if they
        pass, and start counting again. Returns whether it rose; waits for
        the GPU once each judgement."""
        speeds = self.speeds
        if speeds is None or not self.tow.worlds_per_group:
            return False
        settings = self.settings.speeds
        if speeds.level < 0:
            if self.line.level == self.line.level_count - 1:
                speeds.set_level(0)
                self.speed_sums.zero_()
            return False
        self.speed_cycles_at_level += 1
        if self.speed_cycles_at_level < settings.level_judged_cycles:
            return False
        error, steps, early_error, early_steps, energy, distance, lost = (
            self.speed_sums.tolist()
        )
        self.speed_sums.zero_()
        self.speed_cycles_at_level = 0
        if not steps:
            return False
        later_steps = steps - early_steps
        self.speed_judgement = judgement = {
            "level": speeds.level,
            "error": error / steps,
            "error_after_change": early_error / early_steps if early_steps else None,
            "error_later": (error - early_error) / later_steps if later_steps else None,
            "energy_per_mm": energy / distance if distance > 0 else None,
            "kept_share": 1 - lost * self.settings.max_episode_steps / steps,
        }
        passed = (
            judgement["error"] <= settings.level_up_speed_error_mm_per_s
            and judgement["energy_per_mm"] is not None
            and judgement["energy_per_mm"] <= settings.level_up_energy_mj_per_mm
            and judgement["kept_share"] >= settings.level_up_kept_share
        )
        if not passed or speeds.level >= speeds.level_count - 1:
            return False
        speeds.set_level(speeds.level + 1)
        return True

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
        forward_line = self.reward.forward_line(
            progress, self.line.distance, self.line.angle
        )
        self.averaged_command += self.forces.averaging_rate * (
            self._command() - self.averaged_command
        )
        self.averaged_forward_line += self.forces.averaging_rate * (
            forward_line / ACTION_DURATION_S - self.averaged_forward_line
        )
        self.averaged_accelerations += self.forces.averaging_rate * (
            self.reward.squared_accelerations(joint_acceleration)
            - self.averaged_accelerations
        )
        force_change = state.motor_force - self.last_motor_force
        self.last_motor_force.copy_(state.motor_force)
        self.averaged_energy += self.forces.averaging_rate * (
            self.reward.energy(state, force_change).sum(dim=-1) / ACTION_DURATION_S
            - self.averaged_energy
        )
        rewards, reward_parts, own_energy = self.reward.compute(
            state,
            progress,
            self.averaged_line_speed,
            self.averaged_forward_line,
            self.line.distance,
            self.line.angle,
            averaged_forces,
            self.tow.pushes,
            joint_acceleration,
            self.averaged_accelerations,
            self.averaged_energy,
            force_change,
            self.averaged_command,
        )
        strayed = self.line.distance.mean(dim=1) > self.sight_radius_m
        if self.line.level_count and self.tow.worlds_per_group:
            untowed = self.tow.groups[GROUPS.index("without tow")]
            judged = untowed * (self.line.world_levels == self.line.level)
            self.lost_lines += strayed.float() @ judged
            self.judged_world_steps += judged.sum()
        speeds = self.speeds
        if speeds is not None and speeds.level >= 0 and self.tow.worlds_per_group:
            untowed = self.tow.groups[GROUPS.index("without tow")]
            judged = untowed * (speeds.world_levels == speeds.level)
            error = (
                self.averaged_forward_line.mean(dim=1) - self.averaged_command
            ).abs()
            window = round(CHANGE_WINDOW_S / ACTION_DURATION_S)
            early = (speeds.steps_since_change < window).float()
            moving = (speeds.speeds > 0).float()
            self.speed_sums += (
                torch.stack(
                    (
                        error,
                        torch.ones_like(error),
                        error * early,
                        early,
                        own_energy.sum(dim=-1).mean(dim=-1) * moving,
                        progress.mean(dim=-1) * 1e3 * moving,
                        strayed.float(),
                    )
                )
                @ judged
            )
        if speeds is not None:
            speeds.step(self.generator)
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
            self.averaged_command,
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
        self.line.restart(world_mask, state, turn, self.generator)
        self.forces.restart(world_mask, state)
        self.tow.restart(world_mask)
        self.diagnostics.restart(world_mask)
        self.forward_speed.masked_fill_(world_mask[:, None], 0.0)
        self.averaged_line_speed.masked_fill_(
            world_mask[:, None], self.line_cap_m_per_s
        )
        self.last_action.masked_fill_(world_mask[:, None, None], 0.0)
        self.averaged_forward_line.masked_fill_(world_mask[:, None], 0.0)
        if self.speeds is None:
            self.averaged_command.masked_fill_(world_mask, self.commanded_speed)
        else:
            # A restart is a change from rest: the command is averaged from 0,
            # as the body's speed is, so their gap is the body's own lag.
            self.speeds.restart(world_mask, self.generator)
            self.averaged_command.masked_fill_(world_mask, 0.0)
        self.averaged_accelerations.masked_fill_(world_mask[:, None], 0.0)
        self.averaged_energy.masked_fill_(world_mask[:, None], 0.0)
        self.last_motor_force.masked_fill_(world_mask[:, None, None], 0.0)
        self.joint_velocity.copy_(
            torch.where(
                world_mask[:, None, None], self._joint_velocity(), self.joint_velocity
            )
        )

    def _command(self) -> torch.Tensor | float:
        """Each world's command on this step, mm/s: the speeds' schedule's or
        the reward's commanded speed, lowered in bends with ``slow_in_bends``
        just enough for the outer feet to keep it."""
        command = self.commanded_speed if self.speeds is None else self.speeds.speeds
        if self.settings.reward.slow_in_bends:
            bend = self.line.curvature.abs().mean(dim=1)
            command = command / (1 + self.body.foot_reach_m * bend)
        return command

    def _joint_velocity(self) -> torch.Tensor:
        """Each unit's leg joints' speeds and the spine joint's behind it,
        ``(W, N, M + 1)`` rad/s."""
        state = self.simulation.physical_state
        return torch.cat(
            (state.leg_joint_velocity, state.spine_yaw_velocity[..., None]), dim=-1
        )

    def _observations(self) -> torch.Tensor:
        current_command = None
        if self.settings.observe_current_command:
            current_command = torch.as_tensor(
                self._command(), device=self.device
            ).expand(self.world_count)
        return self.observation_builder.build(
            self.simulation.physical_state,
            self.last_action,
            self.forces.values,
            self.line.trail,
            self.tow.levels(),
            self.averaged_energy,
            self.averaged_forward_line,
            self.averaged_command,
            current_command,
        )
