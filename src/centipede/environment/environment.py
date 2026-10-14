"""The environment's front file: the learning task built on the physics simulation.

It keeps each world's episode state (target, step count, previous positions,
and with clocks each segment's or each leg's clock) and coordinates the
observation builder, the reward function, the diagnostics, and the physics
simulation. The interaction loop uses only ``reset`` and ``step``; the
experiment reads ``segment_count``, ``observation_size``, and the action sizes
and columns to create the agents, and its curriculum changes the target ranges
between update cycles with ``set_target_ranges``. See docs/environment.md.
"""

import math

import torch

from centipede.environment.clocks import Clocks, leg_targets
from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.observation_builder import (
    ObservationBuilder,
    head_forward_direction,
)
from centipede.environment.reward_function import RewardFunction, joint_angles
from centipede.environment.settings import STEP_VALUE_CHECK_KEYS, EnvironmentSettings
from centipede.environment.simulation import PhysicsSimulation
from centipede.settings_section import SettingsError

# The step value's four checks, whose means over its last turn a segment
# sees with held curves.
STEP_VALUE_CHECKS = tuple(key.removesuffix("_weight") for key in STEP_VALUE_CHECK_KEYS)


class Environment:
    """Batches of centipede worlds, each with a target for the head.

    Every tensor passed in or out has the worlds as its first dimension. When an
    episode ends in a world, ``step`` starts the next one by itself: it resets
    the world, or after an arrival may only give it a new target.

    ``segment_action_sizes`` lists how many actions each segment takes: one
    per leg joint, six (or eight with ankles, model v5), or with leg clocks
    six per leg (seven with an ankle): its clock's tempo and its step shape,
    the sweep and lift amplitudes and each joint's centre (with
    ``segment_tempo``, one tempo for both legs first, then each leg's step
    shape); with spine
    control, the spine joint behind it, for every segment but the rear (only
    for the head with a passive follower spine); and with a clock per
    segment, last, its clock's tempo.
    ``action_size`` is the width of the joint action, the largest of them; a
    segment with fewer actions is padded. ``tempo_columns`` lists each
    segment's tempo actions, and ``motor_columns`` the action that sets each
    of its sweep, lift and knee motors and its spine motor (with leg clocks,
    each leg joint's centre; an ankle is not reported), for the agents'
    learning diagnostics. ``clocks`` holds the clocks, or None without them.
    With ``neighbour_tempos``, ``tempo_gap_ahead_column`` is the observation
    that holds a segment's tempo gap to the segment ahead, in octaves (None
    otherwise), which the agents may read.

    With ``leg_drive = "swing"`` (model A) a segment's actions are its tempo
    action, then an angle for each of its leg joints, left leg first, added
    to a simple swing under each leg's hand; with ``"waves"`` (model B) its
    tempo action, then for each leg joint, left leg first, its curve's
    centre and each wave's cosine and sine amounts. ``shape_columns`` lists,
    for each segment, the actions that draw its legs' curves (model B's
    amounts; empty otherwise), whose starting values the agents may draw at
    random.

    With ``hold_curves`` (cycle 3), each segment decides once per turn. Its
    two legs' clocks share one hand, which starts at a random place when its
    world restarts and plays both legs' curves; each leg's own timing is its
    curve's. On the step the hand starts a turn, the segment takes both legs'
    curves and its tempo action from its actions and holds them for the
    whole turn. The tempo action changes the tempo only then, by as much as
    the heart would over a turn at the tempo it had, so that every turn is
    played at one tempo. Each segment sees, last, the curves it holds and
    the step value's four checks averaged over its last turn. ``action_mask`` ``(W, N,
    action_size)`` marks, for the next step, the actions that will take
    effect: all of a deciding segment's, and otherwise only the spine
    command, which acts every step; the step ignores the rest, so that the
    agents learn only from the actions that acted. None without
    ``hold_curves``, when every action acts.
    """

    def __init__(self, settings: EnvironmentSettings) -> None:
        """Create the simulation and the parts, and allocate the episode state."""
        self.settings = settings
        self.simulation = PhysicsSimulation(
            settings.simulation,
            average_floor_forces=settings.feet_physics,
            measure_leg_power=settings.rewards.energy_cost_parts > 0,
            measure_carrying=settings.rewards.step_value_scale > 0,
        )
        self.world_count = self.simulation.world_count
        self.segment_count = self.simulation.segment_count
        self.device = self.simulation.physical_state.body_height.device
        # Each leg's joints (3, or 4 with an ankle) and each segment's leg
        # motors, both legs'.
        leg_joint_count = len(self.simulation.leg_roles)
        leg_motor_count = self.simulation.leg_motor_count
        self.leg_motor_count = leg_motor_count

        # The radius comes from the configuration and the segment count from
        # the model, so they can only be checked together, here.
        if settings.observation_radius >= self.segment_count:
            raise SettingsError(
                f"[environment] observation_radius must be less than the model's "
                f"{self.segment_count} segments, got {settings.observation_radius}"
            )
        # The model comes from the configuration too: the step shape gives
        # every leg joint a target angle.
        if settings.leg_clocks and not self.simulation.legs_take_angles:
            raise SettingsError(
                "[environment] leg_clocks drive the legs through a step shape of"
                " target angles, which needs a model whose legs take angles, such"
                f" as models/assembly_v4.xml; {settings.simulation.model_path}'s"
                " legs take torques"
            )

        self.clocks = None
        if settings.clocks or settings.leg_clocks:
            self.clocks = Clocks(
                settings.clock,
                self.world_count,
                self.segment_count,
                settings.observation_radius,
                self.simulation.step_duration_s,
                self.device,
                per_leg=settings.leg_clocks,
                leg_joint_count=leg_joint_count,
                shared_hand=settings.hold_curves,
            )
        # With held curves, each segment holds its tempo action and its legs'
        # curves, model B's leg actions, and sees, last, the curves and how
        # its last turn went.
        held_action_count = (
            1 + leg_motor_count * (1 + 2 * settings.waves)
            if settings.hold_curves
            else 0
        )
        turn_value_count = (
            held_action_count - 1 + len(STEP_VALUE_CHECKS)
            if settings.hold_curves
            else 0
        )
        self.observation_builder = ObservationBuilder(
            self.segment_count,
            settings.observation_radius,
            self.world_count,
            self.device,
            spine_observed=settings.spine_control,
            clock_value_count=0 if self.clocks is None else self.clocks.value_count,
            neighbour_clock_observed=settings.neighbour_clocks,
            step_shape_observed=settings.leg_clocks
            and settings.leg_drive == "step_shape",
            leg_acceleration_observed=settings.leg_accelerations,
            leg_joint_count=leg_joint_count,
            feet_physics_observed=settings.feet_physics,
            tilt_observed=settings.tilt,
            neighbour_tempo_observed=settings.neighbour_tempos,
            turn_value_count=turn_value_count,
        )
        self.observation_size = self.observation_builder.observation_size
        # With the neighbours' tempos, observed last but for the turn's
        # values, the first of them is a segment's tempo gap to the segment
        # just ahead, in octaves.
        self.tempo_gap_ahead_column = (
            self.observation_size - 2 * settings.observation_radius - turn_value_count
            if settings.neighbour_tempos
            else None
        )
        # Which segments command the spine joint behind them: with spine
        # control every segment but the rear, or only the head.
        spine_count = self.segment_count - 1
        commanders = 1 if settings.passive_follower_spine else spine_count
        commands_spine = [
            settings.spine_control and segment_index < commanders
            for segment_index in range(self.segment_count)
        ]
        # The leg actions come first, then the spine command, then a clock
        # per segment's tempo. With leg clocks, each leg's actions are its
        # clock's tempo and its step shape: sweep and lift amplitudes, then
        # each joint's centre, sweep, lift, knee (and ankle).
        self.leg_clock_value_count = 3 + leg_joint_count
        segment_tempo = settings.clock.segment_tempo
        self.leg_drive = settings.leg_drive
        # Each leg joint's values with model B: its centre, then a cosine and
        # a sine amount for each wave.
        self.wave_value_count = 1 + 2 * settings.waves
        if self.leg_drive == "swing":
            leg_action_count = 1 + leg_motor_count
        elif self.leg_drive == "waves":
            leg_action_count = 1 + leg_motor_count * self.wave_value_count
        elif settings.leg_clocks:
            leg_action_count = 2 * self.leg_clock_value_count - int(segment_tempo)
        else:
            leg_action_count = leg_motor_count
        self.leg_action_count = leg_action_count
        self.segment_action_sizes = [
            leg_action_count + int(commands) + int(settings.clocks)
            for commands in commands_spine
        ]
        self.action_size = max(self.segment_action_sizes)
        self.spine_column = leg_action_count
        # Each spine command, of the segment ahead of its joint, is kept or
        # replaced by zero; a follower's spine column may hold its tempo.
        self.spine_command_mask = torch.tensor(
            commands_spine[:spine_count], dtype=torch.float32, device=self.device
        )
        # Each segment's tempo actions, and the action that sets each of its
        # sweep, lift and knee motors, for the agents' learning diagnostics.
        self.shape_columns = [[] for _ in commands_spine]
        if self.leg_drive != "step_shape":
            # The tempo first; then each joint's angle, or its curve's values
            # starting with its centre.
            per_joint = 1 if self.leg_drive == "swing" else self.wave_value_count
            self.tempo_columns = [[0] for _ in commands_spine]
            leg_motor_columns = [
                1 + (leg * leg_joint_count + joint) * per_joint
                for leg in range(2)
                for joint in range(3)
            ]
            if self.leg_drive == "waves":
                self.shape_columns = [
                    list(range(1, leg_action_count)) for _ in commands_spine
                ]
        elif settings.leg_clocks:
            per_leg = self.leg_clock_value_count
            if segment_tempo:
                # The tempo, then each leg's step shape: its centres follow
                # its two amplitudes.
                self.tempo_columns = [[0] for _ in commands_spine]
                leg_motor_columns = [
                    1 + leg * (per_leg - 1) + 2 + joint
                    for leg in range(2)
                    for joint in range(3)
                ]
            else:
                self.tempo_columns = [[0, per_leg] for _ in commands_spine]
                leg_motor_columns = [
                    leg * per_leg + 3 + joint for leg in range(2) for joint in range(3)
                ]
        else:
            self.tempo_columns = [
                [size - 1] if settings.clocks else []
                for size in self.segment_action_sizes
            ]
            leg_motor_columns = [
                leg * leg_joint_count + joint for leg in range(2) for joint in range(3)
            ]
        self.motor_columns = [
            leg_motor_columns + [self.spine_column] * commands
            for commands in commands_spine
        ]
        # A clock per segment reads its tempo from the joint action by index.
        self.tempo_index = torch.tensor(
            self.tempo_columns, dtype=torch.long, device=self.device
        ).expand(self.world_count, -1, -1)
        # The joints each segment commands, among its leg joints and the
        # spine joint behind it, last.
        commanded_joints = torch.ones(
            (self.segment_count, leg_motor_count + 1),
            dtype=torch.float32,
            device=self.device,
        )
        commanded_joints[:, -1] = torch.tensor(commands_spine, dtype=torch.float32)
        # Arrival under the head counts progress no closer than half the head's
        # width: within the head's reach, nothing pays more.
        self.head_outline = self.simulation.head_outline
        target = settings.target
        self.segment_weight_n = self.simulation.body_weight_n / self.segment_count
        self.own_weight_n = self.simulation.own_weight_n
        self.reward_function = RewardFunction(
            settings.rewards,
            settings.max_episode_steps,
            self.head_outline.half_width_m
            if target.arrival == "head"
            else target.arrival_radius_m,
            commanded_joints,
            self.segment_weight_n,
        )
        # The step an action of 0 asks for, added to each leg's clock actions:
        # the sweep and lift amplitudes, in action units.
        self.default_step = torch.zeros(
            self.leg_clock_value_count, dtype=torch.float32, device=self.device
        )
        if settings.leg_clocks:
            _, half_width = self.simulation.leg_angle_range
            sweep_half_width, lift_half_width = half_width.reshape(-1)[:2].tolist()
            self.default_step[1] = (
                math.radians(settings.clock.default_swing_deg) / sweep_half_width
            )
            self.default_step[2] = (
                math.radians(settings.clock.default_lift_deg) / lift_half_width
            )
        # Model A's swing under every leg: the default step's sweep and lift
        # amplitudes, around each joint's middle.
        self.swing_amplitudes = self.default_step[1:3].expand(
            self.world_count, self.segment_count, 2, 2
        )
        self.swing_centres = torch.zeros(
            (self.world_count, self.segment_count, 2, leg_joint_count),
            device=self.device,
        )
        self.diagnostics = EnvironmentDiagnostics(
            self.world_count,
            self.segment_count,
            self.reward_function.term_names,
            self.device,
            self.simulation.diagnostics.facts,
            clocks=self.clocks,
            leg_angle_range=(
                self.simulation.leg_angle_range if settings.leg_clocks else None
            ),
            leg_motor_count=leg_motor_count,
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
            (self.world_count, self.segment_count, leg_motor_count + 1),
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
        # Each foot's load before the step, from 0 to 1, for the legs' clocks'
        # load feedback: its contact flag, or its normal force over the
        # weight one segment carries (docs/environment.md, Clocks).
        self.previous_foot_load = torch.zeros(
            (self.world_count, self.segment_count, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.previous_leg_joint_velocity = torch.zeros(
            (self.world_count, self.segment_count, leg_motor_count),
            dtype=torch.float32,
            device=self.device,
        )
        # Each leg joint's acceleration over the last step, rad/s²: the change
        # of its speed divided by the step's duration; zero after a restart,
        # which has no step behind it.
        self.leg_joint_acceleration = torch.zeros(
            (self.world_count, self.segment_count, leg_motor_count),
            dtype=torch.float32,
            device=self.device,
        )
        # The radius of each world's range circle, around its target; infinite
        # when there is no circle.
        self.range_radius = torch.full(
            (self.world_count,), torch.inf, dtype=torch.float32, device=self.device
        )
        # With the feet's physics: each foot's floor force over its segment's
        # own weight on the last step, and both feet's together averaged over
        # each segment's last stride (an exponential average whose time
        # constant is one turn of its clock, or 0.5 s without clocks).
        self.foot_load = torch.zeros(
            (self.world_count, self.segment_count, 2), device=self.device
        )
        self.stride_support = torch.ones(
            (self.world_count, self.segment_count), device=self.device
        )
        self.foot_slip_speed = torch.zeros(
            (self.world_count, self.segment_count, 2), device=self.device
        )
        self.action_mask = None
        if settings.hold_curves:
            # The actions each segment holds, and which segments start a turn
            # on the next step: every one, after a restart.
            self.held_actions = torch.zeros(
                (self.world_count, self.segment_count, held_action_count),
                device=self.device,
            )
            # The step value's checks summed over each segment's turn so far,
            # its steps, and their mean over its last turn; the selection
            # takes each check's reward part back to segment weights × mm, 0
            # for a check left out.
            check_count = len(STEP_VALUE_CHECKS)
            self.turn_check_sums = torch.zeros(
                (self.world_count, self.segment_count, check_count),
                device=self.device,
            )
            self.turn_steps = torch.zeros(
                (self.world_count, self.segment_count, 1), device=self.device
            )
            self.last_turn_result = torch.zeros_like(self.turn_check_sums)
            term_names = self.reward_function.term_names
            weights = self.reward_function.weights.per_step
            self.check_selection = torch.zeros(
                (len(term_names), check_count), device=self.device
            )
            for check_index, check in enumerate(STEP_VALUE_CHECKS):
                if check in term_names:
                    self.check_selection[term_names.index(check), check_index] = (
                        1 / weights[check]
                    )
            self.deciding = torch.ones(
                (self.world_count, self.segment_count),
                dtype=torch.bool,
                device=self.device,
            )
            # The actions a turn holds, the tempo and the legs' curves, and
            # those that act on every step, the spine command after them.
            self.held_columns = torch.zeros(self.action_size, device=self.device)
            self.held_columns[:held_action_count] = 1.0
            self.always_acting = 1 - self.held_columns
            self.action_mask = torch.ones(
                (self.world_count, self.segment_count, self.action_size),
                device=self.device,
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

    def set_curriculum_level(self, level: float) -> None:
        """Harden the health costs to the curriculum's ``level``, 0 to 1; the
        experiment's curriculum calls this between update cycles."""
        self.reward_function.set_level(level)

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
        self.leg_joint_acceleration.zero_()
        self._start_feet(every_world)
        if self.settings.hold_curves:
            self._restart_held_curves(every_world)
        state = self.simulation.physical_state
        self.diagnostics.start_episodes(
            every_world, state, self.target_position, self.range_radius
        )
        return self._observations()

    def step(
        self, joint_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the ``(W, N, action_size)`` joint action for one 20 ms step.

        Returns the observations, the rewards ``(W, N, 2)``, each segment's
        for its walk and for the task (``StepRewards.channel_rewards``),
        terminated and truncated ``(W,)``, and the final observations. Worlds
        whose episode ended have already started a new one: their
        observations start the new episode, and the final observations of
        those restarted are the ones the old episode ended with; a body that
        walks on after an arrival has the same final observation as its next.
        """
        state = self.simulation.physical_state

        # 1. Copy the positions before the step: the physical state is
        # overwritten in place by the step.
        self.previous_body_planar_position.copy_(state.body_planar_position)
        self.previous_head_tip_position.copy_(state.head_tip_position[:, :2])
        self.previous_joint_position.copy_(joint_angles(state))
        self.previous_foot_planar_position.copy_(state.foot_planar_position)
        self.previous_foot_contact.copy_(_foot_contact(state))
        if self.settings.clock.load_feedback_source == "force":
            torch.div(
                state.foot_normal_force,
                self.segment_weight_n,
                out=self.previous_foot_load,
            ).clamp_(max=1.0)
        else:
            self.previous_foot_load.copy_(self.previous_foot_contact)
        self.previous_leg_joint_velocity.copy_(state.leg_joint_velocity)

        # 2. Move the body. With leg clocks, each leg's clock in use, its
        # tempo and step shape, first follows its actions, the clocks turn at
        # their tempos in use, with load feedback from the feet's loads before
        # the step, and each leg's joints get their targets from its step shape at
        # its clock's new phase.
        # With spine control, the spine column holds the command for the
        # spine joint behind each segment that commands one; without, the
        # spine is passive.
        clock_step = None
        leg_amplitudes = None
        leg_phase = None
        leg_actions_end = None
        if self.leg_drive != "step_shape":
            # Models A and B: the tempo action moves the heart, the hands turn,
            # and each leg's targets follow its curve from the hand's place
            # before the step to its place after it. With held curves, the
            # actions held, the heart moved only as a turn starts, and the
            # segments whose turn hands start a turn decide next.
            if self.settings.hold_curves:
                drive_action = self._hold_curves(joint_action)
                tempo = self._turn_tempo_action(drive_action[..., :1])
            else:
                drive_action = joint_action
                tempo = drive_action[..., :1]
            tempo = tempo.expand(-1, -1, 2)
            clock_actions = torch.zeros(
                (*tempo.shape, self.leg_clock_value_count), device=self.device
            )
            clock_actions[..., 0] = tempo
            self.clocks.follow_actions(clock_actions)
            phase_before = self.clocks.clock_phase.clone()
            clock_step = self.clocks.step(
                foot_load=self.previous_foot_load, foot_down=self.previous_foot_contact
            )
            leg_phase, leg_amplitudes = self._leg_rhythm(drive_action)
            leg_actions = self._drive_targets(drive_action, phase_before)
            leg_actions_end = self._drive_targets(drive_action, self.clocks.clock_phase)
            if self.settings.hold_curves:
                # The segment's hand, its legs' one hand, started a new turn.
                torch.lt(self.clocks.phase, phase_before[..., 0], out=self.deciding)
        elif self.settings.leg_clocks:
            self.clocks.follow_actions(
                self._leg_clock_actions(joint_action) + self.default_step
            )
            clock_step = self.clocks.step(
                foot_load=self.previous_foot_load, foot_down=self.previous_foot_contact
            )
            leg_amplitudes = self.clocks.amplitudes
            leg_actions = leg_targets(
                self.clocks.clock_phase, leg_amplitudes, self.clocks.centres
            )
        else:
            leg_actions = joint_action[..., : self.leg_motor_count]
        spine_actions = None
        if self.settings.spine_control:
            spine_actions = (
                joint_action[:, :-1, self.spine_column] * self.spine_command_mask
            )
        self.simulation.step(leg_actions, spine_actions, leg_actions_end)
        if leg_actions_end is not None:
            leg_actions = leg_actions_end
        self.episode_steps += 1
        torch.sub(
            state.leg_joint_velocity,
            self.previous_leg_joint_velocity,
            out=self.leg_joint_acceleration,
        ).div_(self.simulation.step_duration_s)

        # A clock per segment turns at the tempo its segment chose, and the
        # feet that touched the ground before and after the step are measured
        # for slip.
        if self.settings.clocks:
            clock_step = self.clocks.step(
                joint_action.gather(2, self.tempo_index),
                self.previous_joint_position[..., : self.leg_motor_count],
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
        self.foot_slip_speed.copy_(foot_slip_speed)
        foot_lift_load = None
        if self.settings.feet_physics:
            # The load each foot that left the ground bore on the step before,
            # and with the look-ahead the load still on the feet ahead.
            lift_load = self.foot_load
            if self.settings.rewards.foot_hold_looks_ahead:
                lift_load = lift_load + _load_ahead(
                    self.foot_load, self.settings.observation_radius
                )
            foot_lift_load = lift_load * (
                self.previous_foot_contact & ~_foot_contact(state)
            )
            torch.div(
                state.foot_floor_force,
                self.own_weight_n[:, None],
                out=self.foot_load,
            )
            # One turn of each segment's first clock, or 0.5 s without clocks.
            tempo_hz = self.clocks.tempo_hz if self.clocks is not None else 2.0
            turns = torch.as_tensor(tempo_hz * self.simulation.step_duration_s)
            self.stride_support.lerp_(self.foot_load.sum(dim=-1), 1 - torch.exp(-turns))

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

        # 4. and 5. Rewards, observations, diagnostics. The motor commands are
        # the leg actions and the spine column, which the reward masks for the
        # segments that command no spine joint.
        spine_command = joint_action[..., self.spine_column : self.spine_column + 1]
        step_rewards = self.reward_function.compute(
            state,
            self.previous_body_planar_position,
            self.previous_head_tip_position,
            self.target_position,
            terminated,
            self.previous_joint_position,
            torch.cat((leg_actions, spine_command), dim=-1),
            foot_slip_speed,
            clock_step,
            self.foot_load if self.settings.feet_physics else None,
            self.stride_support if self.settings.feet_physics else None,
            foot_lift_load,
        )
        if self.settings.hold_curves:
            self._add_to_turn(step_rewards.reward_parts)
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
            leg_amplitudes,
            leg_phase,
        )

        # 6. Start a new episode in the worlds whose episode ended: restart
        # them, except, with after_arrival = "new_target", those that arrived,
        # whose body walks on toward a new target. ``build`` returns a new
        # tensor, so the final observations keep the values from before; a
        # body that walks on is not restarted, and its final observation is
        # what it sees next, the new target.
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
            self.leg_joint_acceleration.masked_fill_(restarted[:, None, None], 0.0)
            self._start_feet(restarted)
            if self.settings.hold_curves:
                self._restart_held_curves(restarted)
            self._place_targets(ended)
            self.episode_steps.masked_fill_(ended, 0)
            self.diagnostics.start_episodes(
                ended, state, self.target_position, self.range_radius
            )
            observations = self._observations()
            final_observations = torch.where(
                restarted[:, None, None], final_observations, observations
            )
        if self.settings.hold_curves:
            self._update_action_mask()

        return (
            observations,
            step_rewards.channel_rewards,
            terminated,
            truncated,
            final_observations,
        )

    def _hold_curves(self, joint_action: torch.Tensor) -> torch.Tensor:
        """The joint action with each segment's held actions, its tempo and
        its legs' curves, in place of its own, after the deciding segments
        took theirs from it."""
        held = self.held_actions
        held.copy_(
            torch.where(
                self.deciding[..., None], joint_action[..., : held.shape[-1]], held
            )
        )
        drive_action = joint_action.clone()
        drive_action[..., : held.shape[-1]] = held
        return drive_action

    def _turn_tempo_action(self, tempo_action: torch.Tensor) -> torch.Tensor:
        """The heart's action for this step, ``(W, N, 1)``: 0 inside a turn,
        and as a turn starts the held action times the steps of a turn at the
        tempo it had, so that the heart moves at once as far as it would over
        that turn."""
        steps_per_turn = 1 / (self.clocks.tempo_hz * self.simulation.step_duration_s)
        return torch.where(
            self.deciding[..., None], tempo_action * steps_per_turn[..., None], 0.0
        )

    def _add_to_turn(self, reward_parts: torch.Tensor) -> None:
        """Add the step's checks to each segment's turn; where the turn just
        ended, keep their means over it as its last turn's result and start
        the next. The step on which the hand comes round belongs to the turn
        it ends, which its curves played."""
        self.turn_check_sums.add_(reward_parts @ self.check_selection)
        self.turn_steps.add_(1.0)
        ended = self.deciding[..., None]
        self.last_turn_result.copy_(
            torch.where(
                ended, self.turn_check_sums / self.turn_steps, self.last_turn_result
            )
        )
        self.turn_check_sums.masked_fill_(ended, 0.0)
        self.turn_steps.masked_fill_(ended, 0.0)

    def _restart_held_curves(self, world_mask: torch.Tensor) -> None:
        """The masked worlds hold nothing and know no last turn, and every
        segment starts a turn on the next step."""
        mask = world_mask[:, None, None]
        self.held_actions.masked_fill_(mask, 0.0)
        self.turn_check_sums.masked_fill_(mask, 0.0)
        self.turn_steps.masked_fill_(mask, 0.0)
        self.last_turn_result.masked_fill_(mask, 0.0)
        self.deciding.logical_or_(world_mask[:, None])
        self._update_action_mask()

    def _update_action_mask(self) -> None:
        """``action_mask`` for the next step, from the deciding segments."""
        torch.addcmul(
            self.always_acting,
            self.deciding[..., None].float(),
            self.held_columns,
            out=self.action_mask,
        )

    def _drive_targets(
        self, joint_action: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        """Every leg joint's target, ``(W, N, M)`` in action units, with model A
        or B, each leg's hand at ``phase`` ``(W, N, 2)``.

        Model A: the swing under the leg (``leg_targets`` with the default
        step's amplitudes) plus the segment's angle for the joint. Model B:
        each joint's curve at the hand, its centre plus, for each wave k, its
        cosine amount times cos kφ and its sine amount times sin kφ.
        """
        joint_count = len(self.simulation.leg_roles)
        values = joint_action[..., 1 : self.leg_action_count]
        if self.leg_drive == "swing":
            swing = leg_targets(phase, self.swing_amplitudes, self.swing_centres)
            return swing + values
        amounts = values.unflatten(-1, (2, joint_count, self.wave_value_count))
        wave_count = (self.wave_value_count - 1) // 2
        harmonics = torch.arange(1, wave_count + 1, device=self.device)
        angle = phase[..., None, None] * harmonics  # (W, N, 2, 1, K)
        cosine_amounts = amounts[..., 1::2]
        sine_amounts = amounts[..., 2::2]
        curve = amounts[..., 0] + (
            cosine_amounts * torch.cos(angle) + sine_amounts * torch.sin(angle)
        ).sum(dim=-1)
        return curve.flatten(start_dim=2)

    def _leg_rhythm(
        self, joint_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Where each leg is in its step and how far it swings, for the
        diagnostics, with model A or B: ``(W, N, 2)`` phases and ``(W, N, 2,
        2)`` sweep and lift amplitudes in action units.

        Model A's legs swing under their hands, with the swing's amplitudes.
        Model B's sweep follows its first wave, a cos φ + b sin φ =
        R cos(φ − atan2(b, a)): the leg's step lags its hand by atan2(b, a),
        and its sweep and lift amplitudes are their first waves' R.
        """
        if self.leg_drive == "swing":
            return self.clocks.clock_phase, self.swing_amplitudes
        joint_count = len(self.simulation.leg_roles)
        amounts = joint_action[..., 1 : self.leg_action_count].unflatten(
            -1, (2, joint_count, self.wave_value_count)
        )
        first_wave = amounts[..., :2, 1:3]  # (W, N, 2 legs, sweep and lift, cos, sin)
        sweep_lag = torch.atan2(first_wave[..., 0, 1], first_wave[..., 0, 0])
        return self.clocks.clock_phase - sweep_lag, first_wave.norm(dim=-1)

    def _start_feet(self, world_mask: torch.Tensor) -> None:
        """Start the masked worlds' feet's loads at the starting pose's and
        their stride support at those loads, at most 1."""
        if not self.settings.feet_physics:
            return
        state = self.simulation.physical_state
        load = state.foot_floor_force / self.own_weight_n[:, None]
        mask = world_mask[:, None, None]
        self.foot_load.copy_(torch.where(mask, load, self.foot_load))
        self.stride_support.copy_(
            torch.where(
                mask[..., 0], load.sum(dim=-1).clamp(max=1.0), self.stride_support
            )
        )
        self.foot_slip_speed.masked_fill_(mask, 0.0)

    def _leg_clock_actions(self, joint_action: torch.Tensor) -> torch.Tensor:
        """Each leg's clock actions, ``(W, N, 2, 3 + J)``: its tempo, then
        its step shape. With ``segment_tempo``, both legs take the segment's
        one tempo action."""
        leg_columns = joint_action[..., : self.leg_action_count]
        if not self.settings.clock.segment_tempo:
            return leg_columns.unflatten(-1, (2, self.leg_clock_value_count))
        tempo = leg_columns[..., None, :1].expand(-1, -1, 2, -1)
        shapes = leg_columns[..., 1:].unflatten(-1, (2, self.leg_clock_value_count - 1))
        return torch.cat((tempo, shapes), dim=-1)

    def _observations(self) -> torch.Tensor:
        """Every segment's observation of the current state."""
        clocks = self.clocks
        return self.observation_builder.build(
            self.simulation.physical_state,
            self.target_position,
            None if clocks is None else clocks.observation_values(),
            clocks.neighbour_values() if self.settings.neighbour_clocks else None,
            clocks.step_shape if self.settings.leg_clocks else None,
            self.leg_joint_acceleration,
            self._feet_values() if self.settings.feet_physics else None,
            clocks.neighbour_tempos() if self.settings.neighbour_tempos else None,
            torch.cat((self.held_actions[..., 1:], self.last_turn_result), dim=-1)
            if self.settings.hold_curves
            else None,
        )

    def _feet_values(self) -> torch.Tensor:
        """Each segment's feet as it feels them, ``(W, N, 5)``: each foot's load
        (its floor force over the segment's own weight), each foot's sliding
        speed while planted in grip units, and its support over the stride."""
        return torch.cat(
            (
                self.foot_load,
                self.foot_slip_speed / self.settings.rewards.grip_unit_m_per_s,
                self.stride_support[..., None],
            ),
            dim=-1,
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


def _load_ahead(foot_load: torch.Tensor, radius: int) -> torch.Tensor:
    """For each foot, the loads ``(W, N, 2)`` of the same-side feet up to
    ``radius`` segments ahead, each divided by its distance; 0 for the head.
    The nearest foot ahead counts fully, as the follow nudge read it; farther
    ones warn of the wave coming."""
    ahead = torch.zeros_like(foot_load)
    for distance in range(1, radius + 1):
        ahead[:, distance:] += foot_load[:, :-distance] / distance
    return ahead
