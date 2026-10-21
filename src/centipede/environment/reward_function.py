"""The follow-line reward: two accounts for every unit, the team's and its
own, each judged by its own critic, or one account, the team's, holding
every part (docs/follow-line/reward.md)."""

import torch

from centipede.environment.settings import RewardSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.constants import ACTION_DURATION_S

# The reward's parts, signed, each in its own unit: the line, the squeeze, the
# motion, the accelerations, the energy's share, the force share and the carry
# in millimetres, the rest in millijoules. The team account adds the first
# TEAM_PART_COUNT, each unit's own account the rest.
PART_NAMES = (
    "line",
    "squeeze",
    "pushing",
    "braking",
    "force",
    "motion",
    "acceleration",
    "energy_share",
    "force_share",
    "pain",
    "hanging",
    "carry",
)
TEAM_PART_COUNT = 8
MILLIJOULE_PARTS = ("pushing", "braking", "force", "pain", "hanging")
CHANNEL_NAMES = ("team", "own")
# Energy's parts, each unit's own before the team account takes their mean.
ENERGY_PARTS = ("pushing", "braking", "force")


class AveragedForces:
    """The forces on each unit that the reward prices, averaged through time,
    ``values`` ``(W, N, 6)`` in unit weights: the net force of its two joints
    on it along the body and up, the front joint's force along the body
    (positive pulls it forward), the floor's push on its belly, the force
    with which its legs press on other legs, and the floor's push on its two
    feet. Each step moves the average ``averaging_rate`` of the way to the
    step's value."""

    def __init__(
        self,
        world_count: int,
        segment_count: int,
        unit_weight_n: float,
        averaging_rate: float,
        device: torch.device | str,
    ) -> None:
        self.unit_weight_n = unit_weight_n
        self.averaging_rate = averaging_rate
        self.values = torch.zeros((world_count, segment_count, 6), device=device)

    def _current(self, state: PhysicalState) -> torch.Tensor:
        return (
            torch.cat(
                (
                    state.net_joint_force,
                    state.front_joint_force[..., None],
                    state.belly_normal_force[..., None],
                    state.leg_leg_force[..., None],
                    state.foot_normal_force.sum(dim=-1, keepdim=True),
                ),
                dim=-1,
            )
            / self.unit_weight_n
        )

    def restart(self, world_mask: torch.Tensor, state: PhysicalState) -> None:
        """Start the masked worlds' averages at the current forces."""
        mask = world_mask[:, None, None]
        self.values.copy_(torch.where(mask, self._current(state), self.values))

    def update(self, state: PhysicalState) -> torch.Tensor:
        """Move the averages toward the step's forces and return them."""
        self.values += self.averaging_rate * (self._current(state) - self.values)
        return self.values


class RewardFunction:
    """Every unit's team and own rewards for the step just taken.

    ``commanded_joints`` ``(N, M + 1)`` is 1.0 for the motors each unit
    commands: its leg motors and the spine motor behind it. The line is paid
    within ``band_m`` of it. ``ramp_weight`` weighs energy, the squeeze,
    pain, the motion and the accelerations; the experiment sets it before
    each update cycle.
    """

    def __init__(
        self,
        settings: RewardSettings,
        body: BodyMeasures,
        commanded_joints: torch.Tensor,
        band_m: float,
    ) -> None:
        """Fix the prices once."""
        self.settings = settings
        self.commanded_joints = commanded_joints
        device = commanded_joints.device
        self.ramp_weight = 1.0
        self.band_mm = band_m * 1e3
        self.line_cap_mm = settings.line_speed_cap_mm_per_s * ACTION_DURATION_S
        # Millimetres a squared force cost takes through a step at its scale.
        self.reference_mm = settings.reference_pay_mm_per_s * ACTION_DURATION_S
        # Millijoules for one motor fully commanded through a step.
        self.force_price_mj = settings.force_price_w * ACTION_DURATION_S * 1e3
        self.pain_price_mj = (
            self.force_price_mj * settings.pain_price / body.leg_push_n**2
        )
        # Millijoules for a unit weight hanging through a step.
        self.hanging_price_mj = self.force_price_mj * settings.hanging_price
        # Millimetres for a step of joints turning at 1 rad/s squared, and of
        # joints changing their speed by 1 rad/s a second squared.
        self.motion_price_mm = settings.motion_price * ACTION_DURATION_S
        self.acceleration_price_mm = settings.acceleration_price * ACTION_DURATION_S
        # Millimetres for a step of a unit weight its feet do not carry.
        self.carry_price_mm = settings.carry_price * ACTION_DURATION_S
        self.unit_weight_n = body.unit_weight_n
        # Each unit may take from its neighbours, for free, the weight it
        # carries beyond the others' mean, such as a head; none on the
        # headless tester, whose units are alike. Its feet carry the rest.
        weights = torch.tensor(body.segment_weights_n)
        others = (weights.sum() - weights) / (len(weights) - 1)
        self.free_support = ((weights - others) / body.unit_weight_n).clamp(min=0.0)
        self.free_support = self.free_support.to(device)
        self.own_carry = weights.to(device) / body.unit_weight_n - self.free_support
        # Each part's rate into each account, a millijoule counted as
        # mm_per_mj millimetres.
        self.part_channels = torch.zeros(len(PART_NAMES), len(CHANNEL_NAMES))
        for index, name in enumerate(PART_NAMES):
            rate = settings.mm_per_mj if name in MILLIJOULE_PARTS else 1.0
            own = index >= TEAM_PART_COUNT and settings.own_account
            self.part_channels[index, int(own)] = rate
        self.part_channels = self.part_channels.to(device)

    def squared_accelerations(self, joint_acceleration: torch.Tensor) -> torch.Tensor:
        """Each unit's joints' changes of speed ``(W, N, M + 1)`` rad/s²,
        squared and added up over the joints it commands, ``(W, N)``."""
        return (joint_acceleration.square() * self.commanded_joints).sum(dim=-1)

    def energy(
        self, physical_state: PhysicalState, motor_command: torch.Tensor
    ) -> torch.Tensor:
        """Each unit's energy over the step just taken, ``(W, N, 3)`` mJ in
        ENERGY_PARTS' order, priced: its motors' pushing work, their braking
        work at ``braking_price`` and their squared commands at
        ``force_price_w``, from ``motor_command`` ``(W, N, M + 1)``."""
        squared_commands = (motor_command.square() * self.commanded_joints).sum(dim=-1)
        return torch.stack(
            (
                physical_state.motor_positive_work * 1e3,
                physical_state.motor_negative_work * 1e3 * self.settings.braking_price,
                squared_commands * self.force_price_mj,
            ),
            dim=-1,
        )

    def compute(
        self,
        physical_state: PhysicalState,
        motor_command: torch.Tensor,
        progress: torch.Tensor,
        averaged_speed: torch.Tensor,
        distance: torch.Tensor,
        angle: torch.Tensor,
        averaged_forces: torch.Tensor,
        tow_pushes: torch.Tensor,
        joint_acceleration: torch.Tensor,
        averaged_accelerations: torch.Tensor,
        averaged_energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rewards ``(W, N, 2)``, team and own, their parts ``(W, N, 12)``
        in PART_NAMES' order, and each unit's own energy ``(W, N, 3)`` mJ.

        ``motor_command`` ``(W, N, M + 1)`` is what each motor received;
        ``progress``, ``distance`` and ``angle`` ``(W, N)`` are each unit's
        step along the line (m), distance from it (m) and angle from its
        direction (``StraightLine``); ``averaged_speed`` ``(W, N)`` its speed
        along the line averaged as the forces are (m/s); ``averaged_forces`` is
        ``AveragedForces.values``; ``tow_pushes`` ``(W, N)`` each unit's tow
        on the step, in unit weights; ``joint_acceleration`` ``(W, N, M + 1)``
        how fast each unit's joints (its legs' and the spine joint behind it)
        changed their speed over the step, rad/s², and
        ``averaged_accelerations`` ``(W, N)`` their ``squared_accelerations``
        averaged as the forces are. The motion: each unit's joints turning,
        their speeds squared and added up; the accelerations, the same of
        their changes of speed; the team is charged the units' mean of each.
        With ``acceleration_halving`` the accelerations' part is instead the
        share of the team's forward line pay they take, from the units' mean
        of the averaged accelerations: the forward pay times J / (halving +
        J), so that the team keeps the forward pay times halving / (halving
        + J), and a step back in full. With ``time_value_mj_per_s`` h the
        energy's share takes the same from the units' mean E of
        ``averaged_energy`` ``(W, N)``, each unit's ``energy`` averaged as the
        forces are, in mJ a second: the forward pay times E / (h + E).

        The line: each unit's step forward, up to the cap (or, with
        ``line_cap_averaged``, scaled by the cap over its averaged speed when
        that is above it, less ``line_overspeed_slope`` times the speed above
        it), times its closeness, 1 - (distance / band)², and
        the cosine of its angle (none beyond the band or facing away), and a
        step back in full; the team is paid the units' mean. The squeeze: at
        each joint, the averaged force along the body with the tows behind it
        taken out, a pull free, squared against the force scale and averaged
        over the joints. Energy: each unit's pushing work, braking work and
        squared commands, the team charged the units' mean. Each unit's own:
        its force share, the
        averaged net force of its joints along the body plus its tow,
        squared against the force scale; its hanging, the weight its
        neighbours hold up for it beyond its free support and its belly's
        load; its pain, its legs' averaged press on other legs, squared; and
        its carry, the weight beyond its free support that its feet, on
        average, do not carry.
        """
        state = physical_state
        ramp = self.ramp_weight
        scale = self.settings.joint_force_scale

        step_mm = progress * 1e3
        closeness = (1 - (distance * 1e3 / self.band_mm) ** 2).clamp(min=0.0)
        facing = angle.cos().clamp(min=0.0)
        if self.settings.line_cap_averaged:
            cap = self.settings.line_speed_cap_mm_per_s * 1e-3
            speed = averaged_speed.clamp(min=cap)
            target_pay = cap - self.settings.line_overspeed_slope * (speed - cap)
            paid = step_mm.clamp(min=0.0) * target_pay / speed
        else:
            paid = step_mm.clamp(min=0.0, max=self.line_cap_mm)
        forward = paid * closeness * facing
        line = forward + step_mm.clamp(max=0.0)

        along, up, front, belly, pressed, feet = averaged_forces.unbind(dim=-1)
        legs_push = front + tow_pushes.flip(-1).cumsum(dim=-1).flip(-1)
        pushed = (legs_push[:, 1:].clamp(max=0.0) / scale) ** 2
        squeeze = self.reference_mm * ramp * pushed.mean(dim=-1, keepdim=True)

        own_energy = self.energy(state, motor_command)

        # The rear unit has no spine joint behind it.
        squared_speeds = state.leg_joint_velocity.square().sum(dim=-1)
        squared_speeds += (
            state.spine_yaw_velocity.square() * self.commanded_joints[:, -1]
        )
        motion = self.motion_price_mm * ramp * squared_speeds.mean(dim=1, keepdim=True)
        halving = self.settings.acceleration_halving
        if halving:
            # The share of the forward pay the accelerations take: 1 - the
            # factor 1 / (1 + J / halving).
            jerk = ramp * averaged_accelerations.mean(dim=1, keepdim=True) / halving
            acceleration = forward.mean(dim=1, keepdim=True) * jerk / (1 + jerk)
        else:
            accelerations = self.squared_accelerations(joint_acceleration)
            acceleration = (
                self.acceleration_price_mm
                * ramp
                * accelerations.mean(dim=1, keepdim=True)
            )
        energy_share = torch.zeros_like(motion)
        if self.settings.time_value_mj_per_s:
            # The share of the forward pay the energy takes, as the
            # accelerations' above: 1 - h / (h + E).
            spent = (
                ramp
                * averaged_energy.mean(dim=1, keepdim=True)
                / self.settings.time_value_mj_per_s
            )
            energy_share = forward.mean(dim=1, keepdim=True) * spent / (1 + spent)

        force_share = self.reference_mm * ((along + tow_pushes) / scale) ** 2
        pain = (pressed * self.unit_weight_n) ** 2 * self.pain_price_mj * ramp
        hanging = (up - self.free_support).clamp(min=0.0) + belly
        carry = self.carry_price_mm * (self.own_carry - feet).clamp(min=0.0)

        team = torch.cat(
            (
                line.mean(dim=1, keepdim=True)[..., None],
                -squeeze[..., None],
                -ramp * own_energy.mean(dim=1, keepdim=True),
                -motion[..., None],
                -acceleration[..., None],
                -energy_share[..., None],
            ),
            dim=-1,
        )
        own = torch.stack(
            (-force_share, -pain, -self.hanging_price_mj * hanging, -carry), dim=-1
        )
        parts = torch.cat((team.expand(-1, line.shape[1], -1), own), dim=-1)
        return parts @ self.part_channels, parts, own_energy
