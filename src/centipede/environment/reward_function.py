"""The follow-line reward: two accounts for every unit, the team's and its
own, each judged by its own critic (docs/follow-line/reward.md)."""

import torch

from centipede.environment.settings import RewardSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.constants import ACTION_DURATION_S

# The reward's parts, signed, each in its own unit: the line, the squeeze, the
# motion and the force share in millimetres, the rest in millijoules. The team
# account adds the first TEAM_PART_COUNT, each unit's own account the rest.
PART_NAMES = (
    "line",
    "squeeze",
    "pushing",
    "braking",
    "force",
    "motion",
    "force_share",
    "pain",
    "hanging",
)
TEAM_PART_COUNT = 6
MILLIJOULE_PARTS = ("pushing", "braking", "force", "pain", "hanging")
CHANNEL_NAMES = ("team", "own")
# Energy's parts, each unit's own before the team account takes their mean.
ENERGY_PARTS = ("pushing", "braking", "force")


class AveragedForces:
    """The forces on each unit that the reward prices, averaged through time,
    ``values`` ``(W, N, 5)`` in unit weights: the net force of its two joints
    on it along the body and up, the front joint's force along the body
    (positive pulls it forward), the floor's push on its belly, and the force
    with which its legs press on other legs. Each step moves the average
    ``averaging_rate`` of the way to the step's value."""

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
        self.values = torch.zeros((world_count, segment_count, 5), device=device)

    def _current(self, state: PhysicalState) -> torch.Tensor:
        return (
            torch.cat(
                (
                    state.net_joint_force,
                    state.front_joint_force[..., None],
                    state.belly_normal_force[..., None],
                    state.leg_leg_force[..., None],
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
    within ``band_m`` of it. ``ramp_weight`` weighs energy, the squeeze and
    pain; the experiment sets it before each update cycle.
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
        # Millimetres for a step of joints turning at 1 rad/s squared.
        self.motion_price_mm = settings.motion_price * ACTION_DURATION_S
        self.unit_weight_n = body.unit_weight_n
        # Each unit may take from its neighbours, for free, the weight it
        # carries beyond the others' mean, such as a head; none on the
        # headless tester, whose units are alike.
        weights = torch.tensor(body.segment_weights_n)
        others = (weights.sum() - weights) / (len(weights) - 1)
        self.free_support = ((weights - others) / body.unit_weight_n).clamp(min=0.0)
        self.free_support = self.free_support.to(device)
        # Each part's rate into each account, a millijoule counted as
        # mm_per_mj millimetres.
        self.part_channels = torch.zeros(len(PART_NAMES), len(CHANNEL_NAMES))
        for index, name in enumerate(PART_NAMES):
            rate = settings.mm_per_mj if name in MILLIJOULE_PARTS else 1.0
            self.part_channels[index, int(index >= TEAM_PART_COUNT)] = rate
        self.part_channels = self.part_channels.to(device)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rewards ``(W, N, 2)``, team and own, their parts ``(W, N, 9)``
        in PART_NAMES' order, and each unit's own energy ``(W, N, 3)`` mJ.

        ``motor_command`` ``(W, N, M + 1)`` is what each motor received;
        ``progress``, ``distance`` and ``angle`` ``(W, N)`` are each unit's
        step along the line (m), distance from it (m) and angle from its
        direction (``StraightLine``); ``averaged_speed`` ``(W, N)`` its speed
        along the line averaged as the forces are (m/s); ``averaged_forces`` is
        ``AveragedForces.values``; ``tow_pushes`` ``(W, N)`` each unit's tow
        on the step, in unit weights. The motion: each unit's joints (its legs'
        and the spine joint behind it) turning, their speeds squared and added
        up, the team charged the units' mean.

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
        load; and its pain, its legs' averaged press on other legs, squared.
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

        along, up, front, belly, pressed = averaged_forces.unbind(dim=-1)
        legs_push = front + tow_pushes.flip(-1).cumsum(dim=-1).flip(-1)
        pushed = (legs_push[:, 1:].clamp(max=0.0) / scale) ** 2
        squeeze = self.reference_mm * ramp * pushed.mean(dim=-1, keepdim=True)

        squared_commands = (motor_command.square() * self.commanded_joints).sum(dim=-1)
        own_energy = torch.stack(
            (
                state.motor_positive_work * 1e3,
                state.motor_negative_work * 1e3 * self.settings.braking_price,
                squared_commands * self.force_price_mj,
            ),
            dim=-1,
        )

        # The rear unit has no spine joint behind it.
        squared_speeds = state.leg_joint_velocity.square().sum(dim=-1)
        squared_speeds += (
            state.spine_yaw_velocity.square() * self.commanded_joints[:, -1]
        )
        motion = self.motion_price_mm * ramp * squared_speeds.mean(dim=1, keepdim=True)

        force_share = self.reference_mm * ((along + tow_pushes) / scale) ** 2
        pain = (pressed * self.unit_weight_n) ** 2 * self.pain_price_mj * ramp
        hanging = (up - self.free_support).clamp(min=0.0) + belly

        team = torch.cat(
            (
                line.mean(dim=1, keepdim=True)[..., None],
                -squeeze[..., None],
                -ramp * own_energy.mean(dim=1, keepdim=True),
                -motion[..., None],
            ),
            dim=-1,
        )
        own = torch.stack(
            (-force_share, -pain, -self.hanging_price_mj * hanging), dim=-1
        )
        parts = torch.cat((team.expand(-1, line.shape[1], -1), own), dim=-1)
        return parts @ self.part_channels, parts, own_energy
