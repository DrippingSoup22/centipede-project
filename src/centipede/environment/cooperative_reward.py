"""The cooperators' reward: two channels for every segment, goal and health,
judged apart by a critic each (docs/cooperators/reward.md, cycles 7 and 10)."""

import torch

from centipede.environment.reward_function import RewardContext, StepRewards
from centipede.environment.settings import CooperatorsSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.constants import ACTION_DURATION_S

# The reward's parts, signed: the goal's in millimetres, health's in
# millijoules. The goal channel adds the first GOAL_PARTS, health the rest.
PART_NAMES = (
    "progress",
    "arrival",
    "line",
    "off_line",
    "front_joint",
    "own_ground",
    "pushing",
    "braking",
    "force",
    "pain",
    "hanging",
)
GOAL_PARTS = 6
CHANNEL_NAMES = ("goal", "health")


def energy_sharing_weights(
    segment_count: int, observation_radius: int, decay: float
) -> torch.Tensor:
    """Each segment's shares of its own and its neighbours' energy, ``(N, N)``
    (the user's rule, cycle 5).

    Every segment counts itself 1 and each side decay + decay² + ... up to
    the observation radius, given to that side's neighbours by distance; a
    missing neighbour's weight goes to the nearest one on its side, and a
    missing side's to the other side. Each row is scaled to add up to one:
    at radius 2 with halving, itself 40% and each side 30% (a middle
    segment's nearest neighbours 20% each, the next 10%); the head 40%,
    segment 1 40% and segment 2 20%.
    """
    weights = torch.eye(segment_count)
    side_weights = [decay**distance for distance in range(1, observation_radius + 1)]
    for i in range(segment_count):
        ahead = [i - d for d in range(1, observation_radius + 1) if i - d >= 0]
        behind = [
            i + d for d in range(1, observation_radius + 1) if i + d < segment_count
        ]
        for side, other in ((ahead, behind), (behind, ahead)):
            neighbours = side or other
            for distance, weight in enumerate(side_weights):
                weights[i, neighbours[min(distance, len(neighbours) - 1)]] += weight
    return weights / weights.sum(dim=1, keepdim=True)


class AveragedForces:
    """The forces on each segment that the reward prices, averaged through
    time, ``values`` ``(W, N, 4)`` in unit weights: the net force its joints
    put on it along the body and up, the front joint's force, and the
    floor's push on its body. Each step moves the average ``averaging_rate``
    of the way to the step's value; 1 keeps the step's."""

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
        self.values = torch.zeros((world_count, segment_count, 4), device=device)

    def _current(self, state: PhysicalState) -> torch.Tensor:
        return (
            torch.cat(
                (
                    state.net_joint_force,
                    state.front_joint_force[..., None],
                    state.belly_normal_force[..., None],
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


class CooperativeReward:
    """Every segment's goal and health for the step just taken."""

    def __init__(
        self,
        settings: CooperatorsSettings,
        segment_count: int,
        observation_radius: int,
        body: BodyMeasures,
        commanded_joints: torch.Tensor,
    ) -> None:
        """Fix the prices and the sharing weights once.

        ``commanded_joints`` is ``(N, M + 1)``: 1.0 for the motors each
        segment commands, its leg motors and the spine motor behind it.
        """
        self.settings = settings
        self.commanded_joints = commanded_joints
        device = commanded_joints.device
        self.sharing_weights = energy_sharing_weights(
            segment_count, observation_radius, settings.energy_sharing_decay
        ).to(device)
        # Millijoules for one motor fully commanded through a step.
        self.force_price_mj = settings.force_price_w * ACTION_DURATION_S * 1e3
        self.pain_price_mj = (
            self.force_price_mj * settings.pain_price / body.leg_push_n**2
        )
        # Millijoules for a unit weight hanging through a step.
        self.hanging_price_mj = self.force_price_mj * settings.hanging_price
        # Millimetres of goal a squared cost takes through a step at its scale.
        self.reference_mm = settings.reference_pay_mm_per_s * ACTION_DURATION_S
        self.is_follower = torch.ones(segment_count, device=device)
        self.is_follower[0] = 0.0
        # The support a segment may take from its neighbours for free.
        self.free_support = torch.zeros(segment_count, device=device)
        self.free_support[0] = settings.head_free_support

    def compute(
        self,
        physical_state: PhysicalState,
        previous_body_planar_position: torch.Tensor,
        previous_head_tip_position: torch.Tensor,
        target_position: torch.Tensor,
        arrived: torch.Tensor,
        arrival_worth_m: float,
        previous_joint_position: torch.Tensor,
        motor_command: torch.Tensor,
        line_erased: torch.Tensor,
        off_line: torch.Tensor,
        averaged_forces: torch.Tensor,
        tow: torch.Tensor,
    ) -> tuple[torch.Tensor, StepRewards, torch.Tensor]:
        """The ``(W, N, 2)`` rewards, goal and health, their parts, and each
        segment's own energy before sharing, ``(W, N, 3)`` in mJ.

        The goal, in millimetres: for the head, how much closer its tip came
        to the target and, when it arrives, ``arrival_worth_m``; for each
        follower, the head's line its tail erased on the step, less, each
        squared against its scale and priced at the reference pay: how far
        its tail lies from the line it has not erased (up to a cap), and, as
        ``force_price`` says, its front joint's averaged force, pulled or
        pushed, or its own net ground force: the averaged net force of its
        joints along its body plus its ``tow`` ``(W, N)``, in unit weights.
        Along the body the joints' net force is its momentum's change less
        the floor's force and the tow, so adding the tow back leaves what
        its own feet and body fail to carry: positive dragged, negative
        pushing (``line_erased`` and ``off_line`` ``(W, N)`` in metres, from
        ``HeadLine.update``; ``averaged_forces``,
        ``AveragedForces.values``). Health, in
        millijoules: each segment's energy, shared, its own pain, and its
        hanging on the neighbours and the floor. ``motor_command`` is
        ``(W, N, M + 1)``, what each motor received; the other arguments are
        those of ``RewardContext.from_step``.
        """
        context = RewardContext.from_step(
            physical_state,
            previous_body_planar_position,
            previous_head_tip_position,
            target_position,
            arrived,
            0.0,
            previous_joint_position,
            self.commanded_joints,
            motor_command,
        )
        settings = self.settings
        progress_m = context.distance_before - context.distance_after
        progress = torch.zeros_like(progress_m)
        progress[:, 0] = progress_m[:, 0] * 1e3
        arrival = torch.zeros_like(progress)
        arrival[:, 0] = context.arrived * arrival_worth_m * 1e3

        # The followers' line: paid as erased, charged as left behind.
        line = self.is_follower * line_erased * 1e3
        off = (off_line * 1e3).clamp(max=settings.line_distance_cap_mm)
        off_cost = self.reference_mm * (off / settings.line_distance_scale_mm) ** 2
        along, up, front, floor = averaged_forces.unbind(dim=-1)
        own_ground = settings.force_price == "own_ground"
        priced = along + tow if own_ground else front
        force_cost = self.reference_mm * (priced / settings.joint_force_scale) ** 2
        joint_cost = force_cost * (not own_ground)
        own_ground_cost = force_cost * own_ground
        hanging = (up - self.free_support).clamp(min=0.0) + floor

        squared_commands = context.command_size * self.commanded_joints.sum(dim=-1)
        own_energy = torch.stack(
            (
                physical_state.motor_positive_work * 1e3,
                physical_state.motor_negative_work * 1e3 * settings.braking_price,
                squared_commands * self.force_price_mj,
            ),
            dim=-1,
        )
        shared_energy = torch.einsum("ij,wjp->wip", self.sharing_weights, own_energy)
        pain = physical_state.leg_leg_force_squared * self.pain_price_mj

        parts = torch.cat(
            (
                torch.stack(
                    (
                        progress,
                        arrival,
                        line,
                        -self.is_follower * off_cost,
                        -self.is_follower * joint_cost,
                        -self.is_follower * own_ground_cost,
                    ),
                    dim=-1,
                ),
                -shared_energy,
                -pain[..., None],
                -(self.hanging_price_mj * hanging)[..., None],
            ),
            dim=-1,
        )
        channels = torch.stack(
            (
                parts[..., :GOAL_PARTS].sum(dim=-1),
                parts[..., GOAL_PARTS:].sum(dim=-1),
            ),
            dim=-1,
        )
        step_rewards = StepRewards(
            rewards=channels.sum(dim=-1),
            reward_parts=parts,
            segment_progress=progress_m,
            joint_movement=context.joint_movement,
        )
        return channels, step_rewards, own_energy
