"""The cooperators' reward: a goal and a cost for every segment, and with the
trail a third channel, judged apart by a critic each
(docs/cooperators/reward.md)."""

import math

import torch

from centipede.environment.reward_function import RewardContext, StepRewards
from centipede.environment.settings import CooperatorsSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.constants import ACTION_DURATION_S

# The reward's parts, signed: the goal's in millimetres, the trail's in
# squared degrees, the cost's in millijoules. The goal channel adds the first
# two, the trail channel, when there is one, the third, and the cost channel
# the rest; without a trail channel the goal also adds the third, always 0.
PART_NAMES = (
    "progress",
    "arrival",
    "trail",
    "pushing",
    "braking",
    "force",
    "pain",
    "chain",
    "carry",
)
GOAL_PARTS = 2
TRAIL_PART = 2
# Twice (1 - cos) of an angle, in squared degrees: its square while it is
# small, levelling off past a right angle, so that a body walking backward
# costs only a few times a bad turn.
SQUARED_DEGREES = (180 / math.pi) ** 2 * 2


def energy_sharing_weights(
    segment_count: int, observation_radius: int, decay: float
) -> torch.Tensor:
    """Each segment's shares of its own and its neighbours' energy, ``(N, N)``.

    Row ``i`` weighs every segment it sees, itself included, by ``decay`` to
    the power of their distance in segments, scaled so that the row adds up
    to one: halving per segment at radius 2, a middle segment counts itself
    40%, each nearest neighbour 20% and each second neighbour 10%.
    """
    index = torch.arange(segment_count)
    distance = (index[:, None] - index[None, :]).abs()
    weights = torch.where(
        distance <= observation_radius, decay ** distance.float(), 0.0
    )
    return weights / weights.sum(dim=1, keepdim=True)


class CooperativeReward:
    """Every segment's goal and cost for the step just taken."""

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
        self.sharing_weights = energy_sharing_weights(
            segment_count, observation_radius, settings.energy_sharing_decay
        ).to(commanded_joints.device)
        # Millijoules for one motor fully commanded through a step.
        self.force_price_mj = settings.force_price_w * ACTION_DURATION_S * 1e3
        self.pain_price_mj = (
            self.force_price_mj * settings.pain_price / body.leg_push_n**2
        )
        # Millijoules for a unit weight on a joint through a step.
        self.chain_price_mj = self.force_price_mj * settings.chain_price
        self.unit_weight_n = body.unit_weight_n
        # 1 for the segments paid progress and for those that pay their rear
        # pull; the rear, without a joint behind it, pays none either way.
        device = commanded_joints.device
        self.progress_paid = torch.ones(segment_count, device=device)
        if not settings.follower_goal:
            self.progress_paid[1:] = 0.0
        self.rear_pull_paid = torch.zeros(segment_count, device=device)
        if settings.chain_rear_pull == "every":
            self.rear_pull_paid[:] = 1.0
        elif settings.chain_rear_pull == "head":
            self.rear_pull_paid[0] = 1.0
        # The trail: each segment pays its own offset at its weight, the head
        # at 1, and the offset of the segment behind, which it steers, at 1.
        self.own_offset_paid = torch.full(
            (segment_count,), settings.trail_own_weight, device=device
        )
        self.own_offset_paid[0] = 1.0
        # Carrying itself: the head pays only for being held up.
        self.carry_price_mj = self.force_price_mj * settings.carry_price
        self.along_carry_paid = torch.ones(segment_count, device=device)
        self.along_carry_paid[0] = 0.0
        self.channel_count = 3 if settings.follow_trail else 2

    def compute(
        self,
        physical_state: PhysicalState,
        previous_body_planar_position: torch.Tensor,
        previous_head_tip_position: torch.Tensor,
        target_position: torch.Tensor,
        arrived: torch.Tensor,
        arrival_worth_m: float,
        previous_joint_position: torch.Tensor,
        joint_action: torch.Tensor,
        trail_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, StepRewards, torch.Tensor]:
        """The ``(W, N, C)`` rewards, goal, with the trail the trail, then
        cost, their parts, and each segment's own energy before sharing,
        ``(W, N, 3)`` in mJ.

        The goal: the millimetres each segment came closer to its target, the
        head's tip to the target, every other segment's centre to where the
        segment ahead's centre was before the step (none without
        ``follower_goal``); and, when the head arrives, ``arrival_worth_m`` in
        millimetres. The trail, with ``follow_trail``: the squared offsets,
        ``trail_offset`` ``(W, N)`` (``TrailMemory``), of the segment behind
        and, at its weight, of the segment itself. The cost: each segment's
        energy, shared by the sharing weights, its own pain, its own chain:
        the size of its front joint's force, push or lag, and, for those
        ``chain_rear_pull`` names, the square of its rear joint's where it
        pulls the segment behind, both in unit weights, the head having no
        front joint, the rear no rear one; and carrying itself: the size of
        the net force its joints put on it along the body, and of what they
        hold it up with, in unit weights, the head paying only the second.
        The other arguments are those of ``RewardContext.from_step``; the
        parts come back as ``StepRewards``, whose ``rewards`` add the channels.
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
            joint_action,
        )
        progress_m = context.distance_before - context.distance_after
        arrival = torch.zeros_like(progress_m)
        arrival[:, 0] = context.arrived * arrival_worth_m * 1e3

        settings = self.settings
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
        front = physical_state.front_joint_force / self.unit_weight_n
        rear_pull = torch.zeros_like(front)
        rear_pull[:, :-1] = front[:, 1:].clamp(min=0.0)
        chain = self.chain_price_mj * (
            front.abs() + self.rear_pull_paid * rear_pull.square()
        )
        trail = torch.zeros_like(progress_m)
        if settings.follow_trail:
            squared = SQUARED_DEGREES * (1 - trail_offset.cos())
            trail = self.own_offset_paid * squared
            trail[:, :-1] += squared[:, 1:]
        net = physical_state.net_joint_force / self.unit_weight_n
        carry = self.carry_price_mj * (
            self.along_carry_paid * net[..., 0].abs() + net[..., 1].clamp(min=0.0)
        )
        parts = torch.cat(
            (
                torch.stack(
                    (progress_m * 1e3 * self.progress_paid, arrival, -trail), dim=-1
                ),
                -shared_energy,
                -pain[..., None],
                -chain[..., None],
                -carry[..., None],
            ),
            dim=-1,
        )
        goal_parts = GOAL_PARTS if self.channel_count == 3 else TRAIL_PART + 1
        channels = [parts[..., :goal_parts].sum(dim=-1)]
        if self.channel_count == 3:
            channels.append(parts[..., TRAIL_PART])
        channels.append(parts[..., TRAIL_PART + 1 :].sum(dim=-1))
        channels = torch.stack(channels, dim=-1)
        step_rewards = StepRewards(
            rewards=channels.sum(dim=-1),
            reward_parts=parts,
            segment_progress=progress_m,
            joint_movement=context.joint_movement,
        )
        return channels, step_rewards, own_energy
