"""The cooperators' reward: a goal and a cost for every segment, two channels
that two critics judge apart (docs/cooperators/reward.md)."""

import torch

from centipede.environment.reward_function import RewardContext, StepRewards
from centipede.environment.settings import CooperatorsSettings
from centipede.environment.simulation import PhysicalState
from centipede.environment.simulation.body_measures import BodyMeasures
from centipede.environment.simulation.constants import ACTION_DURATION_S

# The reward's parts, signed, in millimetres for the goal and millijoules for
# the cost: the goal channel adds the first two, the cost channel the rest.
PART_NAMES = ("progress", "arrival", "pushing", "braking", "force", "pain")
GOAL_PARTS = 2


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
    ) -> tuple[torch.Tensor, StepRewards]:
        """The ``(W, N, 2)`` rewards, goal then cost, and their parts.

        The goal: the millimetres each segment came closer to its target, the
        head's tip to the target, every other segment's centre to where the
        segment ahead's centre was before the step; and, when the head
        arrives, ``arrival_worth_m`` in millimetres. The cost: each segment's
        energy, shared by the sharing weights, and its own pain. The arguments
        are those of ``RewardContext.from_step``; the parts come back as
        ``StepRewards``, whose ``rewards`` add both channels.
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
        parts = torch.cat(
            (
                torch.stack((progress_m * 1e3, arrival), dim=-1),
                -shared_energy,
                -pain[..., None],
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
        return channels, StepRewards(
            rewards=channels.sum(dim=-1),
            reward_parts=parts,
            segment_progress=progress_m,
            joint_movement=context.joint_movement,
        )
