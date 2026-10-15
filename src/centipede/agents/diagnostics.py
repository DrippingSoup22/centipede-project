"""The agents' diagnostics: how each segment agent's last update went.

The category is allocated once and refreshed in place. The front file fills it
with one call, ``record_update``, right after every segment agent has updated.
The PPO summaries are already Python numbers, read once per update by RL_lib,
so nothing here adds a wait for the GPU. The values are listed in
docs/diagnostics.md.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch
from rl_lib.algorithms.policy_gradient import PPOUpdateSummary

from centipede.agents.segment_agent import SegmentAgent
from centipede.diagnostics_category import measure


@dataclass(frozen=True)
class LearningSummary:
    """One row per segment agent, from its last update."""

    actor_loss: torch.Tensor = measure(
        "Clipped policy loss, including the entropy bonus, (N,)"
    )
    critic_loss: torch.Tensor = measure(
        "Half the mean squared error of the values, (N,)"
    )
    entropy: torch.Tensor = measure(
        "Estimated entropy of the policy: how widely it explores, (N,)"
    )
    approximate_kl: torch.Tensor = measure("How far the update moved the policy, (N,)")
    clip_fraction: torch.Tensor = measure(
        "Share of samples whose probability ratio was clipped, (N,)"
    )
    explained_variance: torch.Tensor = measure(
        "How much of the returns' variation the critic predicted before the "
        "update: 1 exactly, 0 nothing, (N,); with several critics, the first's"
    )
    critic_explained_variances: torch.Tensor = measure(
        "The same for each critic, one per reward channel; NaN for a channel"
        " the segment does not learn from, (N, K)"
    )
    baseline_explained_variances: torch.Tensor = measure(
        "The same for each critic's baseline, which also sees the neighbours'"
        " actions; NaN without baselines or that channel, (N, K)"
    )
    learning_rate: torch.Tensor = measure(
        "The actor's learning rate in the update; the critics' is the same"
        " unless the schedule is adaptive, (N,)"
    )
    action_std: torch.Tensor = measure(
        "Each leg action's learned spread, before squashing, in leg action"
        " order: the left leg's joints from the body outward, then the right"
        " leg's, (N, M)"
    )
    spine_action_std: torch.Tensor = measure(
        "The learned spread of the command for the spine joint behind the"
        " segment, before squashing; 0 for a segment without one, (N,)"
    )


# The single values copied straight from RL_lib's update summary, in its
# order; the per-critic tuples that follow them are copied apart.
SUMMARY_NAMES = tuple(
    item.name for item in fields(PPOUpdateSummary) if item.type is float
)


class AgentDiagnostics:
    """Fills the agents' learning category, ``learning``, which the experiment reads."""

    def __init__(
        self,
        segment_count: int,
        leg_motor_count: int,
        device: torch.device | str,
        reward_channel_count: int = 1,
        segment_reward_channels: list[tuple[int, ...]] | None = None,
    ) -> None:
        """Allocate the category, all zero, for agents whose first
        ``leg_motor_count`` actions drive their legs and that learn from
        ``segment_reward_channels`` (every channel if not given)."""
        self.leg_motor_count = leg_motor_count
        self.segment_reward_channels = (
            segment_reward_channels
            or [tuple(range(reward_channel_count))] * segment_count
        )
        self.learning = LearningSummary(
            **{
                name: torch.zeros(segment_count, device=device)
                for name in SUMMARY_NAMES
            },
            critic_explained_variances=torch.zeros(
                (segment_count, reward_channel_count), device=device
            ),
            baseline_explained_variances=torch.full(
                (segment_count, reward_channel_count), torch.nan, device=device
            ),
            learning_rate=torch.zeros(segment_count, device=device),
            action_std=torch.zeros((segment_count, leg_motor_count), device=device),
            spine_action_std=torch.zeros(segment_count, device=device),
        )

    def record_update(
        self,
        summaries: Sequence[PPOUpdateSummary],
        segment_agents: Sequence[SegmentAgent],
    ) -> None:
        """Copy every agent's update summary, learning rate and current action
        spreads.

        ``summaries`` and ``segment_agents`` are both in segment order.
        """
        summary_rows = torch.tensor(
            [
                [getattr(summary, name) for name in SUMMARY_NAMES]
                for summary in summaries
            ]
        )
        for column, name in enumerate(SUMMARY_NAMES):
            getattr(self.learning, name).copy_(summary_rows[:, column])
        critics = torch.full_like(self.learning.critic_explained_variances, torch.nan)
        baselines = torch.full_like(critics, torch.nan)
        for segment_index, summary in enumerate(summaries):
            channels = list(self.segment_reward_channels[segment_index])
            critics[segment_index, channels] = torch.tensor(
                summary.explained_variances, device=critics.device
            )
            if summary.baseline_explained_variances:
                baselines[segment_index, channels] = torch.tensor(
                    summary.baseline_explained_variances, device=critics.device
                )
        self.learning.critic_explained_variances.copy_(critics)
        self.learning.baseline_explained_variances.copy_(baselines)
        with torch.no_grad():
            for segment_index, agent in enumerate(segment_agents):
                self.learning.learning_rate[segment_index] = (
                    agent.ppo.actor_optimizer.param_groups[0]["lr"]
                )
                # The policy clamps the log spread to [-20, 2] before using it.
                spread = agent.ppo.actor_network.log_std.clamp(-20.0, 2.0).exp()
                legs = self.leg_motor_count
                self.learning.action_std[segment_index].copy_(spread[:legs])
                if agent.action_size > legs:
                    self.learning.spine_action_std[segment_index] = spread[legs]
