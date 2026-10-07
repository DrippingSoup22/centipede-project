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

from centipede.agents.segment_agent import ACTION_SIZE, SegmentAgent
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
        "update: 1 exactly, 0 nothing, (N,)"
    )
    learning_rate: torch.Tensor = measure(
        "The optimizers' learning rate in the update, the same for every agent"
    )
    action_std: torch.Tensor = measure(
        "Each action's learned spread, before squashing, (N, 6)",
        parts=(
            "left shoulder sweep",
            "left shoulder lift",
            "left knee",
            "right shoulder sweep",
            "right shoulder lift",
            "right knee",
        ),
    )


# The values copied straight from RL_lib's update summary, in its order.
SUMMARY_NAMES = tuple(item.name for item in fields(PPOUpdateSummary))


class AgentDiagnostics:
    """Fills the agents' learning category, ``learning``, which the experiment reads."""

    def __init__(self, segment_count: int, device: torch.device | str) -> None:
        """Allocate the category, all zero."""
        self.learning = LearningSummary(
            **{
                name: torch.zeros(segment_count, device=device)
                for name in SUMMARY_NAMES
            },
            learning_rate=torch.zeros((), device=device),
            action_std=torch.zeros((segment_count, ACTION_SIZE), device=device),
        )

    def record_update(
        self,
        summaries: Sequence[PPOUpdateSummary],
        segment_agents: Sequence[SegmentAgent],
    ) -> None:
        """Copy every agent's update summary and current action spreads.

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
        with torch.no_grad():
            for segment_index, agent in enumerate(segment_agents):
                # The policy clamps the log spread to [-20, 2] before using it.
                log_std = agent.ppo.actor_network.log_std.clamp(-20.0, 2.0)
                self.learning.action_std[segment_index].copy_(log_std.exp())
