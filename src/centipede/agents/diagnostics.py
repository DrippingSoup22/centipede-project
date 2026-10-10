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

from centipede.agents.segment_agent import LEG_ACTION_COUNT, SegmentAgent
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
        "Each leg action's learned spread, before squashing; with leg clocks,"
        " each leg joint's centre's, (N, 6)",
        parts=(
            "left shoulder sweep",
            "left shoulder lift",
            "left knee",
            "right shoulder sweep",
            "right shoulder lift",
            "right knee",
        ),
    )
    spine_action_std: torch.Tensor = measure(
        "The learned spread of the command for the spine joint behind the"
        " segment, before squashing; 0 for a segment without one, (N,)"
    )
    tempo_action_std: torch.Tensor = measure(
        "The learned spread of the segment's clock tempo action, before"
        " squashing; with leg clocks, the mean of its two legs'; 0 without"
        " clocks, (N,)"
    )


# The values copied straight from RL_lib's update summary, in its order.
SUMMARY_NAMES = tuple(item.name for item in fields(PPOUpdateSummary))


class AgentDiagnostics:
    """Fills the agents' learning category, ``learning``, which the experiment reads."""

    def __init__(
        self,
        tempo_columns: list[list[int]],
        motor_columns: list[list[int]],
        device: torch.device | str,
    ) -> None:
        """Allocate the category, all zero.

        For each segment, ``tempo_columns`` lists its clock tempo actions,
        and ``motor_columns`` the action that sets each of its motors: its six
        leg motors, then the spine joint behind it when it commands one.
        """
        self.tempo_columns = tempo_columns
        self.motor_columns = motor_columns
        segment_count = len(tempo_columns)
        self.learning = LearningSummary(
            **{
                name: torch.zeros(segment_count, device=device)
                for name in SUMMARY_NAMES
            },
            learning_rate=torch.zeros((), device=device),
            action_std=torch.zeros((segment_count, LEG_ACTION_COUNT), device=device),
            spine_action_std=torch.zeros(segment_count, device=device),
            tempo_action_std=torch.zeros(segment_count, device=device),
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
                spread = agent.ppo.actor_network.log_std.clamp(-20.0, 2.0).exp()
                motors = spread[self.motor_columns[segment_index]]
                self.learning.action_std[segment_index].copy_(motors[:LEG_ACTION_COUNT])
                if len(motors) > LEG_ACTION_COUNT:
                    self.learning.spine_action_std[segment_index] = motors[-1]
                tempo_columns = self.tempo_columns[segment_index]
                if tempo_columns:
                    self.learning.tempo_action_std[segment_index] = spread[
                        tempo_columns
                    ].mean()
