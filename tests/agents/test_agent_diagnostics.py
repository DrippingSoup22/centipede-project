"""Tests for the agents' learning diagnostics.

The summaries are hand-made, so these tests cover only what the diagnostics
add: each value reaches its segment's row, and the spreads are the policy's.
"""

import math

import torch
from rl_lib.algorithms.policy_gradient import PPOUpdateSummary

from centipede.agents.diagnostics import AgentDiagnostics, LearningSummary
from centipede.agents.segment_agent import SegmentAgent
from centipede.agents.settings import AgentSettings
from centipede.diagnostics_category import descriptions, values


def test_each_segment_row_gets_its_own_summary_and_spreads():
    settings = AgentSettings.from_section({"hidden_layers": [4]})
    segment_agents = [
        SegmentAgent(index, 2, 3, 1, settings, index) for index in range(2)
    ]
    with torch.no_grad():
        segment_agents[1].ppo.actor_network.log_std.fill_(math.log(0.25))
    summaries = [
        PPOUpdateSummary(*(10.0 * index + part for part in range(6)))
        for index in range(2)
    ]
    diagnostics = AgentDiagnostics(segment_count=2, device="cpu")

    diagnostics.record_update(summaries, segment_agents)

    learning = values(diagnostics.learning)
    assert list(learning) == list(descriptions(LearningSummary))
    assert learning["actor_loss"].tolist() == [0.0, 10.0]
    assert learning["explained_variance"].tolist() == [5.0, 15.0]
    torch.testing.assert_close(
        learning["action_std"], torch.tensor([[0.5] * 6, [0.25] * 6])
    )
