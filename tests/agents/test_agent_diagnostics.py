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
    # Segment 0 also commands the spine joint behind it.
    segment_agents = [
        SegmentAgent(index, 2, 3, 1, settings, index, 7 - index) for index in range(2)
    ]
    with torch.no_grad():
        segment_agents[1].ppo.actor_network.log_std.fill_(math.log(0.25))
    summaries = [
        PPOUpdateSummary(*(10.0 * index + part for part in range(6)))
        for index in range(2)
    ]
    diagnostics = AgentDiagnostics(segment_count=2, tempo_actions=False, device="cpu")

    diagnostics.record_update(summaries, segment_agents)

    learning = values(diagnostics.learning)
    assert list(learning) == list(descriptions(LearningSummary))
    assert learning["actor_loss"].tolist() == [0.0, 10.0]
    assert learning["explained_variance"].tolist() == [5.0, 15.0]
    torch.testing.assert_close(
        learning["action_std"], torch.tensor([[0.5] * 6, [0.25] * 6])
    )
    assert learning["spine_action_std"].tolist() == [0.5, 0.0]
    assert not learning["tempo_action_std"].any()

    # With clocks, each agent's last action is its tempo: the head's eighth
    # after its spine command, segment 1's seventh.
    segment_agents = [
        SegmentAgent(index, 2, 3, 1, settings, index, 8 - index) for index in range(2)
    ]
    with torch.no_grad():
        segment_agents[0].ppo.actor_network.log_std[-1] = math.log(0.25)
    clock_diagnostics = AgentDiagnostics(2, tempo_actions=True, device="cpu")
    clock_diagnostics.record_update(summaries, segment_agents)
    learning = values(clock_diagnostics.learning)
    assert learning["spine_action_std"].tolist() == [0.5, 0.0]
    assert learning["tempo_action_std"].tolist() == [0.25, 0.5]
