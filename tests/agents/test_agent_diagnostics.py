"""Tests for the agents' learning diagnostics.

The summaries are hand-made, so these tests cover only what the diagnostics
add: each value reaches its segment's row, and the spreads are the policy's.
"""

import math

import pytest
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
    diagnostics = AgentDiagnostics(
        tempo_columns=[[], []],
        motor_columns=[list(range(7)), list(range(6))],
        device="cpu",
    )

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

    # With leg clocks, each leg's six actions start with its tempo, and its
    # joints' centres stand for its motors; the head's spine command follows.
    segment_agents = [
        SegmentAgent(index, 2, 3, 1, settings, index, 13 - index) for index in range(2)
    ]
    with torch.no_grad():
        log_std = segment_agents[0].ppo.actor_network.log_std
        log_std[0] = math.log(0.1)  # the left leg's tempo
        log_std[3:6] = math.log(0.2)  # its joints' centres
        log_std[12] = math.log(0.3)  # the spine command
    centres = [3, 4, 5, 9, 10, 11]
    clock_diagnostics = AgentDiagnostics(
        [[0, 6], [0, 6]], [centres + [12], centres], device="cpu"
    )
    clock_diagnostics.record_update(summaries, segment_agents)
    learning = values(clock_diagnostics.learning)
    assert learning["action_std"][0].tolist() == pytest.approx([0.2] * 3 + [0.5] * 3)
    assert learning["spine_action_std"].tolist() == pytest.approx([0.3, 0.0])
    assert learning["tempo_action_std"].tolist() == pytest.approx([0.3, 0.5])
