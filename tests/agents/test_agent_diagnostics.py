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
    # Segment 0 also commands the spine joint behind it; segment 1 explores
    # with gSDE, one spread per feature of its last hidden layer and action.
    gsde = AgentSettings.from_section(
        {
            "hidden_layers": [4],
            "state_dependent_noise_steps": 8,
            "actor_activation": "tanh",
        }
    )
    segment_agents = [
        SegmentAgent(index, 2, 3, 1, (settings, gsde)[index], index, 7 - index)
        for index in range(2)
    ]
    with torch.no_grad():
        segment_agents[1].ppo.actor_network.log_std.fill_(math.log(0.25))
    gsde_actor = segment_agents[1].ppo.actor_network
    assert gsde_actor.log_std.shape == (4, 6)
    assert isinstance(gsde_actor.mean_network[1], torch.nn.Tanh)
    summaries = [
        PPOUpdateSummary(*(10.0 * index + part for part in range(6)))
        for index in range(2)
    ]
    diagnostics = AgentDiagnostics(segment_count=2, leg_motor_count=6, device="cpu")

    diagnostics.record_update(summaries, segment_agents)

    learning = values(diagnostics.learning)
    assert list(learning) == list(descriptions(LearningSummary))
    assert learning["actor_loss"].tolist() == [0.0, 10.0]
    assert learning["explained_variance"].tolist() == [5.0, 15.0]
    assert learning["critic_explained_variances"].tolist() == [[5.0], [15.0]]
    torch.testing.assert_close(
        learning["action_std"], torch.tensor([[0.5] * 6, [0.25] * 6])
    )
    assert learning["spine_action_std"].tolist() == [0.5, 0.0]
