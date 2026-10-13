"""Tests for what the workspace branch adds to the agents."""

import torch

from centipede.agents.agents import Agents
from centipede.agents.settings import AgentSettings


def test_curve_actions_start_at_a_random_guess_within_the_spread():
    """Each agent's actions that draw its curves start at its own values within
    plus or minus 0.2; the others start near 0."""
    settings = AgentSettings.from_section(
        {"actor_last_layer_scale": 0.01, "initial_action_offset_spread": 0.2}
    )
    shape_columns = [[1, 2, 3], [1, 2, 3]]
    agents = Agents(2, 5, 4, 1, settings, 0, [4, 4], shape_columns=shape_columns)

    actions = agents.act(torch.zeros(4, 2, 5), training=False)

    guesses = actions[0, :, 1:]
    assert guesses.abs().max() <= 0.21 and guesses.abs().min() > 1e-3
    assert not torch.allclose(guesses[0], guesses[1])
    assert actions[..., 0].abs().max() < 0.05


def test_a_follower_explores_its_tempo_in_proportion_to_its_gap_ahead():
    """Gaps of 0, 0.085 and 0.5 octaves, in the last observation, scale a
    follower's tempo noise to its floor, half, and all of its spread; its
    other actions and the head's whole action keep theirs."""
    settings = AgentSettings.from_section({"follower_tempo_noise_gap_octaves": 0.17})
    agents = Agents(
        2, 5, 3, 1, settings, 0, [4, 4], [[0], [0]], tempo_gap_ahead_column=4
    )
    observations = torch.zeros(3, 2, 5)
    observations[:, :, 4] = torch.tensor([0.0, 0.085, -0.5])[:, None]

    agents.act(observations, training=True)

    head, follower = (agent.rollout_storage for agent in agents.segment_agents)
    assert head.noise_scales is None
    torch.testing.assert_close(
        follower.noise_scales[0, :, 0], torch.tensor([0.03, 0.5, 1.0])
    )
    assert torch.all(follower.noise_scales[0, :, 1:] == 1)
