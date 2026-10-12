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
