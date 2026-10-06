"""Tests for the [agents] configuration section and its nested PPO section.

The generic value checks are tested with ``SettingsSection``; these tests cover
only the agents' keys, defaults, limits, and how the PPO section is handed on.
"""

import pytest

from centipede.agents.settings import AgentSettings, PPOSettings
from centipede.settings_section import SettingsError


def test_an_empty_section_gives_the_agreed_first_values():
    settings = AgentSettings.from_section({})

    assert settings == AgentSettings(
        device="cpu",
        hidden_layers=(64, 64),
        initial_action_std=0.5,
        learning_rate=3e-4,
        normaliser_epsilon=1e-8,
        observation_clip=10.0,
        ppo=PPOSettings(
            discount=0.999,
            gae_lambda=0.95,
            clip_ratio=0.2,
            update_epochs=4,
            minibatch_size=64,
            max_gradient_norm=0.5,
            entropy_coefficient=0.001,
        ),
    )


def test_the_ppo_section_is_read_by_its_own_class():
    settings = AgentSettings.from_section(
        {"device": "cuda", "ppo": {"update_epochs": 2, "entropy_coefficient": 0}}
    )

    assert settings.device == "cuda"
    assert settings.ppo.update_epochs == 2
    assert settings.ppo.entropy_coefficient == 0.0


@pytest.mark.parametrize(
    ("values", "problem"),
    [
        ({"device": "gpu"}, r"\[agents\] device"),
        ({"hidden_layers": [64, 0]}, "hidden_layers"),
        ({"ppo": {"discount": 1.5}}, "discount"),
        ({"ppo": {"clip_ratio": 1.0}}, "clip_ratio"),
        ({"ppo": {"entropy_coefficient": -0.1}}, "entropy_coefficient"),
        ({"ppo": {"learning_rate": 1e-3}}, r"\[agents.ppo\] has unknown"),
    ],
)
def test_invalid_values_and_unknown_keys_name_their_section(values, problem):
    with pytest.raises(SettingsError, match=problem):
        AgentSettings.from_section(values)
