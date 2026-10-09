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
        action_std_schedule="learned",
        final_action_std=0.5,
        exploration_noise_beta=0.0,
        optimizer="adam",
        learning_rate=3e-4,
        learning_rate_schedule="constant",
        final_learning_rate=3e-4 / 10,
        weight_decay=0.0,
        momentum=0.9,
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
    ("schedule", "rates"),
    [
        ("constant", [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]),
        ("linear", [1e-3, 7.75e-4, 5.5e-4, 3.25e-4, 1e-4]),
        # Half a cosine: slow at both ends, halfway in the middle.
        ("cosine", [1e-3, 8.682e-4, 5.5e-4, 2.318e-4, 1e-4]),
    ],
)
def test_the_learning_rate_moves_from_the_first_to_the_last_update(schedule, rates):
    settings = AgentSettings.from_section(
        {
            "learning_rate": 1e-3,
            "learning_rate_schedule": schedule,
            "final_learning_rate": 1e-4,
        }
    )

    assert [settings.learning_rate_at(cycle, 5) for cycle in range(1, 6)] == (
        pytest.approx(rates, rel=1e-3)
    )


def test_a_scheduled_action_spread_moves_by_equal_ratios_and_a_learned_has_none():
    settings = AgentSettings.from_section(
        {
            "initial_action_std": 0.4,
            "action_std_schedule": "log_linear",
            "final_action_std": 0.1,
        }
    )

    assert [settings.action_std_at(cycle, 3) for cycle in range(1, 4)] == (
        pytest.approx([0.4, 0.2, 0.1])
    )
    assert AgentSettings.from_section({}).action_std_at(1, 3) is None


@pytest.mark.parametrize(
    ("values", "problem"),
    [
        ({"device": "gpu"}, r"\[agents\] device"),
        ({"optimizer": "rmsprop"}, "optimizer"),
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
