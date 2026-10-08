"""Tests for one segment agent.

PPO, the normaliser, and advantage estimation are tested in RL_lib, and the
storage in its own file; these tests cover how the agent connects them: when
the normaliser changes, what is stored, how a window is learned from, and its
saved state.
"""

import copy

import pytest
import torch

from centipede.agents.segment_agent import SegmentAgent
from centipede.agents.settings import AgentSettings

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
WORLDS, OBSERVATION_SIZE, WINDOW_STEPS = 4, 5, 3


def make_agent(device="cpu", seed=0) -> SegmentAgent:
    settings = AgentSettings.from_section(
        {"device": device, "hidden_layers": [8], "ppo": {"minibatch_size": 4}}
    )
    return SegmentAgent(0, WORLDS, OBSERVATION_SIZE, WINDOW_STEPS, settings, seed)


def observations(step: int, device="cpu") -> torch.Tensor:
    generator = torch.Generator().manual_seed(step)
    return torch.randn(WORLDS, OBSERVATION_SIZE, generator=generator).to(device)


def collect_window(agent: SegmentAgent, device="cpu") -> None:
    """One window of training steps; the last world terminates on step 1."""
    for step in range(WINDOW_STEPS):
        agent.act(observations(step, device), training=True)
        terminated = torch.tensor([False, False, False, step == 1], device=device)
        agent.record(
            torch.linspace(-1.0, 1.0, WORLDS, device=device),
            terminated,
            torch.zeros(WORLDS, dtype=torch.bool, device=device),
            observations(step + 1, device),
        )


def test_only_training_changes_the_normaliser_and_is_stored():
    agent = make_agent()

    evaluation_actions = agent.act(observations(0), training=False)
    assert agent.observation_normaliser.observation_count == 0
    assert agent.rollout_storage.step_index == 0
    # Evaluation uses each world's mean action, so it repeats exactly.
    torch.testing.assert_close(
        agent.act(observations(0), training=False), evaluation_actions
    )

    actions = agent.act(observations(0), training=True)
    assert actions.shape == (WORLDS, 6)
    assert actions.abs().max() < 1
    assert agent.observation_normaliser.observation_count == WORLDS
    # The stored observation is the normalised one the policy saw.
    torch.testing.assert_close(
        agent.rollout_storage.observations[0],
        agent.observation_normaliser.normalize(observations(0)),
    )

    # The next value is the critic's, after normalising without updating.
    agent.record(
        torch.ones(WORLDS),
        torch.zeros(WORLDS, dtype=torch.bool),
        torch.zeros(WORLDS, dtype=torch.bool),
        observations(1),
    )
    assert agent.observation_normaliser.observation_count == WORLDS
    torch.testing.assert_close(
        agent.rollout_storage.next_values[0],
        agent.ppo.state_value(agent.observation_normaliser.normalize(observations(1))),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_update_learns_from_the_window_then_starts_a_new_one(device):
    agent = make_agent(device)
    actor_before = copy.deepcopy(agent.ppo.actor_network.state_dict())
    collect_window(agent, device)

    agent.update()

    assert agent.update_count == 1
    assert agent.rollout_storage.step_index == 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            actor_before.values(),
            agent.ppo.actor_network.state_dict().values(),
            strict=True,
        )
    )


def test_a_restored_agent_acts_and_learns_exactly_like_the_original():
    original = make_agent()
    collect_window(original)
    original.update()
    restored = make_agent(seed=1)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))

    for agent in (original, restored):
        collect_window(agent)
        agent.update()

    assert restored.update_count == original.update_count == 2
    torch.testing.assert_close(
        restored.observation_normaliser.mean, original.observation_normaliser.mean
    )
    for restored_value, original_value in zip(
        restored.ppo.actor_network.state_dict().values(),
        original.ppo.actor_network.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(restored_value, original_value)


def test_a_widened_agent_acts_as_before_and_commands_zero_to_its_new_motor():
    """A new run that sees and commands more than its parent, such as the spine."""
    original = make_agent()
    collect_window(original)
    original.update()
    widened = SegmentAgent(
        0, WORLDS, OBSERVATION_SIZE + 2, WINDOW_STEPS, original.settings, 1, 7
    )

    widened.load_state_dict(copy.deepcopy(original.state_dict()), widen=True)

    more = torch.cat((observations(9), torch.randn(WORLDS, 2)), dim=-1)
    actions = widened.act(more, training=False)
    torch.testing.assert_close(
        actions[:, :6], original.act(observations(9), training=False)
    )
    assert torch.all(actions[:, 6].abs() < 1e-7)
    spread = widened.ppo.actor_network.log_std.exp()
    assert spread[6].item() == pytest.approx(original.settings.initial_action_std)
    assert not widened.ppo.actor_optimizer.state  # the saved state does not fit
    assert widened.update_count == 1


def test_the_seed_alone_decides_the_starting_networks():
    def starting_weights(seed: int) -> list[torch.Tensor]:
        torch.manual_seed(100)  # the same global state for every agent seed
        return list(make_agent(seed=seed).ppo.actor_network.state_dict().values())

    assert all(map(torch.equal, starting_weights(0), starting_weights(0)))
    assert not all(map(torch.equal, starting_weights(0), starting_weights(1)))


def test_the_settings_choose_the_optimizer_and_loading_keeps_this_agents_own():
    def agent_with(**values):
        settings = AgentSettings.from_section({"hidden_layers": [8], **values})
        return SegmentAgent(0, WORLDS, OBSERVATION_SIZE, WINDOW_STEPS, settings, 0)

    adam = agent_with(weight_decay=0.01)
    sgd = agent_with(optimizer="sgd", momentum=0.5)
    assert type(adam.ppo.actor_optimizer) is torch.optim.Adam
    assert type(sgd.ppo.critic_optimizer) is torch.optim.SGD
    assert sgd.ppo.critic_optimizer.param_groups[0]["momentum"] == 0.5

    adam.set_learning_rate(1e-5)
    restored = agent_with()  # no weight decay
    restored.load_state_dict(adam.state_dict())
    for optimizer in (restored.ppo.actor_optimizer, restored.ppo.critic_optimizer):
        group = optimizer.param_groups[0]
        assert group["weight_decay"] == 0.0  # its own, not the saved 0.01
        assert group["lr"] == 1e-5  # until the experiment sets the next one
