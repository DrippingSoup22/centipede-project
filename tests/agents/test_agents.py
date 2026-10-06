"""Tests for the agents' front file.

A segment agent's own behaviour is tested in its file; these tests cover what
the group adds: how the body's tensors are split and joined, that each agent
learns from its own segment only, the checkpoint bundle, and the baselines.
"""

import io

import pytest
import torch

from centipede.agents.agents import Agents, RandomActionBaseline, ZeroActionBaseline
from centipede.agents.segment_agent import SegmentAgent
from centipede.agents.settings import AgentSettings

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
SEGMENTS, WORLDS, OBSERVATION_SIZE, WINDOW_STEPS, SEED = 3, 4, 5, 2, 7
SETTINGS = AgentSettings.from_section(
    {"hidden_layers": [8], "ppo": {"minibatch_size": 4}}
)


def make_agents(segment_count=SEGMENTS, observation_size=OBSERVATION_SIZE) -> Agents:
    return Agents(segment_count, observation_size, WORLDS, WINDOW_STEPS, SETTINGS, SEED)


def body_tensor(step: int, *trailing_shape: int) -> torch.Tensor:
    """Different values for every world and segment, repeated by ``step``."""
    generator = torch.Generator().manual_seed(step)
    return torch.randn(WORLDS, SEGMENTS, *trailing_shape, generator=generator)


def collect_window(agents) -> None:
    """One training window; world 1 terminates and world 2 is truncated on step 0."""
    for step in range(WINDOW_STEPS):
        agents.act(body_tensor(step, OBSERVATION_SIZE), training=True)
        agents.record(
            body_tensor(100 + step),
            torch.tensor([False, step == 0, False, False]),
            torch.tensor([False, False, step == 0, False]),
            body_tensor(step + 1, OBSERVATION_SIZE),
        )


class LoneSegment:
    """One segment agent fed by hand with its own column of the body's tensors."""

    def __init__(self, segment_index: int) -> None:
        self.segment_index = segment_index
        self.agent = SegmentAgent(
            segment_index,
            WORLDS,
            OBSERVATION_SIZE,
            WINDOW_STEPS,
            SETTINGS,
            SEED * SEGMENTS + segment_index,
        )

    def act(self, observations, training):
        return self.agent.act(observations[:, self.segment_index], training)

    def record(self, rewards, terminated, truncated, final_observations):
        column = self.segment_index
        self.agent.record(
            rewards[:, column], terminated, truncated, final_observations[:, column]
        )


def test_each_agent_acts_and_learns_from_its_own_segment_only():
    agents = make_agents()
    actions = agents.act(body_tensor(0, OBSERVATION_SIZE), training=False)
    assert actions.shape == (WORLDS, SEGMENTS, 6)
    assert actions.abs().max() < 1

    collect_window(agents)
    agents.update()

    # Each agent ends exactly where a lone agent with the same seed ends when
    # given only that segment's column: nothing else reached it.
    for segment_index, agent in enumerate(agents.segment_agents):
        lone = LoneSegment(segment_index)
        collect_window(lone)
        lone.agent.update()
        for grouped, alone in zip(
            agent.ppo.actor_network.state_dict().values(),
            lone.agent.ppo.actor_network.state_dict().values(),
            strict=True,
        ):
            assert torch.equal(grouped, alone)
    assert agents.diagnostics.learning.actor_loss.ne(0).all()


def test_a_saved_file_restores_every_agent_and_rejects_another_body():
    original = make_agents()
    collect_window(original)
    original.update()
    file = io.BytesIO()
    torch.save(original.state_dict(), file)
    file.seek(0)
    # The experiment loads checkpoints with PyTorch's safe loader.
    state = torch.load(file, weights_only=True)

    restored = Agents(SEGMENTS, OBSERVATION_SIZE, WORLDS, WINDOW_STEPS, SETTINGS, 0)
    restored.load_state_dict(state)
    observations = body_tensor(5, OBSERVATION_SIZE)
    assert torch.equal(
        restored.act(observations, training=False),
        original.act(observations, training=False),
    )

    with pytest.raises(ValueError, match="4 segments with 5"):
        make_agents(segment_count=4).load_state_dict(state)
    with pytest.raises(ValueError, match="3 segments with 6"):
        make_agents(observation_size=6).load_state_dict(state)


@pytest.mark.parametrize("device", DEVICES)
def test_baselines_give_zero_and_seeded_uniform_actions(device):
    observations = torch.ones(WORLDS, SEGMENTS, OBSERVATION_SIZE, device=device)

    zero = ZeroActionBaseline().act(observations, training=False)
    random = RandomActionBaseline(device, seed=3).act(observations, training=False)

    assert torch.equal(zero, torch.zeros(WORLDS, SEGMENTS, 6, device=device))
    assert random.shape == (WORLDS, SEGMENTS, 6)
    assert random.device.type == device
    assert -1 <= random.min() and random.max() < 1
    assert torch.equal(
        random, RandomActionBaseline(device, seed=3).act(observations, training=False)
    )
