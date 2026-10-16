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
LEG_MOTORS = 6  # legs of three joints
SETTINGS = AgentSettings.from_section(
    {"hidden_layers": [8], "ppo": {"minibatch_size": 4}}
)


def make_agents(segment_count=SEGMENTS, observation_size=OBSERVATION_SIZE) -> Agents:
    return Agents(
        segment_count,
        observation_size,
        WORLDS,
        WINDOW_STEPS,
        SETTINGS,
        SEED,
        [LEG_MOTORS] * segment_count,
        LEG_MOTORS,
    )


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
            LEG_MOTORS,
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


def test_an_adaptive_schedule_lets_each_actor_steer_its_rate_by_its_own_kl():
    settings = AgentSettings.from_section(
        {
            "hidden_layers": [8],
            "learning_rate_schedule": "adaptive",
            "desired_kl": 1.0,
            "ppo": {"minibatch_size": 4},
        }
    )
    agents = Agents(
        SEGMENTS,
        OBSERVATION_SIZE,
        WORLDS,
        WINDOW_STEPS,
        settings,
        SEED,
        [LEG_MOTORS] * SEGMENTS,
        LEG_MOTORS,
    )
    assert settings.learning_rate_at(1, 10) is None  # left to the agents
    collect_window(agents)
    agents.update()

    # Each update moved its policy far less than half the desired KL: the
    # diagnostics hold the rate it used, and each actor's rose by 1.5 for the
    # next update, while the critics kept theirs.
    learning_rates = agents.diagnostics.learning.learning_rate.tolist()
    assert learning_rates == pytest.approx([3e-4] * SEGMENTS)
    for agent in agents.segment_agents:
        assert agent.ppo.actor_optimizer.param_groups[0]["lr"] == pytest.approx(4.5e-4)
        assert agent.ppo.critic_optimizer.param_groups[0]["lr"] == 3e-4
    # Above twice the desired KL, it falls by 1.5.
    first = agents.segment_agents[0]
    first.adapt_learning_rate(2.5)
    assert first.ppo.actor_optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_a_saved_file_restores_every_agent_and_rejects_another_body():
    original = make_agents()
    collect_window(original)
    original.update()
    file = io.BytesIO()
    torch.save(original.state_dict(), file)
    file.seek(0)
    # The experiment loads checkpoints with PyTorch's safe loader.
    state = torch.load(file, weights_only=True)

    restored = Agents(
        SEGMENTS,
        OBSERVATION_SIZE,
        WORLDS,
        WINDOW_STEPS,
        SETTINGS,
        0,
        [LEG_MOTORS] * SEGMENTS,
        LEG_MOTORS,
    )
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


def test_segments_may_command_more_motors_and_a_new_run_widens_saved_agents():
    """With the spine, every segment but the rear commands a seventh motor."""
    saved = make_agents().state_dict()
    wider = Agents(
        SEGMENTS,
        OBSERVATION_SIZE + 2,
        WORLDS,
        WINDOW_STEPS,
        SETTINGS,
        SEED,
        [7, 7, 6],
        LEG_MOTORS,
    )

    actions = wider.act(body_tensor(0, OBSERVATION_SIZE + 2), training=False)

    assert actions.shape == (WORLDS, SEGMENTS, 7)
    assert not actions[:, 2, 6].any()  # the rear's seventh column is padding
    with pytest.raises(ValueError, match=r"\[6, 6, 6\] actions"):
        wider.load_state_dict(saved)  # continuing or evaluating needs a match
    wider.load_state_dict(saved, widen=True)
    assert [agent.action_size for agent in wider.segment_agents] == [7, 7, 6]


def test_two_reward_channels_and_baselines_that_see_the_neighbours_actions():
    """Each critic's baseline gets the actions the segment's neighbours took
    at the same step, a missing neighbour's as zeros; both channels teach."""
    settings = AgentSettings.from_section(
        {
            "hidden_layers": [8],
            "critic_baselines": True,
            "value_normalization": True,
            "ppo": {"minibatch_size": 4},
        }
    )
    # Radius 1: each segment sees itself, the one ahead, then the one behind.
    neighbour_table = torch.tensor([[0, 3, 1], [1, 0, 2], [2, 1, 3]])
    agents = Agents(
        SEGMENTS,
        OBSERVATION_SIZE,
        WORLDS,
        WINDOW_STEPS,
        settings,
        SEED,
        [LEG_MOTORS] * SEGMENTS,
        LEG_MOTORS,
        reward_channel_count=2,
        neighbour_table=neighbour_table,
    )

    actions = agents.act(body_tensor(0, OBSERVATION_SIZE), training=True)

    middle = agents.segment_agents[1].rollout_storage.baseline_inputs[0]
    assert torch.equal(middle, torch.cat((actions[:, 0], actions[:, 2]), dim=-1))
    head = agents.segment_agents[0].rollout_storage.baseline_inputs[0]
    assert not head[:, :LEG_MOTORS].any()  # nothing ahead of the head
    assert torch.equal(head[:, LEG_MOTORS:], actions[:, 1])

    agents.record(
        body_tensor(100, 2),
        torch.zeros(WORLDS, dtype=torch.bool),
        torch.zeros(WORLDS, dtype=torch.bool),
        body_tensor(1, OBSERVATION_SIZE),
    )
    agents.act(body_tensor(1, OBSERVATION_SIZE), training=True)
    agents.record(
        body_tensor(101, 2),
        torch.zeros(WORLDS, dtype=torch.bool),
        torch.zeros(WORLDS, dtype=torch.bool),
        body_tensor(2, OBSERVATION_SIZE),
    )
    agents.update()

    learning = agents.diagnostics.learning
    assert learning.critic_explained_variances.shape == (SEGMENTS, 2)
    assert torch.isfinite(learning.baseline_explained_variances).all()


def test_the_last_worlds_may_act_with_the_mean_action_and_are_never_learned_from():
    """Of four worlds, the last two measure the policies: they get the mean
    action, and only the first two are stored, baselines included."""
    settings = AgentSettings.from_section(
        {"hidden_layers": [8], "critic_baselines": True, "ppo": {"minibatch_size": 4}}
    )
    agents = Agents(
        SEGMENTS,
        OBSERVATION_SIZE,
        WORLDS,
        WINDOW_STEPS,
        settings,
        SEED,
        [LEG_MOTORS] * SEGMENTS,
        LEG_MOTORS,
        neighbour_table=torch.tensor([[0, 3, 1], [1, 0, 2], [2, 1, 3]]),
        mean_action_world_count=2,
    )
    observations = body_tensor(0, OBSERVATION_SIZE)

    actions = agents.act(observations, training=True)

    mean_actions = agents.act(observations, training=False)
    assert torch.equal(actions[2:], mean_actions[2:])
    assert not torch.equal(actions[:2], mean_actions[:2])
    storage = agents.segment_agents[1].rollout_storage
    assert storage.baseline_inputs.shape[1] == 2
    agents.record(
        body_tensor(100),
        torch.zeros(WORLDS, dtype=torch.bool),
        torch.zeros(WORLDS, dtype=torch.bool),
        body_tensor(1, OBSERVATION_SIZE),
    )
    agents.act(body_tensor(1, OBSERVATION_SIZE), training=True)
    agents.record(
        body_tensor(101),
        torch.zeros(WORLDS, dtype=torch.bool),
        torch.zeros(WORLDS, dtype=torch.bool),
        body_tensor(2, OBSERVATION_SIZE),
    )
    agents.update()
    assert torch.isfinite(agents.diagnostics.learning.approximate_kl).all()


def test_followers_without_a_goal_learn_from_their_cost_alone():
    """The head learns from both channels, the followers from the second only:
    one critic each, their cost stored, and no goal critic reported."""
    agents = Agents(
        SEGMENTS,
        OBSERVATION_SIZE,
        WORLDS,
        WINDOW_STEPS,
        SETTINGS,
        SEED,
        [LEG_MOTORS] * SEGMENTS,
        LEG_MOTORS,
        reward_channel_count=2,
        segment_reward_channels=[(0, 1), (1,), (1,)],
    )
    assert [a.ppo.several_critics for a in agents.segment_agents] == [
        True,
        False,
        False,
    ]
    for step in range(WINDOW_STEPS):
        agents.act(body_tensor(step, OBSERVATION_SIZE), training=True)
        agents.record(
            body_tensor(100 + step, 2),
            torch.zeros(WORLDS, dtype=torch.bool),
            torch.zeros(WORLDS, dtype=torch.bool),
            body_tensor(step + 1, OBSERVATION_SIZE),
        )
    stored = agents.segment_agents[2].rollout_storage.rewards[0]
    torch.testing.assert_close(stored, body_tensor(100, 2)[:, 2, 1])

    agents.update()

    critics = agents.diagnostics.learning.critic_explained_variances
    assert torch.isnan(critics[1:, 0]).all()
    assert torch.isfinite(critics[0]).all() and torch.isfinite(critics[:, 1]).all()


def test_each_segment_weighs_its_own_channels_and_the_head_may_weigh_its_own():
    """With goal, trail and cost, the head weighs all three by its own
    weights and the followers their trail and cost by the common ones."""
    settings = AgentSettings.from_section(
        {
            "hidden_layers": [8],
            "ppo": {"minibatch_size": 4},
            "advantage_weights": [1.0, 2.0, 3.0],
            "head_advantage_weights": [1.0, 0.25, 1.0],
        }
    )
    agents = Agents(
        SEGMENTS,
        OBSERVATION_SIZE,
        WORLDS,
        WINDOW_STEPS,
        settings,
        SEED,
        [LEG_MOTORS] * SEGMENTS,
        LEG_MOTORS,
        reward_channel_count=3,
        segment_reward_channels=[(0, 1, 2), (1, 2), (1, 2)],
    )

    weights = [a.ppo.advantage_weights.tolist() for a in agents.segment_agents]
    assert weights == [[1.0, 0.25, 1.0], [2.0, 3.0], [2.0, 3.0]]


@pytest.mark.parametrize("device", DEVICES)
def test_baselines_give_zero_and_seeded_uniform_actions(device):
    observations = torch.ones(WORLDS, SEGMENTS, OBSERVATION_SIZE, device=device)

    # The joint action's width, here with the spine.
    width = LEG_MOTORS + 1
    zero = ZeroActionBaseline(width).act(observations, training=False)
    random = RandomActionBaseline(device, 3, width).act(observations, training=False)

    assert torch.equal(zero, torch.zeros(WORLDS, SEGMENTS, width, device=device))
    assert random.shape == (WORLDS, SEGMENTS, width)
    assert random.device.type == device
    assert -1 <= random.min() and random.max() < 1
    assert torch.equal(
        random, RandomActionBaseline(device, 3, width).act(observations, training=False)
    )
