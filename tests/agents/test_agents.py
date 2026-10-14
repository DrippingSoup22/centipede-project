"""Tests for what the workspace branch adds to the agents."""

import torch

from centipede.agents.agents import Agents
from centipede.agents.rollout_storage import RolloutStorage
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


def test_the_head_learns_the_task_with_a_critic_of_its_own():
    """With a task critic, the head keeps its walk's and the task's rewards
    apart and the followers learn their sum; masked agents store each step's
    mask, and learn from the actions that took effect."""
    settings = AgentSettings.from_section(
        {"head_task_critic_weight": 0.2, "ppo": {"minibatch_size": 4}}
    )
    agents = Agents(2, 5, 3, 2, settings, 0, masked=True)
    head, follower = agents.segment_agents
    assert (head.critic_count, follower.critic_count) == (2, 1)
    mask = torch.zeros(3, 2, 6)
    mask[:, :, 0] = 1.0
    rewards = torch.tensor([1.0, 0.5]).expand(3, 2, 2)
    for _ in range(2):
        agents.act(torch.randn(3, 2, 5), training=True, action_mask=mask)
        ended = torch.zeros(3, dtype=torch.bool)
        agents.record(rewards, ended, ended, torch.zeros(3, 2, 5))

    assert torch.equal(head.rollout_storage.rewards[0], torch.tensor([[1.0, 0.5]] * 3))
    assert torch.equal(follower.rollout_storage.rewards[0], torch.full((3,), 1.5))
    assert torch.equal(follower.rollout_storage.action_masks[1], mask[:, 1])
    agents.update()
    learning = agents.diagnostics.learning
    assert not learning.task_explained_variance[0].isnan()
    assert learning.task_explained_variance[1].isnan()


def test_held_actions_explore_with_white_noise_and_the_neck_keeps_its_colour():
    """Masked agents: the tempo and curve columns are held for a turn, the
    head's last column, its neck, acts every step."""
    settings = AgentSettings.from_section({"exploration_noise_beta": 1.0})
    agents = Agents(
        2, 5, 3, 1, settings, 0, [4, 3], [[0], [0]],
        shape_columns=[[1, 2], [1, 2]], masked=True,
    )  # fmt: skip

    head, follower = (agent.ppo.policy for agent in agents.segment_agents)
    assert head.noise_beta.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert not follower.noise_is_colored


def test_an_arrival_ends_no_return_when_the_body_walks_on():
    """With returns continuing after arrivals, the arrival in world 0 ends
    neither the head's returns nor the follower's; world 1's restart cuts
    them, bootstrapped."""
    settings = AgentSettings.from_section(
        {"head_task_critic_weight": 0.2, "returns_continue_after_arrival": True}
    )
    agents = Agents(2, 5, 2, 1, settings, 0)
    agents.act(torch.zeros(2, 2, 5), training=True)
    arrived, restarted = torch.tensor([True, False]), torch.tensor([False, True])
    agents.record(torch.zeros(2, 2, 2), arrived, restarted, torch.zeros(2, 2, 5))

    for agent in agents.segment_agents:
        assert agent.rollout_storage.terminated[0].tolist() == [False, False]
        assert agent.rollout_storage.episode_ended[0].tolist() == [False, True]


def test_a_masked_agent_judges_each_decision_on_its_whole_turn():
    """One world, five steps: decisions on steps 0 and 3 (every action marked)
    held by the steps between (none marked); rewards 1, values 0. Each
    decision's advantage adds all its turn's discounted rewards, and the first
    takes lambda times the second's."""
    storage = RolloutStorage(5, 1, 2, "cpu", action_size=2, masked=True)
    storage.rewards.fill_(1.0)
    storage.action_masks.zero_()
    storage.action_masks[[0, 3]] = 1.0

    advantages = storage.training_batch(discount=0.9, gae_lambda=0.5)[3]

    second = 1 + 0.9
    first = 1 + 0.9 + 0.9**2 + 0.9**3 * 0.5 * second
    torch.testing.assert_close(advantages[[0, 3]], torch.tensor([first, second]))


def test_whole_turns_learn_one_sample_from_each_complete_turn():
    """Eight steps, rewards 1, values 0 but 2 after step 5. World 0's turns
    start on steps 0, 3 and 6, and the window's end cuts the last; world 1's
    on 0 and 4, and a restart after step 5 cuts the second (the turn from 6
    is cut by the window). Only the complete turns are samples: each judged
    on its whole turn, the turn before a cut one on its own."""
    storage = RolloutStorage(8, 2, 1, "cpu", action_size=2, masked=True)
    storage.observations[..., 0] = torch.arange(8.0)[:, None] * 10 + torch.arange(2)
    storage.rewards.fill_(1.0)
    storage.next_values[5] = 2.0
    storage.episode_ended[5, 1] = True
    storage.action_masks.zero_()
    storage.action_masks[[0, 3, 6], 0] = 1.0
    storage.action_masks[[0, 4, 6], 1] = 1.0

    observations, _, _, advantages, _, _ = storage.turn_batch(
        discount=0.9, gae_lambda=0.5
    )

    assert observations[:, 0].tolist() == [0.0, 1.0, 30.0]
    world_0_second = 1 + 0.9 + 0.9**2 * (1 + 0.9 * 2)
    world_0_first = 1 + 0.9 + 0.9**2 + 0.9**3 * 0.5 * world_0_second
    world_1_first = 1 + 0.9 + 0.9**2 + 0.9**3
    torch.testing.assert_close(
        advantages, torch.tensor([world_0_first, world_1_first, world_0_second])
    )
