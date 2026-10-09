"""Tests for one segment agent's rollout storage.

Advantage estimation itself is tested in RL_lib; these tests cover what the
storage adds: where each value is written, how the episode ends reach the
estimate, the order of the training batch, and reuse without reallocation.
"""

import pytest
import torch
from rl_lib.algorithms.policy_gradient import PPOActionSample

from centipede.agents.rollout_storage import RolloutStorage

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def sample_for_step(step: int, device="cpu") -> PPOActionSample:
    """A recognisable sample for two worlds, each value marked by its step."""
    return PPOActionSample(
        environment_action=torch.full((2, 1), -9.0, device=device),
        policy_action=torch.full((2, 1), 10.0 + step, device=device),
        log_probability=torch.full((2,), 20.0 + step, device=device),
        value=torch.full((2,), 0.5, device=device),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_steps_fill_rows_in_place_and_clearing_reuses_them(device):
    storage = RolloutStorage(2, 2, observation_size=3, device=device, action_size=1)
    memory = storage.observations.data_ptr()
    no = torch.zeros(2, dtype=torch.bool, device=device)

    for window in range(2):
        storage.clear()
        for step in range(2):
            storage.store_action(
                torch.full((2, 3), 100.0 * window + step, device=device),
                sample_for_step(step, device),
            )
            storage.store_outcome(
                torch.ones(2, device=device),
                no,
                no,
                torch.zeros(2, device=device),
                torch.full((2, 3), 100.0 * window + step + 1, device=device),
            )

    # The second window overwrote the first, in the same memory, on the device.
    assert storage.observations.data_ptr() == memory
    assert storage.observations.device.type == device
    assert storage.observations[:, :, 0].tolist() == [[100.0, 100.0], [101.0, 101.0]]
    assert storage.next_observations[:, :, 0].tolist() == [
        [101.0, 101.0],
        [102.0, 102.0],
    ]
    # Learning keeps the latent sample, never the action the environment got.
    assert storage.policy_actions[:, :, 0].tolist() == [[10.0, 10.0], [11.0, 11.0]]
    assert storage.log_probabilities.tolist() == [[20.0, 20.0], [21.0, 21.0]]


def test_training_batch_estimates_each_world_by_how_its_episodes_ended():
    storage = RolloutStorage(2, 2, observation_size=1, device="cpu", action_size=1)
    # World 0 continues past the window's end. World 1 is truncated on step 0,
    # bootstrapping from its final value 8, and terminates on step 1, whose
    # next value must be ignored.
    terminated = (torch.tensor([False, False]), torch.tensor([False, True]))
    truncated = (torch.tensor([False, True]), torch.tensor([False, False]))
    next_values = (torch.tensor([2.0, 8.0]), torch.tensor([4.0, 2.0]))
    for step in range(2):
        storage.store_action(torch.full((2, 1), float(step)), sample_for_step(step))
        storage.store_outcome(
            torch.ones(2),
            terminated[step],
            truncated[step],
            next_values[step],
            torch.full((2, 1), step + 1.0),
        )

    observations, policy_actions, log_probabilities, advantages, return_targets = (
        storage.training_batch(discount=0.5, gae_lambda=0.5)
    )

    # Rewards 1 and values 0.5 everywhere. World 0: TD errors 1.5 and 2.5, and
    # step 0 adds 0.5 * 0.5 * 2.5. World 1: step 0 ends its episode, so it
    # keeps only its own TD error, 1 + 0.5 * 8 - 0.5; step 1 has no future,
    # 1 - 0.5. Samples run step by step, world by world within a step.
    assert advantages.tolist() == [2.125, 4.5, 2.5, 0.5]
    assert return_targets.tolist() == [2.625, 5.0, 3.0, 1.0]
    assert observations.tolist() == [[0.0], [0.0], [1.0], [1.0]]
    assert policy_actions.tolist() == [[10.0], [10.0], [11.0], [11.0]]
    assert log_probabilities.tolist() == [20.0, 20.0, 21.0, 21.0]
