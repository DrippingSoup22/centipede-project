"""One segment agent's collected data for one window, kept on its device.

Every tensor has the window's steps as rows and the worlds as columns, the
layout RL_lib's advantage estimation works on. The storage is allocated once
and filled in place, one row per step: the action side when the agent acts, the
outcome side when the step's results are recorded. See docs/agents.md.
"""

import torch
from rl_lib.algorithms.policy_gradient import (
    PPOActionSample,
    generalized_advantage_estimates,
)


class RolloutStorage:
    """Rows of ``(W, ...)`` values for ``rollout_window_steps`` steps."""

    def __init__(
        self,
        rollout_window_steps: int,
        world_count: int,
        observation_size: int,
        device: torch.device | str,
        action_size: int = 6,
    ) -> None:
        """Allocate every tensor once, filled with zeros."""
        step_rows = (rollout_window_steps, world_count)
        self.observations = torch.zeros(
            (*step_rows, observation_size), dtype=torch.float32, device=device
        )
        # The latent Gaussian samples, which PPO re-evaluates when learning.
        self.policy_actions = torch.zeros(
            (*step_rows, action_size), dtype=torch.float32, device=device
        )
        self.log_probabilities = torch.zeros(
            step_rows, dtype=torch.float32, device=device
        )
        self.values = torch.zeros_like(self.log_probabilities)
        self.rewards = torch.zeros_like(self.log_probabilities)
        # The observation right after each step, normalised, and its value.
        self.next_observations = torch.zeros_like(self.observations)
        self.next_values = torch.zeros_like(self.log_probabilities)
        self.terminated = torch.zeros(step_rows, dtype=torch.bool, device=device)
        # Terminated or truncated: the next row starts a new episode.
        self.episode_ended = torch.zeros_like(self.terminated)
        # The row the current step is written to.
        self.step_index = 0

    def store_action(
        self, normalised_observations: torch.Tensor, sample: PPOActionSample
    ) -> None:
        """Write the action side of the current step: what the agent saw and did."""
        row = self.step_index
        self.observations[row].copy_(normalised_observations)
        self.policy_actions[row].copy_(sample.policy_action)
        self.log_probabilities[row].copy_(sample.log_probability)
        self.values[row].copy_(sample.value)

    def store_outcome(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_values: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> None:
        """Write the outcome side of the current step, then move to the next row."""
        row = self.step_index
        self.rewards[row].copy_(rewards)
        self.terminated[row].copy_(terminated)
        torch.logical_or(terminated, truncated, out=self.episode_ended[row])
        self.next_values[row].copy_(next_values)
        self.next_observations[row].copy_(next_observations)
        self.step_index += 1

    def training_batch(
        self, discount: float, gae_lambda: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The full window as one batch, in the order ``PPO.update`` takes it.

        Returns the observations, policy actions, log-probabilities,
        advantages, and return targets, each with steps and worlds flattened
        into one dimension of ``T * W`` samples. Each world's column is its own
        sequence of episodes, so a window cut needs no special case: its last
        row bootstraps from the next observation's value.
        """
        advantages, return_targets = generalized_advantage_estimates(
            self.rewards,
            self.values,
            self.next_values,
            terminated=self.terminated,
            episode_ended=self.episode_ended,
            discount=discount,
            gae_lambda=gae_lambda,
        )
        observation_size = self.observations.shape[-1]
        action_size = self.policy_actions.shape[-1]
        return (
            self.observations.reshape(-1, observation_size),
            self.policy_actions.reshape(-1, action_size),
            self.log_probabilities.reshape(-1),
            advantages.reshape(-1),
            return_targets.reshape(-1),
        )

    def clear(self) -> None:
        """Start a new window; the next steps overwrite the old rows."""
        self.step_index = 0
