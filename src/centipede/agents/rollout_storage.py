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
        noise_scaled: bool = False,
        critic_count: int = 1,
        masked: bool = False,
    ) -> None:
        """Allocate every tensor once, filled with zeros; with ``noise_scaled``,
        also each step's noise scales, which PPO needs again when learning,
        and with ``masked`` each step's action mask. With several critics,
        ``critic_count``, the rewards and values end with one column each."""
        step_rows = (rollout_window_steps, world_count)
        value_rows = step_rows if critic_count == 1 else (*step_rows, critic_count)
        self.observations = torch.zeros(
            (*step_rows, observation_size), dtype=torch.float32, device=device
        )
        # The latent Gaussian samples, which PPO re-evaluates when learning.
        self.policy_actions = torch.zeros(
            (*step_rows, action_size), dtype=torch.float32, device=device
        )
        self.noise_scales = (
            torch.ones_like(self.policy_actions) if noise_scaled else None
        )
        self.action_masks = torch.ones_like(self.policy_actions) if masked else None
        self.log_probabilities = torch.zeros(
            step_rows, dtype=torch.float32, device=device
        )
        self.values = torch.zeros(value_rows, dtype=torch.float32, device=device)
        self.rewards = torch.zeros_like(self.values)
        # The observation right after each step, normalised, and its value.
        self.next_observations = torch.zeros_like(self.observations)
        self.next_values = torch.zeros_like(self.values)
        self.terminated = torch.zeros(step_rows, dtype=torch.bool, device=device)
        # Terminated or truncated: the next row starts a new episode.
        self.episode_ended = torch.zeros_like(self.terminated)
        # The row the current step is written to.
        self.step_index = 0

    def store_action(
        self,
        normalised_observations: torch.Tensor,
        sample: PPOActionSample,
        noise_scales: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> None:
        """Write the action side of the current step: what the agent saw and did,
        and the noise scales and action mask it was drawn with, if any."""
        row = self.step_index
        self.observations[row].copy_(normalised_observations)
        self.policy_actions[row].copy_(sample.policy_action)
        self.log_probabilities[row].copy_(sample.log_probability)
        self.values[row].copy_(sample.value)
        if noise_scales is not None:
            self.noise_scales[row].copy_(noise_scales)
        if action_mask is not None:
            self.action_masks[row].copy_(action_mask)

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
        into one dimension of ``T * W`` samples; with several critics, the
        advantages and return targets keep one column each. Each world's
        column is its own sequence of episodes, so a window cut needs no
        special case: its last row bootstraps from the next observation's
        value.
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
        samples = self.log_probabilities.numel()
        return (
            self.observations.reshape(-1, observation_size),
            self.policy_actions.reshape(-1, action_size),
            self.log_probabilities.reshape(-1),
            advantages.reshape(samples, *advantages.shape[2:]),
            return_targets.reshape(samples, *return_targets.shape[2:]),
        )

    def noise_scale_batch(self) -> torch.Tensor | None:
        """The window's noise scales in ``training_batch``'s sample order, or None."""
        if self.noise_scales is None:
            return None
        return self.noise_scales.reshape(-1, self.noise_scales.shape[-1])

    def action_mask_batch(self) -> torch.Tensor | None:
        """The window's action masks in ``training_batch``'s sample order, or None."""
        if self.action_masks is None:
            return None
        return self.action_masks.reshape(-1, self.action_masks.shape[-1])

    def clear(self) -> None:
        """Start a new window; the next steps overwrite the old rows."""
        self.step_index = 0
