"""One segment's learner: its networks, PPO, observation normaliser, and storage.

A segment agent shares nothing with the others. The front file creates one per
segment and gives each its own slice of every tensor it passes on, with the
worlds as the first dimension. See docs/agents.md.
"""

from collections.abc import Mapping
from typing import Any

import torch
from rl_lib.algorithms.policy_gradient import PPO, PPOUpdateSummary
from rl_lib.networks import GaussianPolicyNetwork, StateValueNetwork
from rl_lib.normalization import ObservationNormalizer

from centipede.agents.rollout_storage import RolloutStorage
from centipede.agents.settings import AgentSettings

# Each segment drives the six motors of its two legs.
ACTION_SIZE = 6


class SegmentAgent:
    """PPO for one segment, acting in every world at once."""

    def __init__(
        self,
        segment_index: int,
        world_count: int,
        observation_size: int,
        rollout_window_steps: int,
        settings: AgentSettings,
        seed: int,
    ) -> None:
        """Build the agent's parts on its device; ``seed`` decides all its randomness.

        The seed sets the networks' starting weights and starts the PPO's own
        generator, which draws its actions and orders its minibatches.
        """
        self.segment_index = segment_index
        self.settings = settings
        device = torch.device(settings.device)

        # The starting weights come from PyTorch's global generator. Seed it
        # here only, and restore it afterwards, so that nothing else changes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            actor_network = GaussianPolicyNetwork(
                observation_size=observation_size,
                action_size=ACTION_SIZE,
                hidden_sizes=settings.hidden_layers,
                initial_std=settings.initial_action_std,
                std_mode="global",
            )
            critic_network = StateValueNetwork(
                observation_size=observation_size,
                hidden_sizes=settings.hidden_layers,
            )
        # Optimizers keep references to the parameters, so they are created
        # after the networks have moved to the device.
        actor_network.to(device)
        critic_network.to(device)

        self.ppo = PPO(
            actor_network,
            _optimizer(actor_network, settings),
            critic_network,
            _optimizer(critic_network, settings),
            clip_ratio=settings.ppo.clip_ratio,
            entropy_coefficient=settings.ppo.entropy_coefficient,
            seed=seed,
            max_gradient_norm=settings.ppo.max_gradient_norm,
            action_low=[-1.0] * ACTION_SIZE,
            action_high=[1.0] * ACTION_SIZE,
        )
        self.observation_normaliser = ObservationNormalizer(
            observation_size,
            "running",
            epsilon=settings.normaliser_epsilon,
            clip_limit=settings.observation_clip,
            device=device,
        )
        self.rollout_storage = RolloutStorage(
            rollout_window_steps,
            world_count,
            observation_size,
            device=device,
            action_size=ACTION_SIZE,
        )
        # Completed updates, saved with the agent's state.
        self.update_count = 0

    def act(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """This segment's ``(W, 6)`` actions for its ``(W, observation_size)`` slice.

        In training the observations first update the normaliser, and the step's
        action side is stored for learning. Otherwise nothing changes, and each
        world gets the policy's mean action.
        """
        normalised_observations = self.observation_normaliser.normalize(
            observations, update_statistics=training
        )
        if not training:
            return self.ppo.select_action(normalised_observations, deterministic=True)
        sample = self.ppo.sample_action(normalised_observations)
        self.rollout_storage.store_action(normalised_observations, sample)
        return sample.environment_action

    def record(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        final_observations: torch.Tensor,
    ) -> None:
        """Store the outcome side of the step just taken, each tensor ``(W, ...)``.

        The critic's value of the observation right after the step serves every
        way a stretch can end; see docs/agents.md.
        """
        normalised_observations = self.observation_normaliser.normalize(
            final_observations, update_statistics=False
        )
        next_values = self.ppo.state_value(normalised_observations)
        self.rollout_storage.store_outcome(rewards, terminated, truncated, next_values)

    def set_learning_rate(self, learning_rate: float) -> None:
        """Use ``learning_rate`` in both optimizers from the next update on."""
        for optimizer in self._optimizers():
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

    def update(self) -> PPOUpdateSummary:
        """Learn from this agent's full window, then start a new one."""
        ppo_settings = self.settings.ppo
        summary = self.ppo.update(
            *self.rollout_storage.training_batch(
                discount=ppo_settings.discount, gae_lambda=ppo_settings.gae_lambda
            ),
            update_epochs=ppo_settings.update_epochs,
            minibatch_size=ppo_settings.minibatch_size,
        )
        self.rollout_storage.clear()
        self.update_count += 1
        return summary

    def state_dict(self) -> dict[str, Any]:
        """Everything learning changes, for checkpoints; no collected data."""
        return {
            "ppo": self.ppo.state_dict(),
            "observation_normaliser": self.observation_normaliser.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore what ``state_dict`` saved, onto this agent's device.

        Loading an optimizer also brings back the settings it was saved with;
        this agent's own weight decay and momentum replace them, and the
        experiment sets the learning rate before every update.
        """
        self.ppo.load_state_dict(state["ppo"])
        for optimizer in self._optimizers():
            for group in optimizer.param_groups:
                group["weight_decay"] = self.settings.weight_decay
                if "momentum" in group:
                    group["momentum"] = self.settings.momentum
        self.observation_normaliser.load_state_dict(state["observation_normaliser"])
        self.update_count = int(state["update_count"])

    def _optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        return self.ppo.actor_optimizer, self.ppo.critic_optimizer


def _optimizer(
    network: torch.nn.Module, settings: AgentSettings
) -> torch.optim.Optimizer:
    """The optimizer the settings name, for one network's parameters."""
    parameters = network.parameters()
    if settings.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=settings.learning_rate,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
        )
    optimizer_class = (
        torch.optim.AdamW if settings.optimizer == "adamw" else torch.optim.Adam
    )
    return optimizer_class(
        parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
