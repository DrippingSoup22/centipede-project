"""One segment's learner: its networks, PPO, observation normaliser, and storage.

A segment agent shares nothing with the others. The front file creates one per
segment and gives each its own slice of every tensor it passes on, with the
worlds as the first dimension. See docs/agents.md.
"""

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from rl_lib.algorithms.policy_gradient import PPO, PPOUpdateSummary
from rl_lib.networks import GaussianPolicyNetwork, StateValueNetwork
from rl_lib.normalization import ObservationNormalizer

from centipede.agents.rollout_storage import RolloutStorage
from centipede.agents.settings import AgentSettings

# The adaptive learning rate's step and limits (Rudin et al., CoRL 2021).
ADAPTIVE_STEP = 1.5
ADAPTIVE_LIMITS = (1e-5, 1e-2)


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
        action_size: int,
        reward_channel_count: int = 1,
        advantage_weights: tuple[float, ...] = (),
    ) -> None:
        """Build the agent's parts on its device; ``seed`` decides all its randomness.

        The seed sets the networks' starting weights and starts the PPO's own
        generator, which draws its actions and orders its minibatches.
        ``action_size`` is how many motors the segment commands: its leg
        motors, and with spine control the spine joint behind it. The agent
        has a critic for each of the ``reward_channel_count`` reward channels,
        weighed by ``advantage_weights`` (empty: 1 each).
        """
        self.segment_index = segment_index
        self.settings = settings
        self.action_size = action_size
        device = torch.device(settings.device)

        # The starting weights come from PyTorch's global generator. Seed it
        # here only, and restore it afterwards, so that nothing else changes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            actor_network = GaussianPolicyNetwork(
                observation_size=observation_size,
                action_size=action_size,
                hidden_sizes=settings.hidden_layers,
                initial_std=settings.initial_action_std,
                std_mode="gsde" if settings.state_dependent_noise_steps else "global",
                activation=settings.actor_activation,
            )
            critic_networks = [
                StateValueNetwork(observation_size, settings.hidden_layers)
                for _ in range(reward_channel_count)
            ]
        # Optimizers keep references to the parameters, so they are created
        # after the networks have moved to the device.
        actor_network.to(device)
        for network in critic_networks:
            network.to(device)
        several_critics = reward_channel_count > 1

        self.ppo = PPO(
            actor_network,
            _optimizer(actor_network.parameters(), settings),
            critic_networks if several_critics else critic_networks[0],
            _optimizer(
                [
                    parameter
                    for network in critic_networks
                    for parameter in network.parameters()
                ],
                settings,
            ),
            clip_ratio=settings.ppo.clip_ratio,
            entropy_coefficient=settings.ppo.entropy_coefficient,
            seed=seed,
            max_gradient_norm=settings.ppo.max_gradient_norm,
            action_low=[-1.0] * action_size,
            action_high=[1.0] * action_size,
            noise_beta=settings.exploration_noise_beta,
            exploration_matrix_steps=max(settings.state_dependent_noise_steps, 1),
            temporal_smoothness_coefficient=settings.ppo.temporal_smoothness_coefficient,
            advantage_weights=(advantage_weights or None) if several_critics else None,
            value_normalization=settings.value_normalization,
            value_normalization_horizon=settings.value_normalization_horizon,
        )
        # Scheduled spreads are set by the experiment, never learned.
        if settings.action_std_schedule != "learned":
            actor_network.log_std.requires_grad_(False)
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
            action_size=action_size,
            reward_channel_count=reward_channel_count,
        )
        # Completed updates, saved with the agent's state.
        self.update_count = 0

    def act(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """This segment's ``(W, action_size)`` actions for its observation slice.

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
        self.rollout_storage.store_outcome(
            rewards, terminated, truncated, next_values, normalised_observations
        )

    def set_learning_rate(self, learning_rate: float) -> None:
        """Use ``learning_rate`` in both optimizers from the next update on."""
        for optimizer in self._optimizers():
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

    def adapt_learning_rate(self, approximate_kl: float) -> None:
        """Rudin et al.'s rule (CoRL 2021, algorithm 1), once per update: the
        actor's learning rate falls by ADAPTIVE_STEP after an update whose KL
        was above twice ``desired_kl`` and rises by it after one below half,
        within ADAPTIVE_LIMITS. The critics keep theirs: the KL measures only
        how far the policy moved. Their rule acts after every minibatch; once
        per update, the GPU never waits in the middle of one."""
        desired_kl = self.settings.desired_kl
        low, high = ADAPTIVE_LIMITS
        for group in self.ppo.actor_optimizer.param_groups:
            if approximate_kl > 2 * desired_kl:
                group["lr"] = max(low, group["lr"] / ADAPTIVE_STEP)
            elif 0 < approximate_kl < desired_kl / 2:
                group["lr"] = min(high, group["lr"] * ADAPTIVE_STEP)

    def set_action_std(self, action_std: float) -> None:
        """Give every action the spread ``action_std`` from the next step on."""
        with torch.no_grad():
            self.ppo.actor_network.log_std.fill_(math.log(action_std))

    def update(self) -> PPOUpdateSummary:
        """Learn from this agent's full window, then start a new one."""
        ppo_settings = self.settings.ppo
        storage = self.rollout_storage
        summary = self.ppo.update(
            *storage.training_batch(
                discount=ppo_settings.discount, gae_lambda=ppo_settings.gae_lambda
            ),
            update_epochs=ppo_settings.update_epochs,
            minibatch_size=ppo_settings.minibatch_size,
            next_observations=storage.next_observations.reshape(
                -1, storage.next_observations.shape[-1]
            )
            if ppo_settings.temporal_smoothness_coefficient
            else None,
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

    def load_state_dict(self, state: Mapping[str, Any], widen: bool = False) -> None:
        """Restore what ``state_dict`` saved, onto this agent's device.

        Loading an optimizer also brings back the settings it was saved with;
        this agent's own weight decay and momentum replace them, and the
        experiment sets the learning rate before every update.

        With ``widen``, a state saved for fewer observations or actions, or
        for narrower hidden layers, is widened to this agent's sizes
        (``_widened``): the agent acts exactly as before and commands zero to
        its new motors; each new action's spread starts at
        ``initial_action_std``. The saved optimizers and random generator no
        longer fit, so the agent keeps its fresh ones, and its normaliser gives
        each new input a mean of 0 and a spread of 1. Every critic is widened,
        and the critics' value normaliser restored as saved.
        """
        ppo_state = state["ppo"]
        normaliser_state = state["observation_normaliser"]
        saved_critics = ppo_state["critic_network"]
        if not self.ppo.several_critics:
            saved_critics = [saved_critics]
        # Each network's saved state, fresh state, and the network.
        networks = [
            (ppo_state["actor_network"], self.ppo.actor_network),
            *zip(saved_critics, self.ppo.critic_networks, strict=True),
        ]
        if widen and any(
            saved.shape != fresh.shape
            for saved_state, network in networks
            for saved, fresh in zip(
                saved_state.values(), network.state_dict().values(), strict=True
            )
        ):
            for saved_state, network in networks:
                network.load_state_dict(_widened(saved_state, network.state_dict()))
            if "value_normalizer" in ppo_state:
                self.ppo.value_normalizer.load_state_dict(ppo_state["value_normalizer"])
            normaliser_state = _widened_normaliser(
                normaliser_state, self.observation_normaliser.observation_size
            )
        else:
            self.ppo.load_state_dict(ppo_state)
            for optimizer in self._optimizers():
                for group in optimizer.param_groups:
                    group["weight_decay"] = self.settings.weight_decay
                    if "momentum" in group:
                        group["momentum"] = self.settings.momentum
        self.observation_normaliser.load_state_dict(normaliser_state)
        self.update_count = int(state["update_count"])

    def _optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        return self.ppo.actor_optimizer, self.ppo.critic_optimizer


def _widened(
    saved: Mapping[str, torch.Tensor], fresh: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """A network's saved tensors, each in the leading corner of its fresh one.

    The widened network computes exactly what the saved one did. No saved
    unit takes anything from a new input or a new unit of the layer below, and
    the output layer's new actions start at zero. A new hidden unit keeps its
    fresh random weights from the layer below, so that it can learn, while
    nothing above takes from it yet: the function-preserving growth of Net2Net
    (Chen et al., ICLR 2016). Grown parts of the log spread keep their fresh
    value. A saved tensor larger than its fresh one fails.
    """
    # The last weight matrix belongs to the output layer.
    output_layer = [name for name, tensor in fresh.items() if tensor.dim() == 2][-1]
    output_prefix = output_layer.removesuffix("weight")
    widened = {}
    for name, fresh_tensor in fresh.items():
        saved_tensor = saved[name].to(fresh_tensor.device)
        if saved_tensor.shape == fresh_tensor.shape:
            widened[name] = saved_tensor
            continue
        if saved_tensor.dim() != fresh_tensor.dim() or any(
            saved_size > fresh_size
            for saved_size, fresh_size in zip(
                saved_tensor.shape, fresh_tensor.shape, strict=True
            )
        ):
            raise ValueError(
                f"A saved {name} of shape {tuple(saved_tensor.shape)} cannot be"
                f" widened to {tuple(fresh_tensor.shape)}"
            )
        if name.startswith(output_prefix):
            tensor = torch.zeros_like(fresh_tensor)
        else:
            tensor = fresh_tensor.clone()
            if tensor.dim() == 2:
                tensor[: saved_tensor.shape[0], saved_tensor.shape[1] :] = 0
        tensor[tuple(slice(0, size) for size in saved_tensor.shape)] = saved_tensor
        widened[name] = tensor
    return widened


def _widened_normaliser(
    saved: Mapping[str, Any], observation_size: int
) -> dict[str, Any]:
    """A saved normaliser state with new inputs at the end: mean 0, spread 1."""
    count = int(saved["observation_count"])
    added = observation_size - int(saved["observation_size"])
    mean = torch.as_tensor(saved["mean"], dtype=torch.float64)
    squared_deviation_sum = torch.as_tensor(
        saved["squared_deviation_sum"], dtype=torch.float64
    )
    return {
        **saved,
        "observation_size": observation_size,
        "mean": torch.cat((mean, mean.new_zeros(added))),
        "squared_deviation_sum": torch.cat(
            (squared_deviation_sum, squared_deviation_sum.new_full((added,), count))
        ),
    }


def _optimizer(
    parameters: Iterable[torch.nn.Parameter], settings: AgentSettings
) -> torch.optim.Optimizer:
    """The optimizer the settings name, for the given networks' parameters."""
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
