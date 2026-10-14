"""One segment's learner: its networks, PPO, observation normaliser, and storage.

A segment agent shares nothing with the others. The front file creates one per
segment and gives each its own slice of every tensor it passes on, with the
worlds as the first dimension. See docs/agents.md.
"""

import math
from collections.abc import Mapping
from typing import Any

import torch
from rl_lib.algorithms.policy_gradient import PPO, PPOUpdateSummary
from rl_lib.networks import GaussianPolicyNetwork, StateValueNetwork
from rl_lib.normalization import ObservationNormalizer

from centipede.agents.rollout_storage import RolloutStorage
from centipede.agents.settings import AgentSettings

# Each segment drives the six motors of its two legs; with spine control, the
# spine joint behind it is a seventh action.
LEG_ACTION_COUNT = 6
# The least a follower's tempo noise is scaled to by its gap to the segment
# ahead (``follower_tempo_noise_gap_octaves``): about the noise at which a
# wave holds, so that a locked tempo still explores a little.
MIN_TEMPO_NOISE_SCALE = 0.03


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
        action_size: int = LEG_ACTION_COUNT,
        shape_columns: list[int] | None = None,
        tempo_columns: list[int] | None = None,
        tempo_gap_column: int | None = None,
        task_critic_weight: float = 0.0,
        masked: bool = False,
        whole_turns: bool = False,
    ) -> None:
        """Build the agent's parts on its device; ``seed`` decides all its randomness.

        The seed sets the networks' starting weights and starts the PPO's own
        generator, which draws its actions and orders its minibatches.
        ``action_size`` is how many motors the segment commands, and
        ``shape_columns`` the actions that draw its legs' curves, which start
        at random values with ``initial_action_offset_spread``. With
        ``tempo_gap_column``, the observation that holds the segment's tempo
        gap to the segment ahead in octaves, the noise on its
        ``tempo_columns`` follows that gap
        (``follower_tempo_noise_gap_octaves``). With ``task_critic_weight``
        above 0 the agent learns its walk's and the task's rewards with a
        critic each, the task's advantage weighing this much against the
        walk's 1 (``head_task_critic_weight``); otherwise their sum, with one.
        With ``masked``, ``act`` takes the environment's action mask, and the
        agent learns only from the actions that took effect, each decision
        judged on all the steps it held (``RolloutStorage.training_batch``);
        its held actions, ``tempo_columns`` and ``shape_columns``, explore
        with white noise. With ``whole_turns`` too (every action is held for
        a turn), it learns from each complete turn as one sample
        (``RolloutStorage.turn_batch``), and every action explores with white
        noise.
        """
        self.segment_index = segment_index
        self.settings = settings
        self.whole_turns = whole_turns
        self.action_size = action_size
        self.tempo_columns = tempo_columns
        self.tempo_gap_column = tempo_gap_column
        # The mean scale of the tempo noise over the last window, for the
        # learning diagnostics: 1 for an agent whose noise is not scaled.
        self.tempo_noise_scale = torch.ones((), device=settings.device)
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
                std_mode="global",
                activation=settings.hidden_activation,
                mean_output_scale=settings.actor_last_layer_scale,
            )
            critic_networks = [
                StateValueNetwork(
                    observation_size=observation_size,
                    hidden_sizes=settings.hidden_layers,
                    activation=settings.hidden_activation,
                )
                for _ in range(2 if task_critic_weight else 1)
            ]
            spread = settings.initial_action_offset_spread
            if spread and shape_columns:
                # The actor's mean passes through tanh into the action's
                # range, so a starting action u needs a bias of atanh(u).
                offsets = (2 * torch.rand(len(shape_columns)) - 1) * spread
                with torch.no_grad():
                    actor_network.mean_network[-1].bias[shape_columns] += torch.atanh(
                        offsets
                    )
        # Optimizers keep references to the parameters, so they are created
        # after the networks have moved to the device.
        actor_network.to(device)
        critic = torch.nn.ModuleList(critic_networks).to(device)
        self.critic_count = len(critic_networks)
        # Held for a turn, the tempo and the curves explore with white noise,
        # a fresh trial each turn; the actions that act every step keep the
        # noise's colour.
        noise_beta = settings.exploration_noise_beta
        if whole_turns:
            noise_beta = 0.0
        elif masked:
            held = set(tempo_columns or ()) | set(shape_columns or ())
            noise_beta = [
                0.0 if column in held else noise_beta for column in range(action_size)
            ]

        self.ppo = PPO(
            actor_network,
            _optimizer(actor_network, settings),
            critic_networks if self.critic_count > 1 else critic_networks[0],
            _optimizer(critic, settings),
            clip_ratio=settings.ppo.clip_ratio,
            entropy_coefficient=settings.ppo.entropy_coefficient,
            seed=seed,
            max_gradient_norm=settings.ppo.max_gradient_norm,
            action_low=[-1.0] * action_size,
            action_high=[1.0] * action_size,
            noise_beta=noise_beta,
            noise_sequence_steps=settings.exploration_noise_sequence_steps,
            temporal_smoothness_coefficient=settings.ppo.temporal_smoothness_coefficient,
            advantage_weights=[1.0, task_critic_weight]
            if self.critic_count > 1
            else None,
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
            noise_scaled=tempo_gap_column is not None,
            critic_count=self.critic_count,
            masked=masked,
        )
        # Each world's noise scales for the step being taken, 1 but for the
        # tempo actions.
        self.noise_scales = (
            torch.ones((world_count, action_size), device=device)
            if tempo_gap_column is not None
            else None
        )
        # Completed updates, saved with the agent's state.
        self.update_count = 0

    def act(
        self,
        observations: torch.Tensor,
        training: bool,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This segment's ``(W, action_size)`` actions for its observation slice.

        In training the observations first update the normaliser, and the step's
        action side is stored for learning, with ``action_mask``, its slice of
        the environment's, when the agent is masked. Otherwise nothing
        changes, and each world gets the policy's mean action.
        """
        normalised_observations = self.observation_normaliser.normalize(
            observations, update_statistics=training
        )
        if not training:
            return self.ppo.select_action(normalised_observations, deterministic=True)
        if self.noise_scales is not None:
            # The tempo explores in proportion to the gap, read in octaves
            # before normalising.
            gap = observations[:, self.tempo_gap_column].abs()
            scale = (gap / self.settings.follower_tempo_noise_gap_octaves).clamp(
                MIN_TEMPO_NOISE_SCALE, 1.0
            )
            self.noise_scales[:, self.tempo_columns] = scale[:, None]
        sample = self.ppo.sample_action(
            normalised_observations, self.noise_scales, action_mask
        )
        self.rollout_storage.store_action(
            normalised_observations, sample, self.noise_scales, action_mask
        )
        return sample.environment_action

    def record(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        final_observations: torch.Tensor,
    ) -> None:
        """Store the outcome side of the step just taken, each tensor ``(W, ...)``.

        ``rewards`` ``(W, 2)`` holds the walk's and the task's: learned each
        with its own critic, or their sum with one. The critic's value of the
        observation right after the step serves every way a stretch can end;
        see docs/agents.md. With ``returns_continue_after_arrival`` an
        arrival ends no return: the body walks on toward a new target, and
        only a restart cuts the returns.
        """
        normalised_observations = self.observation_normaliser.normalize(
            final_observations, update_statistics=False
        )
        next_values = self.ppo.state_value(normalised_observations)
        if self.critic_count == 1:
            rewards = rewards.sum(dim=-1)
        if self.settings.returns_continue_after_arrival:
            terminated = torch.zeros_like(terminated)
        self.rollout_storage.store_outcome(
            rewards, terminated, truncated, next_values, normalised_observations
        )

    def set_learning_rate(self, learning_rate: float) -> None:
        """Use ``learning_rate`` in both optimizers from the next update on."""
        for optimizer in self._optimizers():
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

    def set_action_std(self, action_std: float) -> None:
        """Give every action the spread ``action_std`` from the next step on."""
        with torch.no_grad():
            self.ppo.actor_network.log_std.fill_(math.log(action_std))

    def update(self) -> PPOUpdateSummary:
        """Learn from this agent's full window, then start a new one."""
        ppo_settings = self.settings.ppo
        storage = self.rollout_storage
        if self.whole_turns:
            *turns, noise_scales = storage.turn_batch(
                discount=ppo_settings.discount, gae_lambda=ppo_settings.gae_lambda
            )
            summary = self.ppo.update(
                *turns,
                update_epochs=ppo_settings.update_epochs,
                minibatch_size=ppo_settings.minibatch_size,
                noise_scales=noise_scales,
            )
        else:
            summary = self.ppo.update(
                *storage.training_batch(
                    discount=ppo_settings.discount,
                    gae_lambda=ppo_settings.gae_lambda,
                ),
                update_epochs=ppo_settings.update_epochs,
                minibatch_size=ppo_settings.minibatch_size,
                next_observations=storage.next_observations.reshape(
                    -1, storage.next_observations.shape[-1]
                )
                if ppo_settings.temporal_smoothness_coefficient
                else None,
                noise_scales=storage.noise_scale_batch(),
                action_masks=storage.action_mask_batch(),
            )
        if storage.noise_scales is not None:
            self.tempo_noise_scale = storage.noise_scales[
                ..., self.tempo_columns
            ].mean()
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
        each new input a mean of 0 and a spread of 1.
        """
        ppo_state = state["ppo"]
        normaliser_state = state["observation_normaliser"]
        fresh_state = self.ppo.state_dict()
        # The networks' saved and fresh states, several critics' each.
        networks = [(ppo_state["actor_network"], fresh_state["actor_network"])] + list(
            zip(
                _as_list(ppo_state["critic_network"]),
                _as_list(fresh_state["critic_network"]),
                strict=True,
            )
        )
        if widen and any(
            saved.shape != fresh.shape
            for saved_state, fresh_state_part in networks
            for saved, fresh in zip(
                saved_state.values(), fresh_state_part.values(), strict=True
            )
        ):
            for network, (saved_state, fresh_state_part) in zip(
                [self.ppo.actor_network, *self.ppo.critic_networks],
                networks,
                strict=True,
            ):
                network.load_state_dict(_widened(saved_state, fresh_state_part))
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


def _as_list(critic_state: Any) -> list:
    """A saved critic state as a list: several critics' already are one."""
    return critic_state if isinstance(critic_state, list) else [critic_state]


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
