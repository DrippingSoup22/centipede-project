"""The agents' front file: one independent segment agent per body segment.

The group looks like one agent from outside: it takes and returns tensors for
the whole body, with the worlds first and the segments second, and gives each
segment agent only its own segment's slice. The interaction loop uses ``act``,
``record``, and ``update``; the experiment creates the group, sets its
learning rate before each update (unless each agent adapts its own), saves
and loads it, and reads
``diagnostics``. The two baselines offer the same ``act`` for
evaluation. See docs/agents.md.
"""

from dataclasses import asdict

import torch
from rl_lib.algorithms.policy_gradient import PPOUpdateSummary

from centipede.agents.diagnostics import AgentDiagnostics
from centipede.agents.segment_agent import SegmentAgent
from centipede.agents.settings import AgentSettings
from centipede.settings_section import SettingsError

# A checkpoint saved before the spine could be commanded, by agents of legs of
# three joints, holds no action sizes: six per segment.
EARLY_CHECKPOINT_ACTION_SIZE = 6


class Agents:
    """``segment_count`` segment agents that share nothing but the body."""

    def __init__(
        self,
        segment_count: int,
        observation_size: int,
        world_count: int,
        rollout_window_steps: int,
        settings: AgentSettings,
        seed: int,
        segment_action_sizes: list[int],
        leg_motor_count: int,
        reward_channel_count: int = 1,
        neighbour_table: torch.Tensor | None = None,
        segment_reward_channels: list[tuple[int, ...]] | None = None,
        mean_action_world_count: int = 0,
    ):
        """Create one segment agent per segment, each with its own seed.

        Every (run seed, segment) pair gets a different agent seed.
        ``segment_action_sizes`` and ``leg_motor_count`` are the
        environment's: how many motors each segment commands, its leg motors
        first and then, with spine control, the spine joint behind it. So are
        ``reward_channel_count``; ``segment_reward_channels``, the channels
        each segment learns from, one critic each (every channel if not
        given; each weighs its critics by the weights of its channels in
        ``advantage_weights``, the head by ``head_advantage_weights`` when
        given); and ``neighbour_table``, the segments each one sees, itself
        first, from which the critics' baselines take the neighbours' actions.
        In training, the last ``mean_action_world_count`` worlds act with
        the policies' mean action and are neither stored nor learned from, so
        that they measure what the policies have learned while the others
        train.
        """
        self.segment_count = segment_count
        self.learning_world_count = world_count - mean_action_world_count
        self.observation_size = observation_size
        self.settings = settings
        self.segment_action_sizes = segment_action_sizes
        self.action_size = max(self.segment_action_sizes)
        for name in ("advantage_weights", "head_advantage_weights"):
            weights = getattr(settings, name)
            if weights and len(weights) != reward_channel_count:
                raise SettingsError(
                    f"[agents] {name} must have one weight for each of the"
                    f" environment's {reward_channel_count} reward channels, got"
                    f" {list(weights)}"
                )
        # Each segment's neighbours, without itself; a missing one points to
        # the row after the last segment, which holds no action.
        self.neighbours = (
            neighbour_table[:, 1:].to(settings.device)
            if settings.critic_baselines and neighbour_table is not None
            else None
        )
        baseline_input_size = (
            0
            if self.neighbours is None
            else self.neighbours.shape[1] * self.action_size
        )
        self.reward_channel_count = reward_channel_count
        self.segment_reward_channels = (
            segment_reward_channels
            or [tuple(range(reward_channel_count))] * segment_count
        )

        self.segment_agents: list[SegmentAgent] = [
            SegmentAgent(
                segment_index=segment_index,
                observation_size=observation_size,
                world_count=self.learning_world_count,
                rollout_window_steps=rollout_window_steps,
                settings=settings,
                seed=seed * segment_count + segment_index,
                action_size=self.segment_action_sizes[segment_index],
                reward_channel_count=len(self.segment_reward_channels[segment_index]),
                baseline_input_size=baseline_input_size,
                advantage_weights=self._weights_of(segment_index),
            )
            for segment_index in range(segment_count)
        ]

        self.diagnostics = AgentDiagnostics(
            segment_count,
            leg_motor_count,
            device=self.settings.device,
            reward_channel_count=reward_channel_count,
            segment_reward_channels=self.segment_reward_channels,
        )

    def _weights_of(self, segment_index: int) -> tuple[float, ...]:
        """The weights of the segment's critics: those of its channels, the
        head's own if it has them; empty for 1 each."""
        weights = self.settings.advantage_weights
        if segment_index == 0 and self.settings.head_advantage_weights:
            weights = self.settings.head_advantage_weights
        if not weights:
            return ()
        return tuple(weights[c] for c in self.segment_reward_channels[segment_index])

    def act(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """The ``(W, N, action_size)`` joint action for the ``(W, N, ...)`` inputs.

        A segment with fewer actions than the widest gets zeros after its own,
        as padding the environment does not use. In training the mean-action
        worlds act as in evaluation. In training with baselines, once every
        agent has acted, each stores its neighbours' actions in the learning
        worlds, nearest first as in the neighbour table, a missing one's all
        zeros.
        """
        learning = self.learning_world_count
        actions: list[torch.Tensor] = []
        for segment_index, agent in enumerate(self.segment_agents):
            segment_observations = observations[:, segment_index]
            segment_actions = agent.act(segment_observations[:learning], training)
            if learning < observations.shape[0]:
                segment_actions = torch.cat(
                    (
                        segment_actions,
                        agent.act(segment_observations[learning:], training=False),
                    )
                )
            padding = self.action_size - agent.action_size
            actions.append(torch.nn.functional.pad(segment_actions, (0, padding)))
        joint_action = torch.stack(actions, dim=1)
        if training and self.neighbours is not None:
            with_missing = torch.nn.functional.pad(
                joint_action[:learning], (0, 0, 0, 1)
            )
            neighbour_actions = with_missing[:, self.neighbours].flatten(start_dim=2)
            for segment_index, agent in enumerate(self.segment_agents):
                agent.store_baseline_inputs(neighbour_actions[:, segment_index])
        return joint_action

    def record(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        final_observations: torch.Tensor,
    ) -> None:
        """Give each agent its segment's rewards, the channels it learns from,
        and final observations, in the learning worlds.

        The episode ends ``(W,)`` belong to the whole body, so every agent gets
        them in full.
        """
        learning = self.learning_world_count
        for segment_index, agent in enumerate(self.segment_agents):
            segment_rewards = rewards[:learning, segment_index]
            channels = self.segment_reward_channels[segment_index]
            if len(channels) < self.reward_channel_count:
                segment_rewards = segment_rewards[:, list(channels)].squeeze(1)
            agent.record(
                segment_rewards,
                terminated[:learning],
                truncated[:learning],
                final_observations[:learning, segment_index],
            )

    def set_learning_rate(self, learning_rate: float) -> None:
        """Every agent uses ``learning_rate`` from the next update on."""
        for agent in self.segment_agents:
            agent.set_learning_rate(learning_rate)

    def set_action_std(self, action_std: float) -> None:
        """Every agent's actions get the spread ``action_std`` (a schedule's)."""
        for agent in self.segment_agents:
            agent.set_action_std(action_std)

    def update(self) -> None:
        """Let each agent learn from its own window, then fill the diagnostics;
        with the adaptive schedule, each agent then sets its next learning
        rate from its own update."""
        summaries: list[PPOUpdateSummary] = []
        for agent in self.segment_agents:
            summaries.append(agent.update())
        self.diagnostics.record_update(summaries, self.segment_agents)
        if self.settings.learning_rate_schedule == "adaptive":
            for agent, summary in zip(self.segment_agents, summaries, strict=True):
                agent.adapt_learning_rate(summary.approximate_kl)

    def state_dict(self) -> dict:
        """Every agent's state, the body it fits, and the settings as plain values."""
        return {
            "segment_count": self.segment_count,
            "observation_size": self.observation_size,
            "segment_action_sizes": self.segment_action_sizes,
            "settings": asdict(self.settings),
            "segment_agents": [agent.state_dict() for agent in self.segment_agents],
        }

    def load_state_dict(self, state: dict, widen: bool = False) -> None:
        """Restore every agent; a checkpoint made for another body is rejected.

        With ``widen``, as a new run that starts from another run's agents
        does, agents saved with fewer observations or actions, or narrower
        layers, are widened to these agents' sizes (see
        ``SegmentAgent.load_state_dict``). Checkpoints
        saved before the spine could be commanded hold six actions per segment.
        """
        saved_sizes = state.get(
            "segment_action_sizes",
            [EARLY_CHECKPOINT_ACTION_SIZE] * state["segment_count"],
        )
        saved_body = (state["segment_count"], state["observation_size"], saved_sizes)
        this_body = (
            self.segment_count,
            self.observation_size,
            self.segment_action_sizes,
        )
        fits = saved_body == this_body or (
            widen
            and saved_body[0] == this_body[0]
            and saved_body[1] <= this_body[1]
            and all(
                saved <= size
                for saved, size in zip(saved_sizes, this_body[2], strict=True)
            )
        )
        if not fits:
            raise ValueError(
                f"The checkpoint is for {saved_body[0]} segments with "
                f"{saved_body[1]} observations and {saved_sizes} actions, but these "
                f"agents have {this_body[0]} segments with {this_body[1]} and "
                f"{this_body[2]}"
            )
        for agent, agent_state in zip(
            self.segment_agents, state["segment_agents"], strict=True
        ):
            agent.load_state_dict(agent_state, widen=widen)


class ZeroActionBaseline:
    """Evaluation baseline: every motor is always told zero.

    ``action_size`` is the environment's joint action width.
    """

    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def act(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """All-zero ``(W, N, action_size)`` actions; ``training`` is ignored."""
        world_count, segment_count = observations.shape[:2]
        return torch.zeros(
            (world_count, segment_count, self.action_size), device=observations.device
        )


class RandomActionBaseline:
    """Evaluation baseline: uniformly random actions from its own generator.

    ``action_size`` is the environment's joint action width.
    """

    def __init__(self, device: torch.device | str, seed: int, action_size: int) -> None:
        """The generator lives on ``device``, where the actions are drawn."""
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.action_size = action_size

    def act(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """Uniform ``(W, N, action_size)`` actions in [-1, 1); ignores ``training``."""
        world_count, segment_count = observations.shape[:2]
        uniform = torch.rand(
            (world_count, segment_count, self.action_size),
            generator=self.generator,
            device=observations.device,
        )
        return uniform * 2 - 1
