"""The agents' front file: one independent segment agent per body segment.

The group looks like one agent from outside: it takes and returns tensors for
the whole body, with the worlds first and the segments second, and gives each
segment agent only its own segment's slice. The interaction loop uses ``act``,
``record``, and ``update``; the experiment creates the group, sets its
learning rate before each update, saves and loads it, and reads
``diagnostics``. The two baselines offer the same ``act`` for
evaluation. See docs/agents.md.
"""

from dataclasses import asdict

import torch
from rl_lib.algorithms.policy_gradient import PPOUpdateSummary

from centipede.agents.diagnostics import AgentDiagnostics
from centipede.agents.segment_agent import LEG_ACTION_COUNT, SegmentAgent
from centipede.agents.settings import AgentSettings


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
        segment_action_sizes: list[int] | None = None,
        tempo_columns: list[list[int]] | None = None,
        motor_columns: list[list[int]] | None = None,
    ):
        """Create one segment agent per segment, each with its own seed.

        Every (run seed, segment) pair gets a different agent seed.
        ``segment_action_sizes`` is the environment's: how many actions each
        segment takes, six each when not given. ``tempo_columns`` and
        ``motor_columns`` are the environment's too, which only the learning
        diagnostics need: for each segment, which actions set clock tempos
        (none when not given), and which action sets each of its motors (when
        not given, its first six actions its leg motors and a seventh its
        spine joint).
        """
        self.segment_count = segment_count
        self.observation_size = observation_size
        self.settings = settings
        self.segment_action_sizes = (
            segment_action_sizes or [LEG_ACTION_COUNT] * segment_count
        )
        self.action_size = max(self.segment_action_sizes)

        self.segment_agents: list[SegmentAgent] = [
            SegmentAgent(
                segment_index=segment_index,
                observation_size=observation_size,
                world_count=world_count,
                rollout_window_steps=rollout_window_steps,
                settings=settings,
                seed=seed * segment_count + segment_index,
                action_size=self.segment_action_sizes[segment_index],
            )
            for segment_index in range(segment_count)
        ]

        self.diagnostics = AgentDiagnostics(
            tempo_columns or [[] for _ in self.segment_action_sizes],
            motor_columns
            or [list(range(min(size, 7))) for size in self.segment_action_sizes],
            device=self.settings.device,
        )

    def act(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """The ``(W, N, action_size)`` joint action for the ``(W, N, ...)`` inputs.

        A segment with fewer actions than the widest gets zeros after its own,
        as padding the environment does not use.
        """
        actions: list[torch.Tensor] = []
        for segment_index, agent in enumerate(self.segment_agents):
            segment_actions = agent.act(observations[:, segment_index], training)
            padding = self.action_size - agent.action_size
            actions.append(torch.nn.functional.pad(segment_actions, (0, padding)))
        return torch.stack(actions, dim=1)

    def record(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        final_observations: torch.Tensor,
    ) -> None:
        """Give each agent its segment's rewards and final observations.

        The episode ends ``(W,)`` belong to the whole body, so every agent gets
        them in full.
        """
        for segment_index, agent in enumerate(self.segment_agents):
            agent.record(
                rewards[:, segment_index],
                terminated,
                truncated,
                final_observations[:, segment_index],
            )

    def set_learning_rate(self, learning_rate: float) -> None:
        """Every agent uses ``learning_rate`` from the next update on."""
        for agent in self.segment_agents:
            agent.set_learning_rate(learning_rate)
        self.diagnostics.learning.learning_rate.fill_(learning_rate)

    def set_action_std(self, action_std: float) -> None:
        """Every agent's actions get the spread ``action_std`` (a schedule's)."""
        for agent in self.segment_agents:
            agent.set_action_std(action_std)

    def update(self) -> None:
        """Let each agent learn from its own window, then fill the diagnostics."""
        summaries: list[PPOUpdateSummary] = []
        for agent in self.segment_agents:
            summaries.append(agent.update())
        self.diagnostics.record_update(summaries, self.segment_agents)

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
            "segment_action_sizes", [LEG_ACTION_COUNT] * state["segment_count"]
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

    def __init__(self, action_size: int = LEG_ACTION_COUNT) -> None:
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

    def __init__(
        self, device: torch.device | str, seed: int, action_size: int = LEG_ACTION_COUNT
    ) -> None:
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
