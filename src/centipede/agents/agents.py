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
from centipede.agents.segment_agent import ACTION_SIZE, SegmentAgent
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
    ):
        """Create one segment agent per segment, each with its own seed.

        Every (run seed, segment) pair gets a different agent seed.
        """
        self.segment_count = segment_count
        self.observation_size = observation_size
        self.settings = settings

        self.segment_agents: list[SegmentAgent] = [
            SegmentAgent(
                segment_index=segment_index,
                observation_size=observation_size,
                world_count=world_count,
                rollout_window_steps=rollout_window_steps,
                settings=settings,
                seed=seed * segment_count + segment_index,
            )
            for segment_index in range(segment_count)
        ]

        self.diagnostics = AgentDiagnostics(segment_count, device=self.settings.device)

    def act(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """The ``(W, N, 6)`` joint action for ``(W, N, observation_size)`` inputs."""
        actions: list[torch.Tensor] = []
        for segment_index, agent in enumerate(self.segment_agents):
            segment_observations = observations[:, segment_index]
            actions.append(agent.act(segment_observations, training))
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
            "settings": asdict(self.settings),
            "segment_agents": [agent.state_dict() for agent in self.segment_agents],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore every agent; a checkpoint made for another body is rejected."""
        saved_body = (state["segment_count"], state["observation_size"])
        this_body = (self.segment_count, self.observation_size)
        if saved_body != this_body:
            raise ValueError(
                f"The checkpoint is for {saved_body[0]} segments with "
                f"{saved_body[1]} observations each, but these agents have "
                f"{this_body[0]} segments with {this_body[1]}"
            )
        for agent, agent_state in zip(
            self.segment_agents, state["segment_agents"], strict=True
        ):
            agent.load_state_dict(agent_state)


class ZeroActionBaseline:
    """Evaluation baseline: every motor is always told zero."""

    def act(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """All-zero ``(W, N, 6)`` actions; ``training`` is ignored."""
        world_count, segment_count = observations.shape[:2]
        return torch.zeros(
            (world_count, segment_count, ACTION_SIZE), device=observations.device
        )


class RandomActionBaseline:
    """Evaluation baseline: uniformly random actions from its own generator."""

    def __init__(self, device: torch.device | str, seed: int) -> None:
        """The generator lives on ``device``, where the actions are drawn."""
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def act(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """Uniform ``(W, N, 6)`` actions in [-1, 1); ``training`` is ignored."""
        world_count, segment_count = observations.shape[:2]
        uniform = torch.rand(
            (world_count, segment_count, ACTION_SIZE),
            generator=self.generator,
            device=observations.device,
        )
        return uniform * 2 - 1
