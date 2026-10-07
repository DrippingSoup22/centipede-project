"""Tests for the interaction loop's front file.

How the timing and window summaries work is tested with the loop's
diagnostics; these tests cover what the front adds: the order and number of
calls between agents and environment, which first episodes evaluation counts,
and a short run with the real environment and agents.
"""

from typing import cast

import torch

from centipede.agents.agents import Agents, ZeroActionBaseline
from centipede.agents.settings import AgentSettings
from centipede.environment.diagnostics import EnvironmentDiagnostics
from centipede.environment.environment import Environment
from centipede.environment.settings import EnvironmentSettings
from centipede.environment.simulation.diagnostics import SimulationDiagnostics
from centipede.interaction_loop.interaction_loop import InteractionLoop
from centipede.interaction_loop.settings import InteractionLoopSettings

WORLDS, SEGMENTS = 2, 3


class FakeEnvironment:
    """Writes every call into ``calls``; episodes end at the given steps."""

    def __init__(
        self, calls: list, episode_ends: dict[int, list[bool]] | None = None
    ) -> None:
        self.calls = calls
        self.episode_ends = episode_ends or {}
        self.world_count = WORLDS
        self.diagnostics = EnvironmentDiagnostics(
            WORLDS,
            SEGMENTS,
            ["arrival"],
            "cpu",
            SimulationDiagnostics.allocate(WORLDS, 2, "cpu").facts,
        )
        self.steps_taken = 0

    def reset(self, seed):
        self.calls.append(("reset", seed))
        return torch.zeros(WORLDS, SEGMENTS, 1)

    def step(self, actions):
        self.calls.append("step")
        self.steps_taken += 1
        ended = torch.tensor(self.episode_ends.get(self.steps_taken, [False] * WORLDS))
        self.diagnostics.episode.episode_ended.copy_(ended)
        observations = torch.zeros(WORLDS, SEGMENTS, 1)
        rewards = torch.zeros(WORLDS, SEGMENTS)
        return (
            observations,
            rewards,
            ended,
            torch.zeros(WORLDS, dtype=torch.bool),
            observations,
        )


class FakeAgents:
    """Writes every call into ``calls``."""

    def __init__(self, calls: list) -> None:
        self.calls = calls

    def act(self, observations, training):
        self.calls.append(("act", training))
        return torch.zeros(WORLDS, SEGMENTS, 6)

    def record(self, **outcome):
        self.calls.append("record")

    def update(self):
        self.calls.append("update")


def fake_loop(calls, **environment_options) -> InteractionLoop:
    settings = InteractionLoopSettings(rollout_window_steps=3, update_cycles=2)
    # The stand-ins offer the same operations the loop uses.
    environment = cast(Environment, FakeEnvironment(calls, **environment_options))
    return InteractionLoop(environment, cast(Agents, FakeAgents(calls)), settings)


def test_training_resets_once_and_updates_after_every_window():
    calls = []
    loop = fake_loop(calls)

    assert list(loop.train(seed=5)) == [0, 1]

    window = [("act", True), "step", "record"] * 3 + ["update"]
    assert calls == [("reset", 5)] + window * 2
    assert loop.diagnostics.timing.learning_seconds > 0


def test_evaluation_runs_until_every_first_episode_ends_and_counts_only_those():
    # World 0 ends on steps 1 and 2; its second end must not count.
    calls = []
    loop = fake_loop(calls, episode_ends={1: [True, False], 2: [True, True]})

    loop.evaluate(cast(Agents, FakeAgents(calls)), seed=5)

    assert calls == [("reset", 5)] + [("act", False), "step"] * 2
    assert loop.diagnostics.episode_window.result()["episode_ended"] == 2


def test_a_short_run_with_the_real_environment_and_agents_is_finite():
    environment = Environment(
        EnvironmentSettings.from_section(
            {
                "max_episode_steps": 3,
                "simulation": {
                    "model_path": "models/assembly_v2.xml",
                    "backend": "cpu",
                    "world_count": WORLDS,
                },
            }
        )
    )
    settings = InteractionLoopSettings(rollout_window_steps=2, update_cycles=2)
    agents = Agents(
        environment.segment_count,
        environment.observation_size,
        WORLDS,
        settings.rollout_window_steps,
        AgentSettings.from_section(
            {"hidden_layers": [8], "ppo": {"minibatch_size": 4}}
        ),
        seed=1,
    )
    loop = InteractionLoop(environment, agents, settings)

    for _ in loop.train(seed=1):
        pass
    training_steps = loop.diagnostics.step_window.result()
    loop.evaluate(ZeroActionBaseline(), seed=2)
    evaluation_episodes = loop.diagnostics.episode_window.result()

    assert all(agent.update_count == 2 for agent in agents.segment_agents)
    for agent in agents.segment_agents:
        for parameter in agent.ppo.actor_network.parameters():
            assert parameter.isfinite().all()
    for summary in training_steps.values():
        assert summary.isfinite().all()
    # Every world's first episode reaches the three-step limit.
    assert evaluation_episodes["episode_ended"] == WORLDS
    assert evaluation_episodes["length_steps"] == 3
    assert loop.diagnostics.timing.collecting_seconds > 0
