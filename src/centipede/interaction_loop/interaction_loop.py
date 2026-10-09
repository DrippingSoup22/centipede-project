"""The interaction loop's front file: the exchange between agents and environment.

It passes observations to the agents, their joint action to the environment,
and the results back to the agents, and calculates nothing itself. Training
collects a window of steps in every world, then lets the agents learn; a window
ending never resets the environment. Evaluation uses the same exchange without
recording or learning: ``evaluate`` counts each world's first episode, and
``walk`` every episode of a fixed number of steps. ``diagnostics`` times them
and summarises the environment's diagnostics over each window. The experiment
creates the loop and calls ``train``, ``evaluate``, or ``walk``. See
docs/architecture.md.
"""

from collections.abc import Iterator

import torch

from centipede.agents.agents import Agents, RandomActionBaseline, ZeroActionBaseline
from centipede.environment.environment import Environment
from centipede.interaction_loop.diagnostics import LoopDiagnostics
from centipede.interaction_loop.settings import InteractionLoopSettings


class InteractionLoop:
    """Runs one environment with one group of agents, in training or evaluation."""

    def __init__(
        self,
        environment: Environment,
        agents: Agents,
        settings: InteractionLoopSettings,
    ) -> None:
        """Keep the two sides and prepare the loop's diagnostics."""
        self.environment = environment
        self.agents = agents
        self.settings = settings
        self.diagnostics = LoopDiagnostics(environment)

    def train(self, seed: int) -> Iterator[int]:
        """Run ``update_cycles`` windows, each followed by one update.

        The environment is reset once, with ``seed``; afterwards each world's
        episodes continue across windows. A generator: it yields each finished
        cycle's index, so the caller can read the diagnostics and save
        checkpoints between cycles without that time being measured.
        """
        observations = self.environment.reset(seed)
        for cycle_index in range(self.settings.update_cycles):
            with self.diagnostics.collecting():
                for _ in range(self.settings.rollout_window_steps):
                    actions = self.agents.act(observations, training=True)
                    observations, rewards, terminated, truncated, final_observations = (
                        self.environment.step(actions)
                    )
                    self.agents.record(
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        final_observations=final_observations,
                    )
                    self.diagnostics.step_taken()
            with self.diagnostics.learning():
                self.agents.update()
            yield cycle_index

    def evaluate(
        self,
        actor: Agents | ZeroActionBaseline | RandomActionBaseline,
        seed: int,
    ) -> None:
        """Run one episode in every world, from a reset with ``seed``.

        ``actor`` is the trained agents or a baseline; it acts without
        training, and nothing is stored or learned. Worlds that finish early
        start new episodes, as in training, but only each world's first
        episode counts in the diagnostics, which the caller reads afterwards.
        """
        observations = self.environment.reset(seed)
        first_episode_ended = torch.zeros(
            self.environment.world_count, dtype=torch.bool, device=observations.device
        )
        with self.diagnostics.collecting():
            while not first_episode_ended.all():
                actions = actor.act(observations, training=False)
                observations, _, terminated, truncated, _ = self.environment.step(
                    actions
                )
                self.diagnostics.step_taken(~first_episode_ended)
                first_episode_ended |= terminated | truncated

    def walk(
        self,
        actor: Agents | ZeroActionBaseline | RandomActionBaseline,
        seed: int,
        steps: int,
    ) -> None:
        """Run every world for ``steps`` steps, from a reset with ``seed``.

        As in ``evaluate``, ``actor`` acts without training. Every episode
        that ends within the steps counts in the diagnostics, however many a
        world goes through, so they show how many targets the actor reaches
        in a fixed time.
        """
        observations = self.environment.reset(seed)
        with self.diagnostics.collecting():
            for _ in range(steps):
                actions = actor.act(observations, training=False)
                observations, _, _, _, _ = self.environment.step(actions)
                self.diagnostics.step_taken()
