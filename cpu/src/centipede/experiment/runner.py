"""Compose environments, independent learners, and rollouts for one training run."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path

import numpy as np
import torch

from centipede.environment import AgentID, CentipedeParallelEnv
from centipede.experiment.config import ExperimentConfig
from centipede.training.checkpoints import save_checkpoint
from centipede.training.environment_pool import (
    EnvironmentPool,
    ProcessEnvironmentPool,
    SerialEnvironmentPool,
)
from centipede.training.learners import build_learners
from centipede.training.rollout import RolloutCoordinator


@dataclass(frozen=True)
class TrainingResult:
    """Paths and progress returned after a completed training run."""

    run_dir: Path
    checkpoint_path: Path
    completed_update_cycles: int
    total_environment_transitions: int


def run_training(config: ExperimentConfig) -> TrainingResult:
    """Train for the configured windows and save synchronized learner checkpoints."""

    config.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.training_seed)

    pool: EnvironmentPool | None = None
    # Save near each quarter of the run. Short runs may have fewer distinct
    # checkpoint windows, but the completed final window is always included.
    checkpoint_cycles = {
        ceil(config.update_cycles * quarter / 4) for quarter in range(1, 5)
    }
    checkpoint_path = config.run_dir / f"checkpoint_{config.update_cycles}.pt"

    try:
        # Learners need only the model-owned agent IDs and spaces. Keep this
        # reference environment out of the replica pool and close it before any
        # worker processes are spawned.
        reference_environment = _create_environment(config)
        try:
            environment_seeds, learner_seeds = _derive_seeds(
                config.training_seed,
                config.environment_count,
                reference_environment.possible_agents,
            )
            learners = build_learners(
                reference_environment,
                config.learner_config,
                learner_seeds,
            )
        finally:
            reference_environment.close()

        pool = _build_environment_pool(config)
        coordinator = RolloutCoordinator(pool, learners, config.rollout_config)
        coordinator.reset(environment_seeds)

        total_transitions = 0
        window_transitions = (
            config.rollout_config.rollout_window_steps * config.environment_count
        )
        for completed_update_cycles in range(1, config.update_cycles + 1):
            last_filled = -1

            # Redraw this window's bar only when another twentieth is collected.
            # The callback receives actual completed environment transitions.
            def show_collection(
                collected: int, window_number: int = completed_update_cycles
            ) -> None:
                nonlocal last_filled
                filled = 20 * collected // window_transitions
                if filled == last_filled:
                    return
                last_filled = filled
                bar = "#" * filled + "-" * (20 - filled)
                print(
                    f"\rWindow {window_number}/{config.update_cycles} "
                    f"[{bar}] {collected}/{window_transitions} transitions",
                    end="",
                    flush=True,
                )

            show_collection(0)
            try:
                batches = coordinator.collect_window(on_transition=show_collection)
            finally:
                print(flush=True)

            results = coordinator.update_learners(batches=batches)

            # Numerical failure is a training outcome worth stopping and reporting;
            # the internal component contracts do not need revalidation here.
            for agent, updates in results.items():
                for update in updates:
                    if not all(
                        isfinite(value)
                        for value in (
                            update.actor_loss,
                            update.critic_loss,
                            update.entropy,
                            update.approximate_kl,
                            update.clip_fraction,
                        )
                    ):
                        raise FloatingPointError(
                            f"Non-finite PPO update for agent {agent} "
                            f"in window {completed_update_cycles}"
                        )

            total_transitions += window_transitions

            if completed_update_cycles in checkpoint_cycles:
                save_checkpoint(
                    path=config.run_dir / f"checkpoint_{completed_update_cycles}.pt",
                    learners=learners,
                    learner_config=config.learner_config,
                    rollout_config=config.rollout_config,
                    total_environment_transitions=total_transitions,
                    completed_updates=completed_update_cycles,
                )

            print(
                f"Update {completed_update_cycles}/{config.update_cycles} complete"
                f" | {total_transitions} total transitions",
                flush=True,
            )

        return TrainingResult(
            run_dir=config.run_dir,
            checkpoint_path=checkpoint_path,
            completed_update_cycles=config.update_cycles,
            total_environment_transitions=total_transitions,
        )

    finally:
        if pool is not None:
            pool.close()


def _build_environment_pool(config: ExperimentConfig) -> EnvironmentPool:
    """Create the configured serial reference or spawned process pool."""

    if config.worker_count > 1:
        return ProcessEnvironmentPool(
            model_path=config.model_path,
            max_episode_steps=config.max_episode_steps,
            environment_count=config.environment_count,
            worker_count=config.worker_count,
        )

    environments: list[CentipedeParallelEnv] = []
    try:
        # Serial replicas are constructed here so partial failures have one
        # cleanup owner before the completed list moves into the pool.
        for _ in range(config.environment_count):
            environments.append(_create_environment(config))
        return SerialEnvironmentPool(environments)
    except BaseException:
        for environment in environments:
            environment.close()
        raise


def _create_environment(config: ExperimentConfig) -> CentipedeParallelEnv:
    """Construct one environment from the resolved physical task settings."""

    return CentipedeParallelEnv(
        model_path=config.model_path,
        max_episode_steps=config.max_episode_steps,
    )


def _derive_seeds(
    training_seed: int,
    environment_count: int,
    agent_ids: Sequence[AgentID],
) -> tuple[list[int], dict[AgentID, int]]:
    """Derive stable, distinct replica and learner seeds from one run seed."""

    children = np.random.SeedSequence(training_seed).spawn(
        environment_count + len(agent_ids)
    )
    seeds = [int(child.generate_state(1)[0]) for child in children]
    environment_seeds = seeds[:environment_count]
    learner_seeds = {
        agent_id: seed
        for agent_id, seed in zip(agent_ids, seeds[environment_count:], strict=True)
    }
    return environment_seeds, learner_seeds
