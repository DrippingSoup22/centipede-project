"""Resolved settings shared by Centipede training and evaluation."""

from dataclasses import dataclass
from pathlib import Path

from centipede.training.learners import LearnerConfig
from centipede.training.rollout import RolloutConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved inputs needed to construct and run one training experiment."""

    model_path: Path
    run_dir: Path
    training_seed: int
    environment_count: int
    worker_count: int
    update_cycles: int
    max_episode_steps: int
    evaluation_seeds: tuple[int, ...]
    learner_config: LearnerConfig
    rollout_config: RolloutConfig
