"""Focused checks for the Stage 5 training application flow."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, call

import pytest
from rl_lib.data import PPOUpdateResult

from centipede.experiment import runner
from centipede.experiment.config import ExperimentConfig
from centipede.training.learners import LearnerConfig
from centipede.training.rollout import RolloutConfig

CPU_ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path) -> ExperimentConfig:
    """Keep the runner tests small without changing the smoke profile."""
    return ExperimentConfig(
        model_path=tmp_path / "model.xml",
        run_dir=tmp_path / "run",
        training_seed=2026,
        environment_count=2,
        worker_count=1,
        update_cycles=4,
        max_episode_steps=2_048,
        evaluation_seeds=(30001, 30002),
        learner_config=LearnerConfig(),
        rollout_config=RolloutConfig(rollout_window_steps=3),
    )


def test_runner_updates_all_windows_and_saves_quarter_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Use one learner set across replicas and count physics transitions once."""
    config = _config(tmp_path)
    reference = Mock(possible_agents=[0])
    environments = [Mock(possible_agents=[0]), Mock(possible_agents=[0])]
    environment_factory = Mock(side_effect=[reference, *environments])
    learners = {0: object()}
    learner_factory = Mock(return_value=learners)
    pool = Mock()
    pool_factory = Mock(return_value=pool)
    coordinator = Mock()

    def collect_six(*, on_transition: Callable[[int], None]) -> dict[int, object]:
        for collected in range(1, 7):
            on_transition(collected)
        return {0: object()}

    coordinator.collect_window.side_effect = collect_six
    coordinator.update_learners.return_value = {
        0: (PPOUpdateResult(actor_loss=0.1, critic_loss=0.2, entropy=0.3),)
    }
    coordinator_factory = Mock(return_value=coordinator)
    checkpoint_writer = Mock()

    monkeypatch.setattr(runner, "CentipedeParallelEnv", environment_factory)
    monkeypatch.setattr(runner, "build_learners", learner_factory)
    monkeypatch.setattr(runner, "SerialEnvironmentPool", pool_factory)
    monkeypatch.setattr(runner, "RolloutCoordinator", coordinator_factory)
    monkeypatch.setattr(runner, "save_checkpoint", checkpoint_writer)

    result = runner.run_training(config)

    assert environment_factory.call_count == 3
    assert environment_factory.call_args_list == [
        call(model_path=config.model_path, max_episode_steps=2_048),
        call(model_path=config.model_path, max_episode_steps=2_048),
        call(model_path=config.model_path, max_episode_steps=2_048),
    ]
    assert learner_factory.call_count == 1
    assert learner_factory.call_args.args[0] is reference
    reference.close.assert_called_once_with()
    pool_factory.assert_called_once_with(environments)
    assert coordinator_factory.call_args.args[:2] == (pool, learners)
    assert len(coordinator.reset.call_args.args[0]) == 2
    assert coordinator.collect_window.call_count == 4
    assert coordinator.update_learners.call_count == 4
    assert checkpoint_writer.call_args_list == [
        call(
            path=config.run_dir / f"checkpoint_{window}.pt",
            learners=learners,
            learner_config=config.learner_config,
            rollout_config=config.rollout_config,
            total_environment_transitions=window * 6,
            completed_updates=window,
        )
        for window in range(1, 5)
    ]
    assert result == runner.TrainingResult(
        run_dir=config.run_dir,
        checkpoint_path=config.run_dir / "checkpoint_4.pt",
        completed_update_cycles=4,
        total_environment_transitions=24,
    )
    output = capsys.readouterr().out
    assert output.count("[####################] 6/6 transitions") == 4
    assert output.count("Update ") == 4
    assert "\rWindow 1/4 [--------------------] 0/6 transitions" in output
    assert "\rWindow 1/4 [##########----------] 3/6 transitions" in output
    assert output.index("3/6 transitions") < output.index("6/6 transitions")
    assert output.index("6/6 transitions") < output.index("Update 1/4 complete")
    pool.close.assert_called_once_with()
    for environment in environments:
        environment.close.assert_not_called()


def test_runner_closes_constructed_replicas_after_construction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later failed constructor must not leak an earlier MuJoCo instance."""
    reference = Mock(possible_agents=[0])
    first_environment = Mock()
    monkeypatch.setattr(
        runner,
        "CentipedeParallelEnv",
        Mock(
            side_effect=[
                reference,
                first_environment,
                RuntimeError("model load failed"),
            ]
        ),
    )
    monkeypatch.setattr(runner, "build_learners", Mock(return_value={0: object()}))

    with pytest.raises(RuntimeError, match="model load failed"):
        runner.run_training(_config(tmp_path))

    reference.close.assert_called_once_with()
    first_environment.close.assert_called_once_with()


def test_runner_uses_process_pool_after_closing_reference_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass only physics configuration to workers, never the reference object."""
    config = replace(
        _config(tmp_path),
        environment_count=4,
        worker_count=2,
        update_cycles=1,
    )
    reference = Mock(possible_agents=[0])
    learners = {0: object()}
    pool = Mock()
    process_pool_factory = Mock(return_value=pool)
    coordinator = Mock()
    coordinator.collect_window.return_value = {0: object()}
    coordinator.update_learners.return_value = {
        0: (PPOUpdateResult(actor_loss=0.1, critic_loss=0.2, entropy=0.3),)
    }

    monkeypatch.setattr(runner, "CentipedeParallelEnv", Mock(return_value=reference))
    monkeypatch.setattr(runner, "build_learners", Mock(return_value=learners))
    monkeypatch.setattr(runner, "SerialEnvironmentPool", Mock())
    monkeypatch.setattr(runner, "ProcessEnvironmentPool", process_pool_factory)
    monkeypatch.setattr(runner, "RolloutCoordinator", Mock(return_value=coordinator))
    monkeypatch.setattr(runner, "save_checkpoint", Mock())

    runner.run_training(config)

    reference.close.assert_called_once_with()
    process_pool_factory.assert_called_once_with(
        model_path=config.model_path,
        max_episode_steps=config.max_episode_steps,
        environment_count=4,
        worker_count=2,
    )
    runner.SerialEnvironmentPool.assert_not_called()
    pool.close.assert_called_once_with()


def test_runner_stops_before_checkpoint_on_nonfinite_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not save a learner bundle after a numerically invalid PPO update."""
    config = replace(_config(tmp_path), environment_count=1, update_cycles=1)
    reference = Mock(possible_agents=[0])
    environment = Mock()
    pool = Mock()
    coordinator = Mock()
    coordinator.collect_window.return_value = {0: object()}
    coordinator.update_learners.return_value = {
        0: (PPOUpdateResult(actor_loss=float("nan"), critic_loss=0.2, entropy=0.3),)
    }
    checkpoint_writer = Mock()

    monkeypatch.setattr(
        runner,
        "CentipedeParallelEnv",
        Mock(side_effect=[reference, environment]),
    )
    monkeypatch.setattr(runner, "build_learners", Mock(return_value={0: object()}))
    monkeypatch.setattr(runner, "SerialEnvironmentPool", Mock(return_value=pool))
    monkeypatch.setattr(runner, "RolloutCoordinator", Mock(return_value=coordinator))
    monkeypatch.setattr(runner, "save_checkpoint", checkpoint_writer)

    with pytest.raises(FloatingPointError, match="agent 0 in window 1"):
        runner.run_training(config)

    checkpoint_writer.assert_not_called()
    pool.close.assert_called_once_with()
    reference.close.assert_called_once_with()
    environment.close.assert_not_called()


def test_runner_completes_one_real_spawned_window(tmp_path: Path) -> None:
    """Connect the real runner, process pool, PPO update, and checkpoint output."""
    config = ExperimentConfig(
        model_path=CPU_ROOT.parent / "models" / "assembly.xml",
        run_dir=tmp_path / "spawned-run",
        training_seed=2026,
        environment_count=2,
        worker_count=2,
        update_cycles=1,
        max_episode_steps=2,
        evaluation_seeds=(),
        learner_config=LearnerConfig(hidden_sizes=(8, 8)),
        rollout_config=RolloutConfig(
            rollout_window_steps=1,
            update_epochs=1,
            minibatch_size=2,
        ),
    )

    result = runner.run_training(config)

    assert result.completed_update_cycles == 1
    assert result.total_environment_transitions == 2
    assert result.checkpoint_path == config.run_dir / "checkpoint_1.pt"
    assert result.checkpoint_path.is_file()
