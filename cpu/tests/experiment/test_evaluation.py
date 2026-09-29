"""Focused checks for evaluation episodes, summaries, and baselines."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

from centipede.environment import CentipedeParallelEnv
from centipede.experiment import evaluation as evaluation_module
from centipede.experiment.config import ExperimentConfig
from centipede.experiment.evaluation import (
    AgentDiagnostics,
    EpisodeDiagnostics,
    _evaluate_episode,
    _summarize_episodes,
    evaluate_baseline,
    evaluate_checkpoint,
    evaluate_run,
)
from centipede.experiment.presets import TrainingPlan, load_plan
from centipede.training.checkpoints import save_checkpoint
from centipede.training.learners import LearnerConfig, build_learners

CPU_ROOT = Path(__file__).resolve().parents[2]


def _config(run_dir: Path | None = None) -> ExperimentConfig:
    """Build a concrete evaluation config from the supported smoke preset."""
    plan = load_plan(CPU_ROOT / "configs" / "smoke.toml")
    assert isinstance(plan, TrainingPlan)
    preset = plan.preset
    return preset.for_run(
        run_dir or CPU_ROOT.parent / "runs" / "smoke",
        evaluation_seeds=(30001, 30002),
    )


def _step_info(
    *,
    arrival: float = 0.0,
    efficiency: float = 0.0,
    body_reward: float = 0.0,
    leg_reward: float = 0.0,
    body_contact: bool = False,
    leg_contact: bool = False,
    left_foot: bool = False,
    right_foot: bool = False,
) -> dict[str, Any]:
    """Provide only the diagnostic fields published after a real task step."""
    return {
        "reward_arrival": arrival,
        "reward_efficiency": efficiency,
        "reward_body_contact": body_reward,
        "reward_leg_contact": leg_reward,
        "body_ground_contact": body_contact,
        "leg_leg_contact": leg_contact,
        "left_foot_ground_contact": left_foot,
        "right_foot_ground_contact": right_foot,
    }


def test_episode_keeps_each_agents_rewards_contacts_and_actions_separate() -> None:
    """Sum two transitions, including a shared arrival, under one episode seed."""
    environment = Mock(possible_agents=[0, 1])
    first_observations = {
        0: np.array([0], dtype=np.float32),
        1: np.array([1], dtype=np.float32),
    }
    second_observations = {
        0: np.array([2], dtype=np.float32),
        1: np.array([3], dtype=np.float32),
    }
    final_observations = {
        0: np.array([4], dtype=np.float32),
        1: np.array([5], dtype=np.float32),
    }
    environment.reset.return_value = (
        first_observations,
        {0: {"target_distance_m": 0.012}, 1: {}},
    )
    first_infos = {
        0: _step_info(
            efficiency=-0.08, body_reward=-0.02, body_contact=True, left_foot=True
        ),
        1: _step_info(
            efficiency=-0.15, leg_reward=-0.05, leg_contact=True, right_foot=True
        ),
    }
    second_infos = {
        0: _step_info(arrival=1.0, efficiency=-0.10, left_foot=True, right_foot=True),
        1: _step_info(arrival=1.0, efficiency=-0.20, left_foot=True),
    }
    second_infos[0].update(
        head_step_distance_m=0.003,
        target_distance_m=0.0005,
        target_reached=True,
        episode_end="arrival",
        episode_steps=2,
        episode_time_s=0.04,
    )
    first_infos[0]["head_step_distance_m"] = 0.004
    environment.step.side_effect = [
        (
            second_observations,
            {0: -0.1, 1: -0.2},
            {0: False, 1: False},
            {0: False, 1: False},
            first_infos,
        ),
        (
            final_observations,
            {0: 0.9, 1: 0.8},
            {0: True, 1: True},
            {0: False, 1: False},
            second_infos,
        ),
    ]
    first_actions = {
        0: np.array([0.6, 0.8, 0, 0, 0, 0], dtype=np.float32),
        1: np.array([0.3, 0.4, 0, 0, 0, 0], dtype=np.float32),
    }
    second_actions = {
        0: np.array([0.3, 0.4, 0, 0, 0, 0], dtype=np.float32),
        1: np.zeros(6, dtype=np.float32),
    }
    select_actions = Mock(side_effect=[first_actions, second_actions])
    on_step = Mock()

    episode = _evaluate_episode(environment, 30001, select_actions, on_step=on_step)

    environment.reset.assert_called_once_with(seed=30001)
    assert environment.step.call_count == 2
    assert on_step.call_args_list == [call(0), call(1), call(2)]
    assert select_actions.call_args_list[0].args[0] is first_observations
    assert select_actions.call_args_list[1].args[0] is second_observations
    assert episode.seed == 30001
    assert episode.target_reached is True
    assert episode.episode_end == "arrival"
    assert episode.episode_steps == 2
    assert episode.episode_time_s == pytest.approx(0.04)
    assert episode.initial_target_distance_m == pytest.approx(0.012)
    assert episode.final_target_distance_m == pytest.approx(0.0005)
    assert episode.head_distance_traveled_m == pytest.approx(0.007)
    assert episode.agent_diagnostics[0].return_total == pytest.approx(0.8)
    assert episode.agent_diagnostics[1].return_total == pytest.approx(0.6)
    assert episode.agent_diagnostics[0].reward_efficiency == pytest.approx(-0.18)
    assert episode.agent_diagnostics[1].reward_leg_contact == pytest.approx(-0.05)
    assert episode.agent_diagnostics[0].body_contact_steps == 1
    assert episode.agent_diagnostics[1].leg_contact_steps == 1
    assert episode.agent_diagnostics[0].left_foot_ground_steps == 2
    assert episode.agent_diagnostics[0].right_foot_ground_steps == 1
    assert episode.agent_diagnostics[0].mean_action_l2 == pytest.approx(0.75)
    assert episode.agent_diagnostics[1].mean_action_l2 == pytest.approx(0.25)


def test_episode_stops_on_time_limit_without_another_reset() -> None:
    """Truncation uses the environment's final info and ends collection once."""
    environment = Mock(possible_agents=[0])
    environment.reset.return_value = (
        {0: np.zeros(1, dtype=np.float32)},
        {0: {"target_distance_m": 0.019}},
    )
    final_info = _step_info()
    final_info.update(
        head_step_distance_m=0.002,
        target_distance_m=0.018,
        target_reached=False,
        episode_end="time_limit",
        episode_steps=1,
        episode_time_s=0.02,
    )
    environment.step.return_value = (
        {0: np.zeros(1, dtype=np.float32)},
        {0: 0.0},
        {0: False},
        {0: True},
        {0: final_info},
    )

    episode = _evaluate_episode(
        environment,
        30002,
        Mock(return_value={0: np.zeros(6, dtype=np.float32)}),
    )

    assert episode.target_reached is False
    assert episode.episode_end == "time_limit"
    assert episode.episode_steps == 1
    assert episode.initial_target_distance_m == pytest.approx(0.019)
    assert episode.head_distance_traveled_m == pytest.approx(0.002)
    assert episode.agent_diagnostics[0].mean_action_l2 == 0.0
    environment.reset.assert_called_once_with(seed=30002)
    environment.step.assert_called_once()


def test_episode_reads_the_real_environment_diagnostic_contract() -> None:
    """The accumulator must consume actual PettingZoo infos without extra fields."""
    environment = CentipedeParallelEnv(
        model_path=CPU_ROOT.parent / "models" / "assembly.xml",
        max_episode_steps=2,
    )
    try:
        episode = _evaluate_episode(
            environment,
            30001,
            lambda observations: {
                agent: np.zeros(6, dtype=np.float32) for agent in observations
            },
        )
        assert set(episode.agent_diagnostics) == set(environment.possible_agents)
        assert 1 <= episode.episode_steps <= 2
        assert episode.episode_end in {"arrival", "time_limit"}
        assert 0.010 <= episode.initial_target_distance_m <= 0.020
        assert np.isfinite(episode.head_distance_traveled_m)
        assert all(
            np.isfinite(metrics.return_total)
            for metrics in episode.agent_diagnostics.values()
        )
    finally:
        environment.close()


def test_summary_averages_each_agent_across_episodes() -> None:
    """Keep segment identities separate while averaging complete episodes."""
    agent = AgentDiagnostics(
        return_total=1.0,
        reward_arrival=0.0,
        reward_efficiency=-0.1,
        reward_body_contact=0.0,
        reward_leg_contact=0.0,
        body_contact_steps=0.0,
        leg_contact_steps=0.0,
        left_foot_ground_steps=1.0,
        right_foot_ground_steps=1.0,
        mean_action_l2=0.2,
    )
    first = EpisodeDiagnostics(
        seed=1,
        agent_diagnostics={0: agent, 1: replace(agent, return_total=-2.0)},
        target_reached=False,
        episode_end="time_limit",
        episode_steps=40,
        episode_time_s=0.8,
        initial_target_distance_m=0.015,
        final_target_distance_m=0.010,
        head_distance_traveled_m=0.020,
    )
    second = replace(
        first,
        seed=2,
        agent_diagnostics={
            0: replace(agent, return_total=3.0, mean_action_l2=0.6),
            1: replace(agent, return_total=-4.0),
        },
        target_reached=True,
        episode_end="arrival",
        episode_steps=20,
        episode_time_s=0.4,
        final_target_distance_m=0.001,
        head_distance_traveled_m=0.030,
    )

    summary = _summarize_episodes((first, second))

    assert summary.success_rate == pytest.approx(0.5)
    assert summary.mean_episode_steps == pytest.approx(30.0)
    assert summary.mean_head_distance_traveled_m == pytest.approx(0.025)
    assert summary.agent_diagnostics[0].return_total == pytest.approx(2.0)
    assert summary.agent_diagnostics[0].mean_action_l2 == pytest.approx(0.4)
    assert summary.agent_diagnostics[1].return_total == pytest.approx(-3.0)


def test_baselines_use_action_spaces_and_repeat_seeded_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare zero and random actions on the same short seeded episodes."""
    config = replace(_config(), max_episode_steps=2_048)
    episode_limits: list[int] = []

    def short_environment(
        model_path: Path, max_episode_steps: int
    ) -> CentipedeParallelEnv:
        episode_limits.append(max_episode_steps)
        return CentipedeParallelEnv(model_path=model_path, max_episode_steps=2)

    monkeypatch.setattr(
        evaluation_module,
        "CentipedeParallelEnv",
        short_environment,
    )
    seeds = (30001, 30002)

    zero = evaluate_baseline(config, "zero", seeds)
    random = evaluate_baseline(config, "random", seeds)
    random_again = evaluate_baseline(config, "random", seeds)

    assert zero.checkpoint_path is None
    assert zero.completed_updates is None
    assert zero.total_environment_transitions is None
    assert tuple(episode.seed for episode in zero.episodes) == seeds
    assert [episode.initial_target_distance_m for episode in zero.episodes] == [
        episode.initial_target_distance_m for episode in random.episodes
    ]
    assert all(
        metrics.mean_action_l2 == 0.0
        for episode in zero.episodes
        for metrics in episode.agent_diagnostics.values()
    )
    assert all(
        metrics.mean_action_l2 > 0.0
        for episode in random.episodes
        for metrics in episode.agent_diagnostics.values()
    )
    assert random.episodes == random_again.episodes
    assert episode_limits == [2_048] * 3


def test_checkpoint_evaluation_restores_all_agents_and_keeps_actions_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluate a saved learner bundle on short episodes without training."""
    config = replace(
        _config(),
        learner_config=LearnerConfig(hidden_sizes=(8,)),
        max_episode_steps=2_048,
    )
    checkpoint_path = tmp_path / "checkpoint_2.pt"
    source_environment = CentipedeParallelEnv(model_path=config.model_path)
    try:
        learners = build_learners(
            source_environment,
            config.learner_config,
            {agent: agent for agent in source_environment.possible_agents},
        )
        save_checkpoint(
            checkpoint_path,
            learners,
            config.learner_config,
            config.rollout_config,
            total_environment_transitions=512,
            completed_updates=2,
        )
    finally:
        source_environment.close()

    episode_limits: list[int] = []

    def short_environment(
        model_path: Path, max_episode_steps: int
    ) -> CentipedeParallelEnv:
        episode_limits.append(max_episode_steps)
        return CentipedeParallelEnv(model_path=model_path, max_episode_steps=2)

    monkeypatch.setattr(evaluation_module, "CentipedeParallelEnv", short_environment)
    with torch.random.fork_rng(devices=[]):
        result = evaluate_checkpoint(config, checkpoint_path, (30001, 30002))
        repeated = evaluate_checkpoint(config, checkpoint_path, (30001, 30002))

    assert result.label == "checkpoint_2"
    assert result.checkpoint_path == checkpoint_path
    assert result.completed_updates == 2
    assert result.total_environment_transitions == 512
    assert tuple(episode.seed for episode in result.episodes) == (30001, 30002)
    assert result.episodes == repeated.episodes
    assert episode_limits == [2_048, 2_048]
    assert all(
        set(episode.agent_diagnostics) == set(learners) for episode in result.episodes
    )


def test_run_evaluation_orders_checkpoints_and_reuses_the_same_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Orchestrate saved policies and both baselines without running them here."""
    config = _config(tmp_path)
    paths = [tmp_path / f"checkpoint_{number}.pt" for number in (10, 2, 1)]
    for path in paths:
        path.touch()

    checkpoint_results = {path: Mock() for path in paths}
    baseline_results = {mode: Mock() for mode in ("zero", "random")}
    checkpoint_evaluator = Mock(
        side_effect=lambda _config, path, _seeds: checkpoint_results[path]
    )
    baseline_evaluator = Mock(
        side_effect=lambda _config, mode, _seeds: baseline_results[mode]
    )
    monkeypatch.setattr(evaluation_module, "evaluate_checkpoint", checkpoint_evaluator)
    monkeypatch.setattr(evaluation_module, "evaluate_baseline", baseline_evaluator)

    result = evaluate_run(config)

    ordered_paths = (paths[2], paths[1], paths[0])
    assert result.seeds == config.evaluation_seeds
    assert result.checkpoints == tuple(
        checkpoint_results[path] for path in ordered_paths
    )
    assert result.baselines == (
        baseline_results["zero"],
        baseline_results["random"],
    )
    assert checkpoint_evaluator.call_args_list == [
        call(config, path, config.evaluation_seeds) for path in ordered_paths
    ]
    assert baseline_evaluator.call_args_list == [
        call(config, mode, config.evaluation_seeds) for mode in ("zero", "random")
    ]
    assert capsys.readouterr().out == ""


def test_episode_progress_reports_simulated_steps_and_early_arrival(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bar moves inside the episode and never fakes completion on arrival."""
    environment = Mock(max_episode_steps=10)

    def short_episode(_environment: Mock, _seed: int, _actions: Mock, *, on_step):
        for steps in (0, 1, 3):
            on_step(steps)
        return Mock(episode_end="arrival")

    monkeypatch.setattr(evaluation_module, "_evaluate_episode", short_episode)
    evaluation_module._evaluate_with_progress(
        environment, 30001, Mock(), "checkpoint_2"
    )

    output = capsys.readouterr().out
    assert "checkpoint_2 | seed 30001 [--------------------] 0/10 steps" in output
    assert "checkpoint_2 | seed 30001 [######--------------] 3/10 steps" in output
    assert "3/10 steps | arrival" in output
    assert "[####################]" not in output


def test_run_evaluation_requires_at_least_one_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not present baseline-only output as an evaluated training run."""
    config = _config(tmp_path)
    baseline_evaluator = Mock()
    monkeypatch.setattr(evaluation_module, "evaluate_baseline", baseline_evaluator)

    with pytest.raises(FileNotFoundError, match="No checkpoints found"):
        evaluate_run(config)

    baseline_evaluator.assert_not_called()
