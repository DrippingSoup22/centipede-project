"""Focused checks for the human-readable report and its command-line entry."""

from pathlib import Path

from centipede.experiment.config import ExperimentConfig
from centipede.experiment.evaluation import (
    AgentDiagnostics,
    EpisodeDiagnostics,
    PolicyEvaluation,
    PolicySummary,
    RunEvaluation,
)
from centipede.experiment.report import write_html_report
from centipede.training.learners import LearnerConfig
from centipede.training.rollout import RolloutConfig


def _agent_metrics(total_return: float) -> AgentDiagnostics:
    """Provide one complete, readable agent record for report fixtures."""
    return AgentDiagnostics(
        return_total=total_return,
        reward_arrival=1.0,
        reward_efficiency=-0.1,
        reward_body_contact=-0.02,
        reward_leg_contact=-0.03,
        body_contact_steps=2,
        leg_contact_steps=3,
        left_foot_ground_steps=4,
        right_foot_ground_steps=5,
        mean_action_l2=0.25,
    )


def test_html_report_shows_checkpoint_baseline_and_all_agent_diagnostics(
    tmp_path: Path,
) -> None:
    """Keep the summary readable while retaining each episode's full details."""
    config = ExperimentConfig(
        model_path=tmp_path / "model.xml",
        run_dir=tmp_path / "run",
        training_seed=2026,
        environment_count=1,
        worker_count=1,
        update_cycles=4,
        max_episode_steps=2_048,
        evaluation_seeds=(30001,),
        learner_config=LearnerConfig(),
        rollout_config=RolloutConfig(),
    )
    episode = EpisodeDiagnostics(
        seed=30001,
        agent_diagnostics={0: _agent_metrics(0.8), 1: _agent_metrics(-0.3)},
        target_reached=True,
        episode_end="arrival",
        episode_steps=40,
        episode_time_s=0.8,
        initial_target_distance_m=0.015,
        final_target_distance_m=0.001,
        head_distance_traveled_m=0.023,
    )
    summary = PolicySummary(
        agent_diagnostics=episode.agent_diagnostics,
        success_rate=1.0,
        mean_episode_steps=40.0,
        mean_episode_time_s=0.8,
        mean_initial_target_distance_m=0.015,
        mean_final_target_distance_m=0.001,
        mean_head_distance_traveled_m=0.023,
    )
    checkpoint = PolicyEvaluation(
        label="checkpoint_1",
        checkpoint_path=config.run_dir / "checkpoint_1.pt",
        completed_updates=1,
        total_environment_transitions=256,
        episodes=(episode,),
        summary=summary,
    )
    baseline = PolicyEvaluation(
        label="random <trial>",
        checkpoint_path=None,
        completed_updates=None,
        total_environment_transitions=None,
        episodes=(episode,),
        summary=summary,
    )
    evaluation = RunEvaluation(
        seeds=(30001,), checkpoints=(checkpoint,), baselines=(baseline,)
    )

    output = tmp_path / "report" / "evaluation.html"
    assert write_html_report(output, config, evaluation) == output
    html = output.read_text(encoding="utf-8")

    assert "checkpoint_1" in html
    assert "random &lt;trial&gt;" in html
    assert "random <trial>" not in html
    assert "Segment 0" in html and "Segment 1" in html
    assert "Body contact steps" in html and "Mean action L2" in html
    assert "Seed 30001: reached, 40 steps" in html
    assert "100.0%" in html
    assert "256" in html
    assert "Initial target distance" in html
    assert "Head path length" in html
    assert "Episode limit</dt><dd>2048 steps" in html
    assert "15 mm" in html and "23 mm" in html
