"""Render frozen-policy evaluations as a readable static HTML file."""

from html import escape
from pathlib import Path

from centipede.environment import AgentID
from centipede.experiment.config import ExperimentConfig
from centipede.experiment.evaluation import (
    AgentDiagnostics,
    EpisodeDiagnostics,
    PolicyEvaluation,
    RunEvaluation,
)

AGENT_COLUMNS = (
    ("Return", "return_total"),
    ("Arrival", "reward_arrival"),
    ("Efficiency", "reward_efficiency"),
    ("Body reward", "reward_body_contact"),
    ("Leg reward", "reward_leg_contact"),
    ("Body contact steps", "body_contact_steps"),
    ("Leg contact steps", "leg_contact_steps"),
    ("Left foot steps", "left_foot_ground_steps"),
    ("Right foot steps", "right_foot_ground_steps"),
    ("Mean action L2", "mean_action_l2"),
)


def write_html_report(
    path: str | Path,
    config: ExperimentConfig,
    evaluation: RunEvaluation,
) -> Path:
    """Write a standalone report at the chosen result path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html_report(config, evaluation), encoding="utf-8")
    return output_path


def render_html_report(config: ExperimentConfig, evaluation: RunEvaluation) -> str:
    """Format supplied diagnostics only; do not run or recalculate evaluation."""
    policies = (*evaluation.checkpoints, *evaluation.baselines)
    comparison_rows = "\n".join(_comparison_row(policy) for policy in policies)
    policy_sections = "\n".join(_policy_section(policy) for policy in policies)
    seed_list = ", ".join(str(seed) for seed in evaluation.seeds)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Centipede evaluation</title>
<style>
  :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
  body {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #17212b; }}
  h1, h2 {{ line-height: 1.2; }}
  .note {{ padding: .8rem 1rem; background: #eef4fa; border-left: 4px solid #3478b2; }}
  .table-wrap {{ overflow-x: auto; margin: 1rem 0 2rem; }}
  table {{ border-collapse: collapse; min-width: 100%;
           font-variant-numeric: tabular-nums; }}
  th, td {{ padding: .45rem .65rem; text-align: right;
           border-bottom: 1px solid #d9e1e7; white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead {{ background: #edf2f6; }}
  details {{ margin: .8rem 0; padding: .4rem .7rem;
             border: 1px solid #d9e1e7; border-radius: .3rem; }}
  summary {{ cursor: pointer; font-weight: 600; }}
  .metadata {{ display: grid; grid-template-columns: max-content 1fr;
               gap: .3rem 1rem; }}
  .metadata dt {{ font-weight: 600; }}
  .metadata dd {{ margin: 0; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>Centipede evaluation</h1>
<p class="note">Policies were evaluated without training updates. Compare the
same episode seeds across checkpoints; a reward difference alone does not prove
walking or cooperative movement.</p>
<dl class="metadata">
  <dt>Model</dt><dd>{escape(str(config.model_path))}</dd>
  <dt>Run directory</dt><dd>{escape(str(config.run_dir))}</dd>
  <dt>Training seed</dt><dd>{config.training_seed}</dd>
  <dt>Episode limit</dt><dd>{config.max_episode_steps} steps</dd>
  <dt>Evaluation seeds</dt><dd>{escape(seed_list)}</dd>
  <dt>Policy action rule</dt>
  <dd>Deterministic for checkpoints; seeded random baseline</dd>
</dl>
<h2>Comparison</h2>
<div class="table-wrap"><table>
<thead><tr><th>Policy</th><th>Updates</th><th>Transitions</th><th>Episodes</th>
<th>Head return</th><th>Mean follower return</th><th>Success</th>
<th>Initial target distance</th><th>Final target distance</th>
<th>Head path length</th></tr></thead>
<tbody>{comparison_rows}</tbody>
</table></div>
{policy_sections}
</body>
</html>
"""


def _comparison_row(policy: PolicyEvaluation) -> str:
    """Show the primary comparison without hiding per-agent detail below."""
    agent_diagnostics = policy.summary.agent_diagnostics
    head_return = agent_diagnostics[0].return_total if 0 in agent_diagnostics else None
    follower_returns = [
        metrics.return_total
        for agent, metrics in agent_diagnostics.items()
        if agent != 0
    ]
    mean_follower_return = (
        sum(follower_returns) / len(follower_returns) if follower_returns else None
    )
    return (
        f"<tr><th>{escape(policy.label)}</th>"
        f"<td>{_optional_integer(policy.completed_updates)}</td>"
        f"<td>{_optional_integer(policy.total_environment_transitions)}</td>"
        f"<td>{len(policy.episodes)}</td>"
        f"<td>{_number(head_return)}</td>"
        f"<td>{_number(mean_follower_return)}</td>"
        f"<td>{policy.summary.success_rate * 100:.1f}%</td>"
        f"<td>{policy.summary.mean_initial_target_distance_m * 1000:.3g} mm</td>"
        f"<td>{policy.summary.mean_final_target_distance_m * 1000:.3g} mm</td>"
        f"<td>{policy.summary.mean_head_distance_traveled_m * 1000:.3g} mm</td></tr>"
    )


def _policy_section(policy: PolicyEvaluation) -> str:
    """Show the checkpoint summary and each seed's complete diagnostics."""
    summary = policy.summary
    episodes = "\n".join(_episode_section(episode) for episode in policy.episodes)
    checkpoint = (
        escape(str(policy.checkpoint_path))
        if policy.checkpoint_path is not None
        else "No checkpoint (baseline)"
    )
    return f"""<section>
<h2>{escape(policy.label)}</h2>
<p>Source: {checkpoint}. Mean episode length: {_number(summary.mean_episode_steps)}
steps ({_number(summary.mean_episode_time_s)} s). Mean initial/final target distance:
{summary.mean_initial_target_distance_m * 1000:.3g} /
{summary.mean_final_target_distance_m * 1000:.3g} mm.
Mean head path length: {summary.mean_head_distance_traveled_m * 1000:.3g} mm.</p>
<details open><summary>Per-agent averages</summary>
{_agent_table(summary.agent_diagnostics)}
</details>
<h3>Episodes</h3>
{episodes}
</section>"""


def _episode_section(episode: EpisodeDiagnostics) -> str:
    """Keep every episode's task outcome and eight agent records inspectable."""
    outcome = "reached" if episode.target_reached else "not reached"
    return f"""<details>
<summary>Seed {episode.seed}: {outcome}, {episode.episode_steps} steps</summary>
<p>End: {escape(episode.episode_end)}. Duration: {_number(episode.episode_time_s)} s.
Initial/final target distance: {episode.initial_target_distance_m * 1000:.3g} /
{episode.final_target_distance_m * 1000:.3g} mm.
Head path length: {episode.head_distance_traveled_m * 1000:.3g} mm.</p>
{_agent_table(episode.agent_diagnostics)}
</details>"""


def _agent_table(agent_diagnostics: dict[AgentID, AgentDiagnostics]) -> str:
    """Render every declared agent diagnostic in a stable column order."""
    headers = "".join(f"<th>{escape(label)}</th>" for label, _ in AGENT_COLUMNS)
    rows = []
    for agent, metrics in sorted(agent_diagnostics.items()):
        cells = "".join(
            f"<td>{_number(getattr(metrics, field))}</td>" for _, field in AGENT_COLUMNS
        )
        rows.append(f"<tr><th>Segment {agent}</th>{cells}</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr><th>Agent</th>'
        f"{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _number(value: float | None) -> str:
    """Keep small signed rewards visible while formatting absent values plainly."""
    return "—" if value is None else f"{value:.6g}"


def _optional_integer(value: int | None) -> str:
    """Show baseline metadata without pretending it has training counters."""
    return "—" if value is None else str(value)
