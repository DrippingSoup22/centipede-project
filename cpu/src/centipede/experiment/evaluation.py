"""Evaluate frozen checkpoints and baselines for one Centipede run.

``evaluate_run`` coordinates policies; each policy supplies actions to
``_evaluate_episode``, then ``_summarize_episodes`` averages its episodes.
The result types do not depend on the future number of environment replicas.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from gymnasium import spaces

from centipede.environment import Action, AgentID, CentipedeParallelEnv, Observation
from centipede.experiment.config import ExperimentConfig
from centipede.training.checkpoints import load_checkpoint
from centipede.training.learners import build_learners


@dataclass(frozen=True)
class AgentDiagnostics:
    """One agent's episode totals or across-episode means.

    Contact fields count flagged control steps in an episode and are averages
    of those counts in a policy summary. Action magnitude is the mean L2 norm
    of the six-component action over an episode.
    """

    return_total: float
    reward_arrival: float
    reward_efficiency: float
    reward_body_contact: float
    reward_leg_contact: float
    body_contact_steps: float
    leg_contact_steps: float
    left_foot_ground_steps: float
    right_foot_ground_steps: float
    mean_action_l2: float


@dataclass(frozen=True)
class EpisodeDiagnostics:
    """Complete task and per-agent diagnostics for one seeded episode."""

    seed: int
    agent_diagnostics: dict[AgentID, AgentDiagnostics]
    target_reached: bool
    episode_end: str
    episode_steps: int
    episode_time_s: float
    initial_target_distance_m: float
    final_target_distance_m: float
    head_distance_traveled_m: float


@dataclass(frozen=True)
class PolicySummary:
    """Means across the same held-out episode seeds for one frozen policy."""

    agent_diagnostics: dict[AgentID, AgentDiagnostics]
    success_rate: float
    mean_episode_steps: float
    mean_episode_time_s: float
    mean_initial_target_distance_m: float
    mean_final_target_distance_m: float
    mean_head_distance_traveled_m: float


@dataclass(frozen=True)
class PolicyEvaluation:
    """One checkpoint or baseline, with both episode records and a summary."""

    label: str
    checkpoint_path: Path | None
    completed_updates: int | None
    total_environment_transitions: int | None
    episodes: tuple[EpisodeDiagnostics, ...]
    summary: PolicySummary


@dataclass(frozen=True)
class RunEvaluation:
    """Comparable results for every selected checkpoint and both baselines."""

    seeds: tuple[int, ...]
    checkpoints: tuple[PolicyEvaluation, ...]
    baselines: tuple[PolicyEvaluation, ...]


def evaluate_run(config: ExperimentConfig) -> RunEvaluation:
    """Evaluate every run checkpoint and the zero/random baselines.

    Discover the synchronized checkpoints, keep their training order, and give
    each policy the same held-out seeds. This orchestration may later distribute
    seeds over several environments without changing the returned data types.
    """
    seeds = config.evaluation_seeds

    checkpoint_paths = sorted(
        config.run_dir.glob("checkpoint_*.pt"),
        key=lambda path: int(path.stem.removeprefix("checkpoint_")),
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {config.run_dir}")

    checkpoint_results: list[PolicyEvaluation] = []
    for path in checkpoint_paths:
        checkpoint_results.append(evaluate_checkpoint(config, path, seeds))

    baseline_results: list[PolicyEvaluation] = []
    for mode in ("zero", "random"):
        baseline_results.append(evaluate_baseline(config, mode, seeds))

    return RunEvaluation(
        seeds=seeds,
        checkpoints=tuple(checkpoint_results),
        baselines=tuple(baseline_results),
    )


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Path,
    seeds: Sequence[int],
) -> PolicyEvaluation:
    """Restore all eight learners and evaluate them without state updates.

    Use the existing checkpoint loader and each learner's saved normalizer with
    ``update=False``. Select deterministic bounded actions from RL_lib and close
    every environment after its assigned episodes.
    """

    environment = CentipedeParallelEnv(
        model_path=config.model_path,
        max_episode_steps=config.max_episode_steps,
    )

    try:
        learner_seeds = {agent: agent for agent in environment.possible_agents}
        learners = build_learners(
            environment,
            config.learner_config,
            learner_seeds,
        )
        metadata = load_checkpoint(checkpoint_path, learners)

        def select_actions(
            observations: Mapping[AgentID, Observation],
        ) -> dict[AgentID, Action]:
            actions: dict[AgentID, Action] = {}

            for agent, observation in observations.items():
                learner = learners[agent]
                normalized = learner.normalizer.normalize(observation, update=False)
                actions[agent] = cast(
                    Action,
                    learner.ppo.select_action(normalized, deterministic=True),
                )
            return actions

        episodes = tuple(
            _evaluate_with_progress(
                environment, seed, select_actions, checkpoint_path.stem
            )
            for seed in seeds
        )

    finally:
        environment.close()

    return PolicyEvaluation(
        label=checkpoint_path.stem,
        checkpoint_path=checkpoint_path,
        completed_updates=metadata.completed_updates,
        total_environment_transitions=metadata.total_environment_transitions,
        episodes=episodes,
        summary=_summarize_episodes(episodes),
    )


def evaluate_baseline(
    config: ExperimentConfig,
    mode: Literal["zero", "random"],
    seeds: Sequence[int],
) -> PolicyEvaluation:
    """Run the same episodes with zero or independently seeded random actions."""
    environment = CentipedeParallelEnv(
        model_path=config.model_path,
        max_episode_steps=config.max_episode_steps,
    )
    episodes: list[EpisodeDiagnostics] = []

    try:
        # The public action spaces determine each segment's action shape and bounds.
        action_spaces = {
            agent: cast(spaces.Box, environment.action_space(agent))
            for agent in environment.possible_agents
        }

        rng: np.random.Generator

        def select_actions(
            observations: Mapping[AgentID, Observation],
        ) -> dict[AgentID, Action]:
            """Choose one new action for each active segment at this step."""
            actions: dict[AgentID, Action] = {}
            for agent in observations:
                space = action_spaces[agent]
                actions[agent] = (
                    np.zeros_like(space.low, dtype=np.float32)
                    if mode == "zero"
                    else rng.uniform(space.low, space.high).astype(np.float32)
                )
            return actions

        for seed in seeds:
            # Reset only the random stream; the episode loop calls this same rule
            # after each transition to obtain the next action.
            rng = np.random.default_rng(seed)
            episodes.append(
                _evaluate_with_progress(
                    environment, seed, select_actions, f"{mode} baseline"
                )
            )
    finally:
        environment.close()

    episode_records = tuple(episodes)
    return PolicyEvaluation(
        label=f"{mode} baseline",
        checkpoint_path=None,
        completed_updates=None,
        total_environment_transitions=None,
        episodes=episode_records,
        summary=_summarize_episodes(episode_records),
    )


def _evaluate_with_progress(
    environment: CentipedeParallelEnv,
    seed: int,
    select_actions: Callable[[Mapping[AgentID, Observation]], dict[AgentID, Action]],
    label: str,
) -> EpisodeDiagnostics:
    """Show actual simulated steps for one policy and held-out episode."""
    limit = environment.max_episode_steps
    last_percent = -1

    def show_step(steps: int) -> None:
        nonlocal last_percent
        percent = min(100, 100 * steps // limit)
        if percent == last_percent:
            return
        last_percent = percent
        filled = 20 * steps // limit
        bar = "#" * filled + "-" * (20 - filled)
        print(
            f"\r{label} | seed {seed} [{bar}] {steps}/{limit} steps",
            end="",
            flush=True,
        )

    try:
        episode = _evaluate_episode(
            environment, seed, select_actions, on_step=show_step
        )
    except BaseException:
        # Keep the terminal prompt on a new line even after Ctrl+C.
        print(flush=True)
        raise
    # Arrival can end an episode before the bar fills; show that result plainly.
    print(f" | {episode.episode_end}", flush=True)
    return episode


def _evaluate_episode(
    environment: CentipedeParallelEnv,
    seed: int,
    select_actions: Callable[[Mapping[AgentID, Observation]], dict[AgentID, Action]],
    *,
    on_step: Callable[[int], None] | None = None,
) -> EpisodeDiagnostics:
    """Accumulate task infos, undiscounted rewards, contacts, and action norms."""
    observations, reset_infos = environment.reset(seed=seed)
    head = environment.possible_agents[0]
    initial_target_distance = float(reset_infos[head]["target_distance_m"])
    head_distance_traveled = 0.0
    steps = 0
    if on_step is not None:
        on_step(0)

    # These totals belong to this episode only. Every key below is either a
    # reward/contact field published by environment.step() or an action measure.
    totals = {
        agent: {
            "return_total": 0.0,
            "reward_arrival": 0.0,
            "reward_efficiency": 0.0,
            "reward_body_contact": 0.0,
            "reward_leg_contact": 0.0,
            "body_contact_steps": 0.0,
            "leg_contact_steps": 0.0,
            "left_foot_ground_steps": 0.0,
            "right_foot_ground_steps": 0.0,
            "action_l2_sum": 0.0,
        }
        for agent in environment.possible_agents
    }

    while True:
        actions = select_actions(observations)
        observations, rewards, terminations, truncations, infos = environment.step(
            actions
        )
        steps += 1
        head_distance_traveled += float(infos[head]["head_step_distance_m"])

        # A contact flag contributes one for each completed control step where
        # it is true. No reset-state flag or episode-level duplicate is counted.
        for agent in environment.possible_agents:
            info = infos[agent]
            agent_totals = totals[agent]
            agent_totals["return_total"] += float(rewards[agent])
            agent_totals["reward_arrival"] += float(info["reward_arrival"])
            agent_totals["reward_efficiency"] += float(info["reward_efficiency"])
            agent_totals["reward_body_contact"] += float(info["reward_body_contact"])
            agent_totals["reward_leg_contact"] += float(info["reward_leg_contact"])
            agent_totals["body_contact_steps"] += float(info["body_ground_contact"])
            agent_totals["leg_contact_steps"] += float(info["leg_leg_contact"])
            agent_totals["left_foot_ground_steps"] += float(
                info["left_foot_ground_contact"]
            )
            agent_totals["right_foot_ground_steps"] += float(
                info["right_foot_ground_contact"]
            )
            agent_totals["action_l2_sum"] += float(np.linalg.norm(actions[agent]))

        if on_step is not None:
            on_step(steps)

        # The public environment ends all segment agents together. Its head info
        # supplies the final task outcome and elapsed episode time.
        if terminations[head] or truncations[head]:
            break

    head_info = infos[head]
    episode_steps = int(head_info["episode_steps"])
    agent_diagnostics = {
        agent: AgentDiagnostics(
            return_total=agent_totals["return_total"],
            reward_arrival=agent_totals["reward_arrival"],
            reward_efficiency=agent_totals["reward_efficiency"],
            reward_body_contact=agent_totals["reward_body_contact"],
            reward_leg_contact=agent_totals["reward_leg_contact"],
            body_contact_steps=agent_totals["body_contact_steps"],
            leg_contact_steps=agent_totals["leg_contact_steps"],
            left_foot_ground_steps=agent_totals["left_foot_ground_steps"],
            right_foot_ground_steps=agent_totals["right_foot_ground_steps"],
            mean_action_l2=agent_totals["action_l2_sum"] / episode_steps,
        )
        for agent, agent_totals in totals.items()
    }

    return EpisodeDiagnostics(
        seed=seed,
        agent_diagnostics=agent_diagnostics,
        target_reached=bool(head_info["target_reached"]),
        episode_end=str(head_info["episode_end"]),
        episode_steps=episode_steps,
        episode_time_s=float(head_info["episode_time_s"]),
        initial_target_distance_m=initial_target_distance,
        final_target_distance_m=float(head_info["target_distance_m"]),
        head_distance_traveled_m=head_distance_traveled,
    )


def _summarize_episodes(episodes: Sequence[EpisodeDiagnostics]) -> PolicySummary:
    """Average complete per-episode diagnostics without mixing agent identities."""

    agent_summaries: dict[AgentID, AgentDiagnostics] = {}

    for agent_id in episodes[0].agent_diagnostics:
        records = [episode.agent_diagnostics[agent_id] for episode in episodes]

        agent_summaries[agent_id] = AgentDiagnostics(
            return_total=float(np.mean([record.return_total for record in records])),
            reward_arrival=float(
                np.mean([record.reward_arrival for record in records])
            ),
            reward_efficiency=float(
                np.mean([record.reward_efficiency for record in records])
            ),
            reward_body_contact=float(
                np.mean([record.reward_body_contact for record in records])
            ),
            reward_leg_contact=float(
                np.mean([record.reward_leg_contact for record in records])
            ),
            body_contact_steps=float(
                np.mean([record.body_contact_steps for record in records])
            ),
            leg_contact_steps=float(
                np.mean([record.leg_contact_steps for record in records])
            ),
            left_foot_ground_steps=float(
                np.mean([record.left_foot_ground_steps for record in records])
            ),
            right_foot_ground_steps=float(
                np.mean([record.right_foot_ground_steps for record in records])
            ),
            mean_action_l2=float(
                np.mean([record.mean_action_l2 for record in records])
            ),
        )

    return PolicySummary(
        agent_diagnostics=agent_summaries,
        success_rate=np.mean([episode.target_reached for episode in episodes]),
        mean_episode_steps=np.mean([episode.episode_steps for episode in episodes]),
        mean_episode_time_s=np.mean([episode.episode_time_s for episode in episodes]),
        mean_initial_target_distance_m=np.mean(
            [episode.initial_target_distance_m for episode in episodes]
        ),
        mean_final_target_distance_m=np.mean(
            [episode.final_target_distance_m for episode in episodes]
        ),
        mean_head_distance_traveled_m=np.mean(
            [episode.head_distance_traveled_m for episode in episodes]
        ),
    )
