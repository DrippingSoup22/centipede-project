"""Public PettingZoo environment for the Centipede task.

The environment composes one internal MuJoCo simulation and owns all task-level
state: agents, spaces, target generation, episode timing, partial observations,
rewards, endings, and diagnostic information.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pettingzoo import ParallelEnv

from centipede.environment.observations import (
    SEGMENT_BLOCK_SIZE,
    build_observations,
)
from centipede.environment.rewards import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    RewardTerms,
    calculate_reward_terms,
)
from centipede.environment.simulation import CentipedeSimulation, PhysicalSnapshot

# Public PettingZoo types: integer segment IDs and float32 policy arrays.
AgentID = int
Observation = NDArray[np.float32]
Action = NDArray[np.float32]

# Targets begin close to the head and within a narrow forward-facing wedge.
TARGET_DISTANCE_RANGE_M = (0.010, 0.020)
TARGET_BEARING_LIMIT_RAD = np.deg2rad(15.0)

# The head reaches a point when it enters this planar tolerance radius.
ARRIVAL_RADIUS_M = 0.001

# At 50 control transitions per second, 1,000 steps give a 20-second episode.
DEFAULT_MAX_EPISODE_STEPS = 1_000


class CentipedeParallelEnv(ParallelEnv[AgentID, Observation, Action]):
    """Expose the shared centipede simulation as eight simultaneous agents.

    This is the only public task environment. It validates external actions once,
    advances the internal simulation once per parallel step, and returns separate
    PettingZoo dictionaries for every active segment.
    """

    # PettingZoo uses this metadata to identify the task and supported rendering.
    metadata = {
        "name": "centipede_v0",
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    # PettingZoo lifecycle and model-derived per-agent interfaces.
    simulation: CentipedeSimulation
    possible_agents: list[AgentID]
    agents: list[AgentID]
    observation_spaces: dict[AgentID, spaces.Space[Observation]]
    action_spaces: dict[AgentID, spaces.Space[Action]]
    # Task state established by reset and advanced by each public transition.
    target_position: NDArray[np.float64] | None
    previous_snapshot: PhysicalSnapshot | None
    episode_steps: int

    def __init__(
        self,
        model_path: str | Path,
        render_mode: str | None = None,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
    ) -> None:
        """Construct the simulation, stable agents, spaces, and task state.

        Model-dependent dimensions and segment IDs must be derived from the one
        XML loaded by ``CentipedeSimulation`` rather than hardcoded here.
        """

        if max_episode_steps <= 0:
            raise ValueError("Max episode steps must be > 0")

        self.simulation = CentipedeSimulation(
            model_path=model_path, render_mode=render_mode
        )

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.reward_config = reward_config

        # The loaded model is the single source of truth for segment membership.
        # Agents become active together only when reset starts an episode.
        self.possible_agents = list(self.simulation.segment_ids)
        self.agents = []

        # Cache stable spaces once; PettingZoo requires repeated space lookups for
        # the same agent to return the same object throughout the environment.
        action_size = self.simulation.leg_actuator_ids.shape[1]
        self.action_spaces = {
            agent: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_size,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.observation_spaces = {}
        last_agent = self.possible_agents[-1]
        for agent in self.possible_agents:
            # Radius one includes self and only the neighbors that actually exist.
            visible_blocks = 1
            if agent > 0:
                visible_blocks += 1
            if agent < last_agent:
                visible_blocks += 1

            observation_size = visible_blocks * SEGMENT_BLOCK_SIZE

            if agent == 0:
                # Only the head receives forward and lateral target displacement.
                observation_size += 2

            self.observation_spaces[agent] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(observation_size,),
                dtype=np.float32,
            )

        # Target and transition history do not exist until the first reset.
        self.target_position = None
        self.previous_snapshot = None
        self.episode_steps = 0

        # Reset may reseed this generator and derive independent physical and
        # target streams from it while unseeded resets continue its sequence.
        self._random_generator = np.random.default_rng()

    def observation_space(self, agent: AgentID) -> spaces.Space[Observation]:
        """Return the cached observation space belonging to one segment."""
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> spaces.Space[Action]:
        """Return the cached six-value normalized action space for one segment."""
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[
        dict[AgentID, Observation],
        dict[AgentID, dict[str, Any]],
    ]:
        """Begin an episode with independent physical and target randomness.

        Reset restores all agents, resets physics, samples one nearby target,
        records the initial snapshot, and returns observations and initial
        head-target distance. The same seed must reproduce the same episode.
        """
        # PettingZoo permits callers to supply an options dictionary. Version 1
        # defines no option-controlled reset behavior, so accepting it has no
        # effect on the seeded physical or target streams.
        del options

        # An explicit seed restarts the public sequence. Without one, the current
        # generator continues so consecutive episodes receive different states.
        if seed is not None:
            self._random_generator = np.random.default_rng(seed)

        # Separate child seeds prevent target sampling from affecting the leg
        # configuration generated for the same public seed.
        maximum_seed = np.iinfo(np.int32).max
        physical_seed = int(self._random_generator.integers(0, maximum_seed))
        target_seed = int(self._random_generator.integers(0, maximum_seed))
        target_generator = np.random.default_rng(target_seed)

        # Reset only physical state in the MuJoCo layer, then translate it into
        # the complete snapshot used by task-level calculations.
        self.simulation.reset(seed=physical_seed)
        snapshot = self.simulation.snapshot()
        target = self._sample_target(snapshot, target_generator)

        # All segments become active together and the reset snapshot becomes the
        # starting point for evaluating the first transition's reward.
        self.agents = self.possible_agents.copy()
        self.episode_steps = 0
        self.previous_snapshot = snapshot
        self.target_position = target

        # Reset publishes policy observations and the initial navigation distance.
        # Followers receive no target information, even in diagnostic infos.
        observations = build_observations(snapshot, target)
        infos = {agent: {} for agent in self.agents}
        infos[self.possible_agents[0]]["target_distance_m"] = float(
            np.linalg.norm(snapshot.head_tip_position[:2] - target)
        )

        return observations, infos

    def step(
        self,
        actions: dict[AgentID, Action],
    ) -> tuple[
        dict[AgentID, Observation],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        """Validate and apply one simultaneous 20 ms transition.

        The method must update snapshots and episode time, detect arrival before
        timeout, calculate every agent's reward, and remove all agents together
        when the episode ends.
        """
        # A transition needs the target and physical history established by reset.
        previous_snapshot = self.previous_snapshot
        target_position = self.target_position

        if not self.agents or previous_snapshot is None or target_position is None:
            raise RuntimeError("reset() must be called before step()")

        # Preserve the participants because a final transition must still return
        # results for them after the public active-agent list becomes empty.
        active_agents = self.agents.copy()
        joint_actions = self._validate_actions(actions)

        # Apply all segment actions in one shared physical transition, then copy
        # the resulting MuJoCo state into the task-facing snapshot format.
        self.simulation.step(joint_actions)
        current_snapshot = self.simulation.snapshot()
        self.episode_steps += 1

        # Arrival is measured after the transition in the ground plane. It takes
        # precedence over the time limit when both occur on the final step.
        target_distance = np.linalg.norm(
            current_snapshot.head_tip_position[:2] - target_position
        )
        target_reached = bool(target_distance <= ARRIVAL_RADIUS_M)

        terminated = target_reached
        truncated = not target_reached and self.episode_steps >= self.max_episode_steps

        episode_end: str | None = None
        if terminated:
            episode_end = "arrival"
        elif truncated:
            episode_end = "time_limit"

        # Reward and observation helpers consume the trusted completed
        # transition; they do not inspect public actions or raw MuJoCo state.
        reward_terms = calculate_reward_terms(
            previous_snapshot=previous_snapshot,
            current_snapshot=current_snapshot,
            target_position=target_position,
            target_reached=target_reached,
            config=self.reward_config,
        )
        rewards = {agent: reward_terms[agent].total for agent in active_agents}
        observations = build_observations(current_snapshot, target_position)

        # All segments share one physical body, so they share episode boundaries
        # even though their observations and scalar rewards remain separate.
        terminations = {agent: terminated for agent in active_agents}
        truncations = {agent: truncated for agent in active_agents}
        infos = self._build_infos(
            snapshot=current_snapshot,
            reward_terms=reward_terms,
            target_reached=target_reached,
            episode_end=episode_end,
        )
        infos[self.possible_agents[0]]["head_step_distance_m"] = float(
            np.linalg.norm(
                current_snapshot.head_tip_position[:2]
                - previous_snapshot.head_tip_position[:2]
            )
        )

        # Retain the final snapshot for inspection and as transition history when
        # the episode continues. Clear agents only after all final outputs exist.
        self.previous_snapshot = current_snapshot
        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _validate_actions(
        self,
        actions: Mapping[AgentID, np.ndarray],
    ) -> NDArray[np.float32]:
        """Validate the complete public action dictionary before physics changes.

        Accepted actions are returned as one trusted flat vector in segment order.
        Missing keys, extra keys, invalid shapes, non-finite values, and values
        outside ``[-1, 1]`` must raise instead of being clipped.
        """
        expected_agents = set(self.agents)
        received_agents = set(actions)
        if received_agents != expected_agents:
            missing = sorted(expected_agents - received_agents, key=repr)
            extra = sorted(received_agents - expected_agents, key=repr)
            raise ValueError(
                "Action keys must match active agents; "
                f"missing={missing}, extra={extra}"
            )

        action_size = self.simulation.leg_actuator_ids.shape[1]
        accepted_actions: list[NDArray[np.float32]] = []

        for agent in self.agents:
            raw_action = np.asarray(actions[agent])

            # Reject non-numeric values before conversion so strings and complex
            # values cannot be silently interpreted as real-valued controls.
            if not np.issubdtype(raw_action.dtype, np.number) or np.iscomplexobj(
                raw_action
            ):
                raise TypeError(f"Action for agent {agent} must contain real numbers")
            if raw_action.shape != (action_size,):
                raise ValueError(
                    f"Action for agent {agent} must have shape ({action_size},), "
                    f"got {raw_action.shape}"
                )
            if not np.isfinite(raw_action).all():
                raise ValueError(f"Action for agent {agent} must be finite")
            if np.any((raw_action < -1.0) | (raw_action > 1.0)):
                raise ValueError(f"Action for agent {agent} must be within [-1, 1]")

            # Convert exactly once after validating the original values, so a
            # float32 rounding cannot hide an input just outside the bounds.
            accepted_actions.append(raw_action.astype(np.float32, copy=False))

        # Concatenation follows active-agent order, which is also segment order.
        return np.concatenate(accepted_actions, dtype=np.float32)

    def _sample_target(
        self,
        snapshot: PhysicalSnapshot,
        random_generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample a nearby target relative to the initial head position and yaw."""
        # Sample polar coordinates inside the narrow wedge in front of the head.
        distance = random_generator.uniform(*TARGET_DISTANCE_RANGE_M)
        bearing = random_generator.uniform(
            -TARGET_BEARING_LIMIT_RAD, TARGET_BEARING_LIMIT_RAD
        )

        # Rotate the head's local forward axis (+x) into the world XY plane.
        # Its two planar components define the head's signed yaw angle.
        w, x, y, z = snapshot.body_quaternion[0]
        forward_world_x = 1.0 - 2.0 * (y * y + z * z)
        forward_world_y = 2.0 * (w * z + x * y)
        yaw = np.arctan2(forward_world_y, forward_world_x)

        # The bearing is relative to the head, so adding yaw converts it to a
        # world-space angle. Cosine and sine then form a unit direction vector.
        target_angle = yaw + bearing
        direction = np.array(
            [
                np.cos(target_angle),
                np.sin(target_angle),
            ],
            dtype=np.float64,
        )

        # A point plus a distance-scaled direction gives the target's absolute
        # planar position in world coordinates.
        return snapshot.head_tip_position[:2] + distance * direction

    def _build_infos(
        self,
        snapshot: PhysicalSnapshot,
        reward_terms: dict[AgentID, RewardTerms] | None = None,
        target_reached: bool = False,
        episode_end: str | None = None,
    ) -> dict[AgentID, dict[str, Any]]:
        """Publish agreed diagnostics without exposing unrestricted task state."""
        infos: dict[AgentID, dict[str, Any]] = {}

        for agent in self.possible_agents:
            # Contact facts come directly from the trusted final snapshot.
            info: dict[str, Any] = {
                "left_foot_ground_contact": bool(
                    snapshot.left_foot_ground_contact[agent]
                ),
                "right_foot_ground_contact": bool(
                    snapshot.right_foot_ground_contact[agent]
                ),
                "body_ground_contact": bool(snapshot.body_ground_contact[agent]),
                "leg_leg_contact": bool(snapshot.leg_leg_contact[agent]),
            }

            # Reward terms are already weighted, signed contributions. Exposing
            # them separately makes evaluation possible without recomputation.
            if reward_terms is not None:
                terms = reward_terms[agent]
                info.update(
                    reward_arrival=float(terms.arrival),
                    reward_efficiency=float(terms.efficiency),
                    reward_body_contact=float(terms.body_contact),
                    reward_leg_contact=float(terms.leg_contact),
                )

            infos[agent] = info

        if self.target_position is None:
            raise RuntimeError("Target information is unavailable before reset")

        # Navigation and episode summaries belong only to the head's diagnostics.
        head = self.possible_agents[0]
        infos[head]["target_distance_m"] = float(
            np.linalg.norm(snapshot.head_tip_position[:2] - self.target_position)
        )
        infos[head]["target_reached"] = bool(target_reached)

        if episode_end is not None:
            infos[head].update(
                episode_end=episode_end,
                episode_steps=self.episode_steps,
                episode_time_s=float(self.episode_steps * self.simulation.dt),
            )

        return infos

    def render(self) -> None | NDArray[np.uint8] | NDArray[np.float32]:
        """Delegate rendering to the internal Gymnasium MuJoCo simulation."""
        # The simulation owns Gymnasium's renderer and knows the render mode
        # selected when the model was constructed. The public task adds no
        # second viewer or image-processing layer.
        rendered = self.simulation.render()

        # Gymnasium's shared renderer annotation includes modes such as
        # segmentation and combined RGB/depth output. This environment exposes
        # only human, RGB-array, and depth-array rendering, so the result follows
        # the narrower public return type declared above.
        return cast(None | NDArray[np.uint8] | NDArray[np.float32], rendered)

    def close(self) -> None:
        """Release viewer and rendering resources owned by the simulation."""
        # Gymnasium's MuJoCo environment owns every viewer and renderer resource.
        self.simulation.close()
