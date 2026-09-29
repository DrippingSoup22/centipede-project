# Data type reference

This file is a compact reference for the named records and aliases that carry
data through the Centipede pipeline. It includes project-owned data structures,
the RL_lib records directly consumed by this project, and fixed serialized
schemas. Behavioral components such as environments, coordinators, algorithms,
neural networks, and normalizers are outside this reference.

Array dimensions use these symbols:

- `N`: number of centipede segments.
- `A`: action size for one segment; currently 6.
- `O`: observation size for one agent; currently 56, 81, or 54.
- `T`: number of transitions in one trajectory fragment.
- `B`: number of samples in one combined PPO batch.
- `E`: number of complete environment replicas in a pool.

## Index

### Environment

- [`AgentID`](#agentid)
- [`Observation`](#observation)
- [`Action`](#action)
- [`PhysicalSnapshot`](#physicalsnapshot)
- [`RewardConfig`](#rewardconfig)
- [`RewardTerms`](#rewardterms)
- [Agent info mapping](#agent-info-mapping)

### Training

- [`AgentObservations`](#agentobservations)
- [`AgentActions`](#agentactions)
- [`EnvironmentStep`](#environmentstep)
- [`LearnerConfig`](#learnerconfig)
- [`SegmentLearner`](#segmentlearner)
- [`RolloutConfig`](#rolloutconfig)
- [`TrajectoryFragment`](#trajectoryfragment)
- [`PPOBatch`](#ppobatch)
- [`CheckpointMetadata`](#checkpointmetadata)

### Experiment and evaluation

- [`ExperimentConfig`](#experimentconfig)
- [`TrainingPreset`](#trainingpreset)
- [`TrainingPlan`](#trainingplan)
- [`EvaluationPlan`](#evaluationplan)
- [`ComparisonVariant`](#comparisonvariant)
- [`ComparisonPlan`](#comparisonplan)
- [`TrainingResult`](#trainingresult)
- [`AgentDiagnostics`](#agentdiagnostics)
- [`EpisodeDiagnostics`](#episodediagnostics)
- [`PolicySummary`](#policysummary)
- [`PolicyEvaluation`](#policyevaluation)
- [`RunEvaluation`](#runevaluation)

### RL_lib records used directly

- [`EpisodeStep`](#episodestep)
- [`RolloutArrays`](#rolloutarrays)
- [`ContinuousPPOActionSample`](#continuousppoactionsample)
- [`PPOUpdateResult`](#ppoupdateresult)

### Serialized schemas

- [Checkpoint state mapping](#checkpoint-state-mapping)
- [Checkpoint learner-state mapping](#checkpoint-learner-state-mapping)
- [Resolved training snapshot](#resolved-training-snapshot)

## Environment types

### `AgentID`

**Definition:** `int`

Identifies the agent that controls one physical segment. IDs come from the
loaded MuJoCo model, are consecutive from `0` to `N - 1`, and also define the
head-to-rear segment order.

### `Observation`

**Definition:** `NDArray[np.float32]`

The flat partial observation supplied to one agent. Its shape is `(56,)` for
the head, `(81,)` for an interior segment, and `(54,)` for the rear under the
current radius-one visibility rule.

### `Action`

**Definition:** `NDArray[np.float32]`

The six bounded motor commands produced by one segment policy. Its shape is
`(A,)`, its values lie in `[-1, 1]`, and its fixed order is left sweep, left
lift, left knee, right sweep, right lift, right knee.

### `PhysicalSnapshot`

**Source:** `centipede.environment.simulation`

An immutable, copied view of the complete physical state after one MuJoCo
transition. The environment uses it to construct partial observations, rewards,
targets, contacts, and diagnostics without exposing mutable MuJoCo arrays.

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `body_height` | `float64 (N,)` | World Z coordinate of every segment center. |
| `body_quaternion` | `float64 (N, 4)` | World orientation of each segment in MuJoCo `w, x, y, z` order. |
| `leg_joint_position` | `float64 (N, A)` | Angular position of every controlled leg degree of freedom. |
| `leg_joint_velocity` | `float64 (N, A)` | Angular velocity of every controlled leg degree of freedom. |
| `body_linear_velocity` | `float64 (N, 3)` | Segment-center linear velocity in that segment's local frame. |
| `body_angular_velocity` | `float64 (N, 3)` | Segment-center angular velocity in that segment's local frame. |
| `body_planar_position` | `float64 (N, 2)` | World XY coordinate of every segment center. |
| `left_foot_ground_contact` | `bool (N,)` | Whether each left foot touches the floor. |
| `right_foot_ground_contact` | `bool (N,)` | Whether each right foot touches the floor. |
| `body_ground_contact` | `bool (N,)` | Whether each segment body touches the floor. |
| `leg_leg_contact` | `bool (N,)` | Whether a leg owned by each segment touches another leg. |
| `head_tip_position` | `float64 (3,)` | World XYZ coordinate of the head-tip site. |

### `RewardConfig`

**Source:** `centipede.environment.rewards`

The immutable coefficient set used by the first reward formulation. It defines
the relative size of arrival, efficiency, and contact terms without storing any
episode state.

| Field | Type | Meaning |
| --- | --- | --- |
| `c_arrival` | `float` | Shared positive reward when the head reaches the target. |
| `c_efficiency` | `float` | Scale of the head-progress or follower-path distance-ratio cost. |
| `c_body` | `float` | Cost applied for one body-ground contact step. |
| `c_leg` | `float` | Cost applied for one leg-leg contact step. |
| `epsilon_ratio` | `float` | Small offset that keeps a distance ratio defined near zero. |

### `RewardTerms`

**Source:** `centipede.environment.rewards`

The four signed reward contributions calculated for one agent and one
transition. Its `total` property sums the fields to produce the scalar reward
returned by the environment.

| Field | Type | Meaning |
| --- | --- | --- |
| `arrival` | `float` | Shared nonnegative target-arrival contribution. |
| `efficiency` | `float` | Nonpositive head-progress or follower-path contribution. |
| `body_contact` | `float` | Nonpositive contribution from the agent's body-ground flag. |
| `leg_contact` | `float` | Nonpositive contribution from the agent's leg-leg flag. |

### Agent info mapping

**Runtime type:** `dict[AgentID, dict[str, Any]]`

PettingZoo returns one diagnostic dictionary per agent on reset and step. All
agents receive contact fields; reward fields appear after transitions; task and
episode fields belong only to the head.

| Field | Present for | Meaning |
| --- | --- | --- |
| `left_foot_ground_contact` | Every agent | Whether the left foot touches the floor in the current state. |
| `right_foot_ground_contact` | Every agent | Whether the right foot touches the floor in the current state. |
| `body_ground_contact` | Every agent | Whether the agent's body touches the floor. |
| `leg_leg_contact` | Every agent | Whether one of the agent's legs touches another leg. |
| `reward_arrival` | Every agent after `step` | Signed arrival contribution for the transition. |
| `reward_efficiency` | Every agent after `step` | Signed efficiency contribution for the transition. |
| `reward_body_contact` | Every agent after `step` | Signed body-contact contribution for the transition. |
| `reward_leg_contact` | Every agent after `step` | Signed leg-contact contribution for the transition. |
| `target_distance_m` | Head | Current planar distance from the head tip to the target in metres. |
| `head_step_distance_m` | Head after `step` | Planar distance travelled by the head tip during the transition. |
| `target_reached` | Head | Whether the target tolerance was reached in the current state. |
| `episode_end` | Head at episode end | Either `arrival` or `time_limit`. |
| `episode_steps` | Head at episode end | Number of completed control transitions in the episode. |
| `episode_time_s` | Head at episode end | Simulated episode duration in seconds. |

## Training types

### `AgentObservations`

**Definition:** `dict[AgentID, Observation]`

One environment replica's simultaneous observations, keyed by segment agent.
The pool retains this PettingZoo representation instead of flattening agents or
mixing data between learners.

### `AgentActions`

**Definition:** `dict[AgentID, Action]`

One environment replica's simultaneous bounded actions, keyed by segment agent.
A rollout step sends one such dictionary for every replica in stable index order.

### `EnvironmentStep`

**Source:** `centipede.training.environment_pool`

The unchanged five-part result of one PettingZoo parallel transition: agent
observations, rewards, terminations, truncations, and diagnostic infos. Pool
results are ordered by stable environment-replica index.

### `LearnerConfig`

**Source:** `centipede.training.learners`

The immutable architecture, optimizer, PPO, and observation-normalization
settings shared in form by all segment learners. Each segment still receives
separate models, optimizers, normalizer state, samples, and gradients.

| Field | Type | Meaning |
| --- | --- | --- |
| `hidden_sizes` | `tuple[int, ...]` | Width of each hidden layer in both actor and critic networks. |
| `initial_standard_deviation` | `float` | Initial Gaussian action-distribution standard deviation. |
| `standard_deviation_mode` | `str` | RL_lib strategy used to represent the Gaussian standard deviation. |
| `actor_learning_rate` | `float` | Adam learning rate for the policy network. |
| `critic_learning_rate` | `float` | Adam learning rate for the value network. |
| `clip_ratio` | `float` | PPO probability-ratio clipping threshold. |
| `entropy_coefficient` | `float` | Weight of the entropy term that discourages premature policy collapse. |
| `maximum_gradient_norm` | `float` | Maximum combined gradient norm before gradient clipping. |
| `normalization_mode` | `str` | ObservationNormalizer mode, currently `running`. |
| `normalizer_epsilon` | `float` | Numerical-stability offset used by observation normalization. |
| `normalizer_clip` | `float` | Absolute bound applied to normalized observation values. |

### `SegmentLearner`

**Source:** `centipede.training.learners`

The mutable bundle that owns one agent's independent learning components and
fixed dimensions. Environment replicas share this same learner for the same
segment; different segments never share a learner.

| Field | Type | Meaning |
| --- | --- | --- |
| `agent_id` | `AgentID` | Segment controlled by this learner. |
| `observation_size` | `int` | Number of scalar inputs accepted by its networks. |
| `action_size` | `int` | Number of continuous actions produced by its policy. |
| `ppo` | `rl_lib.PPO` | Actor, critic, optimizers, sampling RNG, and PPO update behavior. |
| `normalizer` | `rl_lib.ObservationNormalizer` | Independent running observation statistics for this agent. |

### `RolloutConfig`

**Source:** `centipede.training.rollout`

The immutable collection, advantage-estimation, and update settings used by the
rollout coordinator. A window contains `rollout_window_steps` transitions from
each environment replica before the synchronized PPO updates begin.

| Field | Type | Meaning |
| --- | --- | --- |
| `rollout_window_steps` | `int` | Transitions collected from each environment in one rollout window. |
| `discount` | `float` | Gamma factor applied to future rewards and bootstrap values. |
| `gae_lambda` | `float` | GAE trace factor controlling the bias-variance tradeoff. |
| `update_epochs` | `int` | Passes over the finalized batch during one PPO update. |
| `minibatch_size` | `int` | Maximum samples used by one optimizer step. |

### `TrajectoryFragment`

**Source:** `centipede.training.rollout`

A mutable, per-environment and per-agent buffer that never crosses an episode
reset. It accumulates consecutive transitions until termination, truncation, or
the end of the current rollout window, then becomes one `PPOBatch`.

| Field | Type | Meaning |
| --- | --- | --- |
| `observation_size` | `int` | Expected scalar size of every stored normalized observation. |
| `action_size` | `int` | Expected scalar size of every stored continuous action. |
| `steps` | `list[EpisodeStep]` | Consecutive normalized states, actions, latent actions, and rewards. |
| `old_log_probabilities` | `list[float]` | Behavior-policy log probability stored for every step. |
| `values` | `list[float]` | Critic estimate stored for every step. |

### `PPOBatch`

**Source:** `centipede.training.rollout`

One agent's finalized, immutable training arrays. Fragments from several
environment replicas or episode pieces can be concatenated along `B`, but data
from different agents are never combined.

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `observations` | `float32 (B, O)` | Normalized observations used as actor and critic inputs. |
| `latent_actions` | `float32 (B, A)` | Unsquashed Gaussian samples used for stable PPO reevaluation. |
| `old_log_probabilities` | `float32 (B,)` | Log probabilities recorded before the policy update. |
| `advantages` | `float32 (B,)` | Generalized advantage estimates for the policy objective. |
| `return_targets` | `float32 (B,)` | Bootstrapped return targets used to train the critic. |

### `CheckpointMetadata`

**Source:** `centipede.training.checkpoints`

The validated training position and configuration returned after a checkpoint
is restored. It reports what was loaded; model, optimizer, normalizer, and RNG
states are restored directly into the supplied learners.

| Field | Type | Meaning |
| --- | --- | --- |
| `total_environment_transitions` | `int` | Total transitions collected across all environment replicas. |
| `completed_updates` | `int` | Number of completed synchronized PPO window updates. |
| `learner_config` | `LearnerConfig` | Network, optimizer, PPO, and normalizer settings saved with the checkpoint. |
| `rollout_config` | `RolloutConfig` | Collection and update settings saved with the checkpoint. |

## Experiment and evaluation types

### `ExperimentConfig`

**Source:** `centipede.experiment.config`

The fully resolved inputs required to construct one training or evaluation
application. Unlike reusable presets, its paths point to one concrete model and
one concrete run directory.

| Field | Type | Meaning |
| --- | --- | --- |
| `model_path` | `Path` | Resolved MuJoCo XML model used by every environment replica. |
| `run_dir` | `Path` | Concrete directory containing this run's checkpoints and artifacts. |
| `training_seed` | `int` | Root seed from which environment and learner seeds are derived. |
| `environment_count` | `int` | Number of environment replicas used for data collection. |
| `worker_count` | `int` | Number of CPU execution workers; one selects the serial reference path. |
| `update_cycles` | `int` | Number of collect-and-update cycles in the run. |
| `max_episode_steps` | `int` | Time-limit transitions before an unfinished episode is truncated. |
| `evaluation_seeds` | `tuple[int, ...]` | Held-out episode seeds used only during evaluation. |
| `learner_config` | `LearnerConfig` | Resolved learner settings. |
| `rollout_config` | `RolloutConfig` | Resolved collection and update settings. |

### `TrainingPreset`

**Source:** `centipede.experiment.presets`

The validated training settings of a plan file. It points to an output root
rather than one run directory; a workflow creates a unique run and then
converts the preset into an `ExperimentConfig`.

| Field | Type | Meaning |
| --- | --- | --- |
| `model_path` | `Path` | Resolved MuJoCo XML model selected by the preset. |
| `output_root` | `Path` | Parent directory under which unique runs are created. |
| `training_seed` | `int` | Root random seed for the training run. |
| `environment_count` | `int` | Number of environment replicas to construct. |
| `worker_count` | `int` | Number of CPU workers assigned to those replicas. |
| `update_cycles` | `int` | Number of rollout collection and PPO update cycles. |
| `max_episode_steps` | `int` | Maximum transitions in one environment episode. |
| `learner_config` | `LearnerConfig` | Validated learner settings from the TOML file. |
| `rollout_config` | `RolloutConfig` | Validated rollout settings from the TOML file. |

### `TrainingPlan`

**Source:** `centipede.experiment.presets`

A checked plan file of kind `training`. The training workflow creates a unique
run under the preset's output root, trains it, and evaluates it when seeds are
present.

| Field | Type | Meaning |
| --- | --- | --- |
| `plan_file` | `Path` | Plan file copied into the run as `training_preset.toml`. |
| `preset` | `TrainingPreset` | Validated training settings. |
| `evaluation_seeds` | `tuple[int, ...]` | Held-out seeds for evaluating the new run; empty when the plan has no `[evaluation]` table. |

### `EvaluationPlan`

**Source:** `centipede.experiment.presets`

A checked plan file of kind `evaluation`. It deliberately owns only the source
run and held-out seeds, so the run's saved snapshot remains the source of model,
environment, and learner settings.

| Field | Type | Meaning |
| --- | --- | --- |
| `plan_file` | `Path` | Plan file copied into the evaluation directory as `evaluation_preset.toml`. |
| `source_run` | `Path` | Existing run directory whose checkpoints are evaluated. |
| `evaluation_seeds` | `tuple[int, ...]` | Distinct nonnegative seeds applied to every checkpoint and baseline. |

### `ComparisonVariant`

**Source:** `centipede.experiment.presets`

One training preset derived from a comparison's base plan.

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | Directory and report name; generated from varying grid values or given by `[[variant]]`. |
| `changes` | `dict[str, Any]` | Changed settings keyed as `section.field`. |
| `preset` | `TrainingPreset` | Validated base settings with the changes applied. |

### `ComparisonPlan`

**Source:** `centipede.experiment.presets`

A checked plan file of kind `comparison`. Every variant is validated when the
plan is loaded, before the first variant trains.

| Field | Type | Meaning |
| --- | --- | --- |
| `plan_file` | `Path` | Plan file copied into the comparison as `comparison_preset.toml`. |
| `output_root` | `Path` | Parent directory under which the unique comparison directory is created. |
| `variants` | `tuple[ComparisonVariant, ...]` | Variants in execution order. |
| `evaluation_seeds` | `tuple[int, ...]` | Held-out seeds for evaluating every variant; empty for timing-only comparisons. |

### `TrainingResult`

**Source:** `centipede.experiment.runner`

The compact result returned after a complete training call. It identifies the
run and final checkpoint and reports the amount of completed training work.

| Field | Type | Meaning |
| --- | --- | --- |
| `run_dir` | `Path` | Directory containing the completed run's artifacts. |
| `checkpoint_path` | `Path` | Path of the final synchronized checkpoint. |
| `completed_update_cycles` | `int` | Number of collect-and-update cycles completed. |
| `total_environment_transitions` | `int` | Total transitions collected across every replica. |

### `AgentDiagnostics`

**Source:** `centipede.experiment.evaluation`

One agent's undiscounted episode totals or its means across evaluated episodes.
Contact values count flagged control transitions, while action magnitude is
already a per-transition mean.

| Field | Type | Meaning |
| --- | --- | --- |
| `return_total` | `float` | Sum of the agent's scalar rewards. |
| `reward_arrival` | `float` | Sum or mean of arrival contributions. |
| `reward_efficiency` | `float` | Sum or mean of efficiency contributions. |
| `reward_body_contact` | `float` | Sum or mean of body-contact contributions. |
| `reward_leg_contact` | `float` | Sum or mean of leg-contact contributions. |
| `body_contact_steps` | `float` | Count or mean count of body-ground contact transitions. |
| `leg_contact_steps` | `float` | Count or mean count of leg-leg contact transitions. |
| `left_foot_ground_steps` | `float` | Count or mean count of left-foot ground-contact transitions. |
| `right_foot_ground_steps` | `float` | Count or mean count of right-foot ground-contact transitions. |
| `mean_action_l2` | `float` | Mean Euclidean magnitude of the agent's six-value action. |

### `EpisodeDiagnostics`

**Source:** `centipede.experiment.evaluation`

The complete diagnostic record for one frozen policy on one held-out seed. It
combines per-agent measurements with the shared task outcome and head motion.

| Field | Type | Meaning |
| --- | --- | --- |
| `seed` | `int` | Held-out environment seed used for this episode. |
| `agent_diagnostics` | `dict[AgentID, AgentDiagnostics]` | One diagnostic record for every segment agent. |
| `target_reached` | `bool` | Whether the head entered the target tolerance radius. |
| `episode_end` | `str` | Episode stopping reason, currently `arrival` or `time_limit`. |
| `episode_steps` | `int` | Number of completed control transitions. |
| `episode_time_s` | `float` | Simulated duration in seconds. |
| `initial_target_distance_m` | `float` | Head-to-target planar distance immediately after reset. |
| `final_target_distance_m` | `float` | Head-to-target planar distance in the final state. |
| `head_distance_traveled_m` | `float` | Sum of the head tip's planar displacement over the episode. |

### `PolicySummary`

**Source:** `centipede.experiment.evaluation`

The arithmetic means across all held-out episodes for one checkpoint or
baseline. It provides one comparable result while preserving the individual
episodes separately in `PolicyEvaluation`.

| Field | Type | Meaning |
| --- | --- | --- |
| `agent_diagnostics` | `dict[AgentID, AgentDiagnostics]` | Across-episode mean diagnostics for every segment agent. |
| `success_rate` | `float` | Fraction of held-out episodes that reached the target. |
| `mean_episode_steps` | `float` | Mean number of transitions before the episode ended. |
| `mean_episode_time_s` | `float` | Mean simulated episode duration in seconds. |
| `mean_initial_target_distance_m` | `float` | Mean initial head-to-target distance. |
| `mean_final_target_distance_m` | `float` | Mean final head-to-target distance. |
| `mean_head_distance_traveled_m` | `float` | Mean total planar distance travelled by the head tip. |

### `PolicyEvaluation`

**Source:** `centipede.experiment.evaluation`

All evaluation data for one frozen checkpoint, zero-action baseline, or random
baseline. Optional checkpoint fields are `None` for baselines because they have
no training state.

| Field | Type | Meaning |
| --- | --- | --- |
| `label` | `str` | Human-readable checkpoint or baseline name. |
| `checkpoint_path` | `Path \| None` | Source checkpoint path, or `None` for a baseline. |
| `completed_updates` | `int \| None` | PPO updates stored in the checkpoint, or `None` for a baseline. |
| `total_environment_transitions` | `int \| None` | Collected transitions stored in the checkpoint, or `None` for a baseline. |
| `episodes` | `tuple[EpisodeDiagnostics, ...]` | Detailed results in evaluation-seed order. |
| `summary` | `PolicySummary` | Means calculated from those episode records. |

### `RunEvaluation`

**Source:** `centipede.experiment.evaluation`

The top-level result of evaluating one training run. It keeps every checkpoint
and both reference policies under the same seed set so their results remain
directly comparable.

| Field | Type | Meaning |
| --- | --- | --- |
| `seeds` | `tuple[int, ...]` | Held-out seeds shared by all evaluated policies. |
| `checkpoints` | `tuple[PolicyEvaluation, ...]` | Checkpoint evaluations in training order. |
| `baselines` | `tuple[PolicyEvaluation, ...]` | Zero-action and random-action evaluations. |

## RL_lib records used directly

These records are defined in `../RL_lib/src/rl_lib/data` and reused rather than
reimplemented in Centipede.

### `EpisodeStep`

One generic RL_lib transition record. `TrajectoryFragment` stores one for each
agent transition, using the normalized observation as `state` and retaining the
unsquashed policy action needed by continuous PPO.

| Field | Type | Meaning |
| --- | --- | --- |
| `state` | `NDArray[np.float32]` | Normalized observation before the action. |
| `action` | `NDArray[np.float32]` | Bounded action sent to the environment. |
| `reward` | `float` | Scalar reward returned for the transition. |
| `policy_action` | `NDArray[np.float32] \| None` | Unsquashed Gaussian sample; required for this continuous rollout. |

### `RolloutArrays`

RL_lib's validated array form of consecutive `EpisodeStep` records. Centipede
uses it while finalizing a fragment, then adds PPO-specific log probabilities,
advantages, and return targets to create a `PPOBatch`.

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `observations` | `float32 (T, O)` | Consecutive normalized states. |
| `actions` | `float32 (T, A)` | Latent continuous policy actions in this project. |
| `rewards` | `float32 (T,)` | Consecutive scalar rewards. |
| `final_state` | `float32 (O,)` | State after the final stored transition for bootstrapping. |

### `ContinuousPPOActionSample`

The complete result of sampling one continuous PPO policy. It freezes the
behavior-policy measurements needed later, after the learner parameters may
have changed.

| Field | Type | Meaning |
| --- | --- | --- |
| `action` | `NDArray[np.float32]` | Bounded action sent to the environment. |
| `latent_action` | `NDArray[np.float32]` | Unsquashed Gaussian action used for PPO reevaluation. |
| `log_probability` | `float` | Behavior-policy log probability of the sampled action. |
| `value` | `float` | Critic estimate for the sampled observation. |

### `PPOUpdateResult`

The diagnostic result returned by RL_lib for one PPO minibatch optimizer step.
The rollout coordinator groups these records by agent so losses are never mixed
between segment learners.

| Field | Type | Meaning |
| --- | --- | --- |
| `actor_loss` | `float` | Policy-objective loss for the minibatch update. |
| `critic_loss` | `float` | State-value loss for the minibatch update. |
| `entropy` | `float` | Mean action-distribution entropy. |
| `approximate_kl` | `float` | Approximate KL divergence from the behavior policy. |
| `clip_fraction` | `float` | Fraction of samples whose PPO ratio was clipped. |

## Serialized schemas

These mappings do not currently have Python dataclass names, but their fields
form validated, versioned interfaces stored on disk.

### Checkpoint state mapping

The top-level dictionary saved in every `.pt` checkpoint. It is one synchronized
bundle for all segment learners and deliberately excludes live environments,
unfinished rollout fragments, MuJoCo state, and targets.

| Field | Stored type | Meaning |
| --- | --- | --- |
| `format_version` | `int` | Checkpoint schema version, currently 2. |
| `total_environment_transitions` | `int` | Transitions collected across all replicas before this checkpoint. |
| `completed_updates` | `int` | Synchronized PPO window updates completed before saving. |
| `learner_config` | `dict[str, Any]` | Serialized fields of `LearnerConfig`. |
| `rollout_config` | `dict[str, Any]` | Serialized fields of `RolloutConfig`. |
| `learners` | `dict[AgentID, mapping]` | One learner-state mapping for every segment. |
| `torch_rng_state` | `torch.Tensor` | Global PyTorch random-generator state. |

### Checkpoint learner-state mapping

One entry inside the checkpoint's `learners` mapping. It preserves everything
needed to continue training a single segment learner while the outer checkpoint
keeps all agents synchronized.

| Field | Stored type | Meaning |
| --- | --- | --- |
| `observation_size` | `int` | Network input size used by this learner. |
| `action_size` | `int` | Policy output size used by this learner. |
| `actor_model_state` | `dict[str, Tensor]` | Actor parameters and registered buffers. |
| `critic_model_state` | `dict[str, Tensor]` | Critic parameters and registered buffers. |
| `actor_optimizer_state` | `dict[str, Any]` | Adam state associated with the actor. |
| `critic_optimizer_state` | `dict[str, Any]` | Adam state associated with the critic. |
| `normalizer_state` | `dict[str, Any]` | ObservationNormalizer mode and accumulated statistics. |
| `ppo_rng_state` | `dict[str, Any]` | NumPy generator state used by this PPO instance. |

### Resolved training snapshot

The `training_resolved.json` record saved before a training run begins. It lets
later evaluation reconstruct the exact training configuration without depending
on a reusable TOML file that may have changed.

| Field | Stored type | Meaning |
| --- | --- | --- |
| `format_version` | `int` | Snapshot schema version, currently 3. |
| `training` | `dict[str, Any]` | Resolved paths, seed, replica count, update cycles, and episode limit. |
| `learner` | `dict[str, Any]` | Serialized fields of `LearnerConfig`. |
| `rollout` | `dict[str, Any]` | Serialized fields of `RolloutConfig`. |

The nested `training` mapping contains `model_path`, `output_root`,
`training_seed`, `environment_count`, `worker_count`, `update_cycles`, and
`max_episode_steps`, with the same meanings documented in `TrainingPreset`.
Version-1 snapshots remain readable and restore `worker_count = 1`, matching
their original serial execution. Versions 1 and 2 stored `update_cycles` as
`rollout_windows` and `rollout_window_steps` as `steps_per_environment`; the
loader translates those names. Version-1 checkpoints are translated the same
way.
