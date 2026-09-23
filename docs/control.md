# Agent control and learning

This document owns the segment-agent design, observations, action ownership,
and future network/training architecture. Confirmed requirements and proposals
are identified separately; no agent networks or training code exist yet.
See [model.md](model.md) for mechanics and [environment.md](environment.md) for
the task, rewards, and evaluation considerations.

## Cooperative control

- Use one agent per body segment, responsible for that segment's control, rather
  than one agent controlling the entire centipede.
- Each agent has partial observations. "Above and behind" means ahead and behind
  along the body chain, not spatial proximity or vertical position.
- Fix the initial observation radius at one: an agent observes only its immediate
  existing neighbor ahead and its immediate existing neighbor behind. Larger
  radii are later experiment variants, not an initial hyperparameter.
- Do not pad missing end neighbors or add neighbor masks in the first version.
  The independent networks may have different input sizes: the head and rear
  have one neighbor block, while an interior segment has two.
- Cooperation is a core requirement: agents must learn to coordinate to produce
  walking, improve that walking, and achieve the desired wave-like gait.
- The head's role is to reach the navigation target. The other segments support
  its movement through their own local control; they do not each have a separate
  navigation target. Target arrival gives a shared reward to all agents.
- Every segment uses separate policy and value networks and learns only from its
  own rollout samples. Parameter sharing, centralized critics, pooled losses, and
  direct network communication are prohibited. Detailed optimization settings
  remain open.

## First-version observation contract

The observation follows the useful parts of Gymnasium's Ant convention: body
pose and joint positions form a `qpos`-like group, while body and joint velocities
form a `qvel`-like group. External contact-force arrays are deliberately omitted.
MuJoCo uses SI units, and all final observations are flat `float32` arrays.

### Complete segment block

Every segment contributes the same 27-value block. Its fields appear in this
fixed order:

| Range | Size | Meaning | Frame and unit |
| --- | ---: | --- | --- |
| `0` | 1 | Main-body height | World vertical position, metres |
| `1:5` | 4 | Main-body orientation quaternion `(w, x, y, z)` | Body orientation relative to the world |
| `5:11` | 6 | Leg-joint angles | Radians, in the action order below |
| `11:14` | 3 | Main-body linear velocity | Segment-local frame, metres per second |
| `14:17` | 3 | Main-body angular velocity | Segment-local frame, radians per second |
| `17:23` | 6 | Leg-joint angular velocities | Radians per second, in the action order below |
| `23` | 1 | Left-foot ground contact | `0.0` or `1.0` |
| `24` | 1 | Right-foot ground contact | `0.0` or `1.0` |
| `25` | 1 | Main-body ground contact | `0.0` or `1.0` |
| `26` | 1 | Leg-to-leg contact involving this segment | `0.0` or `1.0` |

The six joint fields always use left shoulder sweep, left shoulder lift, left
knee, right shoulder sweep, right shoulder lift, and right knee order. Global
body `x` and `y` are excluded from policy observations. A quaternion contains one
orientation represented by four components; its components are not four angles.

Each contact value describes the final MuJoCo state after the completed 20 ms
environment transition. A contact that persists remains active on every step.
Leg collisions remain physically enabled. No spine position, velocity, action,
or contact field is exposed in the first version.

### Visible blocks and dimensions

An observation always begins with the observing segment's own complete block.
It then contains the immediate existing neighbor ahead and finally the immediate
existing neighbor behind. Radius one is fixed; absent end neighbors are neither
padded nor represented by masks.

| Agent | Block order | Additional values | Shape |
| --- | --- | --- | ---: |
| Head `0` | self `0`, behind `1` | target forward, target lateral | `(56,)` |
| Interior `1`-`6` | self `i`, ahead `i-1`, behind `i+1` | none | `(81,)` |
| Rear `7` | self `7`, ahead `6` | none | `(54,)` |

The declared spaces are unbounded Gymnasium boxes with the corresponding shape
and `float32` dtype. The environment performs one conversion from MuJoCo's
internal `float64` values when it assembles each output. Observation scaling and
normalization belong to the individual learners, not the environment.

### Head target values

Only the head receives target information. The two appended values are the
planar displacement from the `head_tip` site to the target, rotated into the
head's yaw frame:

- **forward:** positive in front of the head and negative behind it;
- **lateral:** positive to the head's left and negative to its right.

Both are measured in metres. They fully locate the target in the head's planar
frame without prescribing an action or turn. Followers receive no target field,
target-derived reward, learned message, or other direct target information.
Distance-only sensing and antenna-like limited perception remain possible later
experiments, not part of this interface.

## Network and training architecture

PPO is the initial learning algorithm. Every segment is an independent learner
with its own actor, critic, optimizers, observation normalization state, rollout
history, advantages, losses, and updates. Parameters and training samples are
never pooled or shared between segments. There is no centralized critic, shared
policy, shared loss, or learned communication channel.

The agents interact only through the shared MuJoCo dynamics, their permitted
environment observations, and agreed reward signals. A shared team reward means
copying the same scalar reward to each independent learner; it does not combine
their losses or data. Any later architecture change must preserve independent
learning.

The environment emits raw observations in its declared spaces. Observation
normalization belongs to each independent `RL_lib` learner and is never performed
by the environment. Evaluation restores each learner's saved normalization state
and must not update it.

Use the reusable package under `../RL_lib/src/rl_lib` for algorithms, data types,
models, policies, normalization, and related generic utilities. Do not import or
adapt the runners under `../RL_lib/experiments`; Centipede owns its complete
experiment application.

### First PPO integration configuration

The Phase 4 baseline uses separate two-layer `(64, 64)` ReLU actors and critics.
Each actor is a tanh-squashed Gaussian with six global trainable standard
deviations initialized to `0.5`. Actor and critic each use an independent Adam
optimizer with learning rate `3e-4` and no weight decay. No learning-rate
scheduler is active in Phase 4; cosine decay remains a later training proposal.

Each learner owns a running `ObservationNormalizer` with epsilon `1e-8` and
output clipping at `10.0`. A raw observation updates only its segment's running
statistics when it is used to select a training action. Rollouts store the exact
normalized observation used for sampling. Bootstrap and evaluation observations
use the saved statistics without updating them.

The first rollout and update settings are:

| Setting | Value |
| --- | ---: |
| Steps per environment | `256` |
| Discount factor | `0.999` |
| GAE lambda | `0.95` |
| PPO clip ratio | `0.2` |
| Update epochs | `4` |
| Minibatch size | `64` |
| Maximum gradient norm | `0.5` |
| Entropy coefficient | `0.001` |

An environment replica is a data collector, not another set of policies. With
`E` replicas, each segment learner receives `256 * E` samples per update. GAE is
calculated separately for every uninterrupted environment-agent trajectory
fragment before fragments from the same segment are concatenated. Samples,
advantages, normalizers, gradients, and losses are never combined across segment
IDs.

Phase 4 learner checkpoints contain all eight actors, critics, optimizers,
normalizers, PPO shuffle states, configurations, dimensions, and training
counters in one synchronized bundle. They are created only after completed PPO
updates and exclude rollout samples, live MuJoCo state, targets, and unfinished
episodes. The initial untrained policies require evaluation metrics, not a routine
zero-percent checkpoint.

## Parallel environment interface

The public task environment follows PettingZoo's `ParallelEnv` contract. It has
eight stable integer segment identifiers, `0` through `7`, ordered from head to
rear and retained for the complete episode. Thus `possible_agents` is
`[0, 1, 2, 3, 4, 5, 6, 7]`. A call to `step()`
receives a dictionary containing one six-value action for every segment and
returns dictionaries of observations, rewards, terminations, truncations, and
information keyed by the same identifiers. No agent takes a physics step before
another; the environment assembles all actions and advances the shared model once.

Every per-agent action space is a Gymnasium `Box` with shape `(6,)`, `float32`
values, and normalized bounds `[-1, 1]`. The public action spaces intentionally
omit the seven spine-yaw actuators. The internal `MujocoEnv` simulation expands
the 48 enabled leg commands into the model's 55-value control vector and writes
zero to each spine actuator.

The six entries for every segment have this fixed order:

| Index | Actuator role |
| ---: | --- |
| 0 | Left shoulder sweep |
| 1 | Left shoulder lift |
| 2 | Left knee |
| 3 | Right shoulder sweep |
| 4 | Right shoulder lift |
| 5 | Right knee |

### Action validation

Before changing MuJoCo control state, `step()` must validate the complete action
dictionary:

- Its keys are exactly the eight active segment IDs; missing or unknown agents
  are errors.
- Every value is numeric, can be represented as `float32`, and has shape `(6,)`.
- Every component is finite and lies within the closed interval `[-1, 1]`.
- Invalid input raises a clear exception before any physics step occurs.
- Actions are not silently clipped. Clipping would hide policy or runner errors
  and could make the action executed by MuJoCo differ from the action stored for
  PPO probability calculations.

This is the only Centipede runtime boundary that validates the eight-agent action
dictionary. After it succeeds, the environment passes a trusted 48-value leg
vector inward. The rollout coordinator and reward or observation helpers must not
repeat these checks. Gymnasium may still enforce its own control-vector shape at
the library boundary when the simulation receives the assembled 55-value vector.

### Actuator discovery and mapping

Do not depend on actuator order in the XML. At initialization, construct the six
expected names for each segment using the canonical pattern
`segment_{id:02d}_{side}_{role}_motor`, resolve each with MuJoCo's name lookup,
and cache the resulting actuator IDs in the fixed action order above.

Store the leg result as a two-dimensional `(N, 6)` table indexed by
`[segment_id, action_index]`, where `N` is derived from the controlled segment
owners in the model loaded at construction. For example, `[3, 4]` means segment
3's right shoulder-lift actuator. This is the human-facing fixed mapping: the two
indices keep segment and joint role separate and the role table above gives every
action index a name. Do not encode both meanings into decimal values such as
`34`, which would create a second identifier system unrelated to MuJoCo's numeric
IDs. For the v1 configuration, `N` is eight and the table shape is `(8, 6)`.

Use `model.actuator_user[:, 0]` as an independent ownership check: every resolved
leg or yaw actuator must contain the same segment ID as its name. Also verify
that each actuator targets the expected named joint, has a limited `[-1, 1]`
control range, and occurs exactly once. Initialization fails if an expected
actuator is missing, duplicated, assigned to another segment, or if any of the
model's 55 actuators is unclassified. This combines readable semantic names with
explicit XML ownership metadata and remains safe if XML element order changes.

These model-schema checks run once while `CentipedeSimulation` is initialized.
They are not repeated on every action. The cached mapping is trusted for the
lifetime of that loaded model.

Any accepted XML version that changes a name or actuator contract must update
this mapping and its schema tests in the same reviewed change. Name resolution at
startup is deliberately retained: 55 one-time lookups have no meaningful cost
relative to simulation, while they prevent a reordered XML file from silently
changing policy action ownership.

The joint control vector begins as 55 zeros. The cached mapping fills only the 48
leg actuator positions. The seven actuators named `segment_01_yaw_motor` through
`segment_07_yaw_motor` are verified but remain zero.

### Control interval

The accepted first-version `frame_skip` is 200. Gymnasium defines the duration of
one `MujocoEnv` step as `model.opt.timestep * frame_skip`; therefore the frozen
`0.0001 s` physics timestep produces a `0.02 s` control interval and a 50 Hz
agent decision frequency. PettingZoo adds no separate physical-time convention:
one parallel step is this same joint control transition for all eight agents.

If `render_fps` is declared for the Gymnasium simulation, set it to 50 so it
matches the control interval. Ordinary training remains unrendered. The 200-step
choice must receive a deterministic runtime and stability check in Stage 2; any
later change is an experiment configuration change because it alters control and
reward timing.

Every observation space is the Gymnasium `Box` defined by the first-version
observation contract above. The observation builder, rather than the Gymnasium
simulation class, owns the exact partial-observation layout.

The PettingZoo environment is an application boundary, not a communication
mechanism between policies. Its dictionaries route independent values by agent;
they do not imply shared networks, buffers, gradients, or losses.

## Prototype spine action mapping

The frozen v1 flat-ground physical baseline uses active yaw. A fully passive-yaw
spine may be evaluated later as a separate model variant; do not change the v1
actuator set in place. No neural policies are implemented yet.

For the **first learning stage**, spine control is disabled at the environment
boundary: all seven yaw actuator commands stay at zero and no spine action is
exposed to an agent. The yaw joints, springs, damping, limits, and passive motion
remain in the frozen XML. This makes every segment action a six-value leg vector
without creating another physical model. Spine state is omitted from the first
observation contract; adding controlled yaw later remains an environment and
policy-interface change, not an in-place model edit.

The model's 0.1 ms physics timestep is not the policy decision interval. The
environment holds each action for 200 physics steps and computes elapsed-time
reward terms from the resulting 20 ms control interval. Neural inference occurs
at 50 Hz, not at the 10 kHz physics integration rate.

At the physical-model level, each non-head segment owns its connection to the
preceding segment as well as its six leg motors. This gives 6N + (N-1) motors: 55
for N=8. At the initial policy boundary, however, every segment exposes exactly
six leg actions and all seven physical spine-yaw motors remain zero. Passive
pitch has no motor.

`actuator_user[:, 0]` stores the owning segment. The environment reads actuator
names, indices, joint coordinates, and torque limits directly from the loaded
MuJoCo model rather than assuming joint indices match action indices. All controls
are normalized to [-1, 1]; the XML motor gear scales them to physical torque.
For the active assembly the hard scales are 8 micro-N m for shoulder sweep,
18 for shoulder lift, 10 for knee, and 15 for spine yaw. These are physical-model
parameters documented and validated in `model.md`, not policy output biases.

Setting `spine.active_yaw` to false gives six motors per segment. That alternative
has compilation coverage but is not the frozen v1 dynamically validated model.
Leg-driven turning with a passive spine remains to be demonstrated. The shared
RL_lib has not been modified or integrated during the model-only phase.

## RL_lib compatibility audit

Audit date: 2026-08-31. The audit inspected both the reusable package under
`../RL_lib/src/rl_lib` and the reference applications under
`../RL_lib/experiments`. It made no changes to RL_lib.

### Library boundary

Only `src/rl_lib` is a dependency of Centipede. It provides the generic building
blocks: PPO, Gaussian policies, actor and value models, PPO action samples,
observation normalization, generalized advantage estimation, and update results.
The existing experiment runners demonstrate how those pieces can be used, but
are not library interfaces and must not be imported by Centipede.

Centipede implements anew everything at the application level:

- MuJoCo environment and simultaneous segment actions.
- Construction of eight independent PPO instances.
- Per-agent observation and reward routing.
- Fixed-length rollout coordination and episode continuation.
- Experiment configuration, seeding, training loop, and progress reporting.
- Project checkpoint bundles, evaluation metrics, and recordings.

Centipede must not redefine a generic algorithm, model, policy, or data type that
RL_lib already exports. If a missing capability is generic, discuss adding it to
`src/rl_lib`. If it is inherently specific to this body or task, implement the
smallest project extension here using the library component as its base.

### Verified reusable capabilities

| Requirement | Status |
| --- | --- |
| Bounded continuous PPO | Implemented with a tanh-squashed diagonal Gaussian and correct latent-action probability evaluation. |
| Six-value leg actions | Supported by arbitrary fixed-size Gaussian policy outputs and finite per-component bounds. |
| Independent learners | Supported by constructing eight separate PPO objects with separate models, optimizers, normalizers, seeds, and samples. |
| Rollout representation | `EpisodeStep` and `rollout_arrays` store and validate normalized observations, environment actions, latent continuous-policy actions, rewards, and the final state. Centipede only adds the frozen log probabilities and values required by PPO. |
| Partial-rollout updates | Already supported by the algorithm primitives: `PPO.update` accepts arbitrary fixed batches, `PPO.state_value` supplies a boundary bootstrap, and `generalized_advantage_estimates` handles terminal versus nonterminal boundaries. |
| PPO minibatch reuse | Implemented with frozen old log probabilities and targets, full-batch advantage normalization, shuffled epochs, and minibatches. |
| Deterministic evaluation actions | Implemented by selecting the bounded Gaussian mean. |

The existing RL_lib PPO experiment runner collects complete episodes, but this
is not a limitation of the PPO algorithm and the runner will not be reused.
Centipede's runner can end a collection window without ending the environment,
bootstrap each learner from its own critic, update each learner independently,
and resume from the same MuJoCo state.

The targeted PPO and environment suite passed 100 tests, and an artifact-free
continuous PPO run on `Pendulum-v1` completed. The RL_lib README and roadmap still
call continuous PPO planned; that is stale library documentation, not a missing
implementation.

### Independent rollout contract

During collection, all eight current policies act on the same environment step,
but every learner stores only its own values:

```text
segment observation -> its PPO sample -> its environment action
                    -> its reward/value/log probability history
                    -> its bootstrap, GAE, targets, and PPO update
```

A rollout window should keep every policy unchanged until the joint environment
samples for that window have been collected. At the boundary, each learner is
updated from its own buffer. Synchronizing when collection stops and updates
begin does not share information; it prevents some agents from changing policy
while other agents are still collecting the same window.

The coordinator may hold these eight buffers in a dictionary keyed by segment ID;
separate ownership does not require eight custom runner or buffer classes. Each
buffer uses RL_lib's `EpisodeStep` and `rollout_arrays`; its thin Centipede wrapper
stores only the PPO log probabilities and critic values absent from that generic
rollout type. The coordinator routes transitions and triggers updates but does
not recompute environment termination, truncate the episode at a rollout
boundary, or implement PPO math.

For a true terminal state, each learner uses zero final value and
`terminated=True`. At a time limit or ordinary rollout cutoff, each learner uses
its own next observation and critic value with `terminated=False`. Buffers are
then cleared independently and collection continues after a nonterminal cutoff.

### Current and later gaps

No RL_lib source change is required for the first Centipede prototype based on
the audited interfaces. Possible later generic gaps must be evaluated only when
needed:

- The source package currently has no explicit CPU/CUDA device selection; all
  audited PPO tensors and models remain on CPU despite CUDA being visible in the
  WSL runtime.
- A recurrent experiment would require generic recurrent models, policies, and
  sequence-aware PPO data in RL_lib. It is not part of the initial feed-forward
  design.
- A new generic checkpoint abstraction may be useful eventually, but the first
  eight-agent checkpoint bundle is Centipede application composition and belongs
  here.

Parameter sharing, centralized critics, pooled multi-agent rollout types, policy
groups, and learned communication are explicitly outside the intended design.
The first prototype uses eight separate six-action PPO learners and fixes all
seven spine-yaw actuator commands at zero.
