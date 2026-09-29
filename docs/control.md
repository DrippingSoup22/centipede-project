# Agent control and learning

This document owns the segment-agent design, observations, action ownership,
and network/training architecture. Confirmed requirements and proposals are
identified separately. The contract applies to both implementations; the
[CPU plan](../cpu/plan.md) and [GPU plan](../gpu/plan.md) describe how each
realizes it. See [model.md](model.md) for mechanics and
[environment.md](environment.md) for the task, rewards, and evaluation
considerations.

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

Observations are unbounded `float32` values with the shapes above. Observation
scaling and normalization belong to the individual learners, not the
environment.

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

### First learner configuration

The first learner configuration uses separate two-layer `(64, 64)` ReLU actors
and critics. Each actor is a tanh-squashed Gaussian with six global trainable
standard deviations initialized to `0.5`. Actor and critic each use an independent Adam
optimizer with learning rate `3e-4` and no weight decay. No learning-rate
scheduler is active; cosine decay remains a later training proposal.

Each learner owns a running `ObservationNormalizer` with epsilon `1e-8` and
output clipping at `10.0`. A raw observation updates only its segment's running
statistics when it is used to select a training action. Rollouts store the exact
normalized observation used for sampling. Bootstrap and evaluation observations
use the saved statistics without updating them.

The first rollout and update settings are:

| Setting | Value |
| --- | ---: |
| Rollout window steps per replica | `256` |
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

Learner checkpoints contain all eight actors, critics, optimizers,
normalizers, PPO shuffle states, configurations, dimensions, and training
counters in one synchronized bundle. They are created only after completed PPO
updates and exclude rollout samples, live physical state, targets, and unfinished
episodes. The initial untrained policies require evaluation metrics, not a routine
zero-percent checkpoint.

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

For a true terminal state, each learner uses zero final value and
`terminated=True`. At a time limit or ordinary rollout cutoff, each learner uses
its own next observation and critic value with `terminated=False`. Buffers are
then cleared independently and collection continues after a nonterminal cutoff.

## Action interface

There are eight stable integer segment identifiers, `0` through `7`, ordered
from head to rear and retained for the complete episode. At every transition
each segment supplies one action of six `float32` values in the normalized
range `[-1, 1]`. All eight actions are applied simultaneously: no agent takes a
physics step before another, and the shared model advances once. Actions are
never silently clipped, because the executed action must match the action
stored for PPO probability calculations. The policy boundary omits the seven
spine-yaw actuators, which receive zero commands.

The six entries for every segment have this fixed order:

| Index | Actuator role |
| ---: | --- |
| 0 | Left shoulder sweep |
| 1 | Left shoulder lift |
| 2 | Left knee |
| 3 | Right shoulder sweep |
| 4 | Right shoulder lift |
| 5 | Right knee |

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

These model-schema checks run once while the simulation is initialized.
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

The accepted first-version control interval holds each action for 200 physics
steps. The frozen `0.0001 s` physics timestep therefore produces a `0.02 s`
control interval and a 50 Hz agent decision frequency. One transition is this
joint control interval for all eight agents. Any later change is an experiment
configuration change because it alters control and reward timing.

How each implementation exposes these actions is described in the
[CPU plan](../cpu/plan.md) (PettingZoo dictionaries and Gymnasium spaces) and
the [GPU plan](../gpu/plan.md) (batched device arrays).

## Prototype spine action mapping

The frozen v1 flat-ground physical baseline uses active yaw. A fully passive-yaw
spine may be evaluated later as a separate model variant; do not change the v1
actuator set in place.

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
Leg-driven turning with a passive spine remains to be demonstrated.
