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
  direct network communication are prohibited. Allocation of any additional
  reward terms and detailed optimization settings remain open.

## Initial observation contract

Define one complete segment-state block containing:

- Six leg-joint positions and six leg-joint velocities.
- Body coordinates and orientation relative to the agreed reference frame.
- Body linear and angular velocity expressed in the segment's local frame.
- One left-foot and one right-foot ground-contact flag.
- One main-body ground-contact flag.
- One leg-to-leg contact flag for the segment.

Each agent receives the **entire block**, including all flags, for itself and for
every immediate existing neighbor at radius one. Therefore, an interior agent
observes three complete blocks: ahead, self, and behind. The head and rear each
observe two complete blocks because they have only one existing neighbor. No
padding or mask is added. Neighbor joint positions, joint velocities, body
coordinates, body linear and angular velocities, foot contacts, body contact, and
leg-leg contact are all observable; neighbor information is not reduced to body
motion alone.

No spine position, velocity, actuator, or contact field is included initially.
The exact body-coordinate/orientation reference frame and contact sampling
semantics (instantaneous versus any contact during the preceding control
interval) still need to be selected before implementation.

The leg-to-leg flag identifies whether a leg owned by that segment participates
in a leg-leg collision. It will support a local penalty for that learner. Leg
collisions remain physically enabled; the flag reports their result rather than
preventing the action. The penalty weight and whether the flag records contact
onset, duration, or any contact within one control interval remain open.

## Target observation proposal (not yet finalized)

- The user favors giving the head exact target coordinates initially, avoiding
  the complexity of antenna-like perception. Limited perception remains a
  possible later extension, not a current implementation requirement.
- Assistant recommendation: give the head the planar displacement to the target
  expressed in its heading frame (forward and lateral components). This preserves
  exact target information relative to the head without requiring it to learn
  world-coordinate subtraction and rotation. This representation is a proposal.
- An equivalent alternative is planar distance plus signed bearing. Distance
  alone is insufficient because it does not tell the head which direction leads
  to the target.
- Assistant recommendation: initially provide the target signal only to the head;
  other agents use agreed local environment observations to support it. Agents do
  not transmit target information or learned messages through their networks.

## Network and training architecture

PPO is the initial learning algorithm. Every segment is an independent learner
with its own actor, critic, optimizers, observation normalization state, rollout
history, advantages, losses, and updates. Parameters and training samples are
never pooled or shared between segments. There is no centralized critic, shared
policy, shared loss, or learned communication channel.

The agents interact only through the shared MuJoCo dynamics, their permitted
environment observations, and agreed reward signals. A shared team reward means
copying the same scalar reward to each independent learner; it does not combine
their losses or data. Network sizes, recurrence, and observation history remain
open, but any later choice must preserve independent learning.

Use the reusable package under `../RL_lib/src/rl_lib` for algorithms, data types,
models, policies, normalization, and related generic utilities. Do not import or
adapt the runners under `../RL_lib/experiments`; Centipede owns its complete
experiment application.

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

The model's 0.1 ms physics timestep is not the future policy decision interval.
The environment should hold each action for an agreed integer number of physics
steps and compute elapsed-time reward terms from the resulting control interval.
That interval remains undecided; do not run neural inference at 10 kHz merely
because the physics engine integrates at that rate.

Each non-head segment owns its connection to the preceding segment as well as
its six leg motors. This gives 6N + (N-1) motors: 55 for N=8. The head has six
actions and the other segments seven; passive pitch has no motor.

`actuator_user[:, 0]` stores the owning segment. The future environment should
read actuator names, indices, joint coordinates, and torque limits directly from
the loaded MuJoCo model rather than assume joint indices match action indices. All
controls are normalized to [-1, 1]; the XML motor gear scales them to physical torque.
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
