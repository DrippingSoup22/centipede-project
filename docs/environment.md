# Environment, rewards, and evaluation

This document owns target generation, physical snapshots, reset and episode
behavior, rewards, task information, and future difficulty changes. The
first-version contract is stated directly; later possibilities are kept separate.
Target sensing and policy observations are defined in [control.md](control.md),
while physical geometry and contact groups are defined in [model.md](model.md).

## Framework boundary

The public task is a PettingZoo `ParallelEnv` containing one small internal
Gymnasium `MujocoEnv`. The internal simulation owns MuJoCo loading, state, physical
reset, frame skipping, stepping, rendering, cleanup, and translation of raw model
data into a trusted snapshot. The public environment owns the target, partial
observations, rewards, episode clock, arrival, termination, truncation, and task
information.

`reset(seed=...)` is the sole episode-reset entry point. It returns observation
and information dictionaries for all eight agents. `step(actions)` applies all
eight accepted actions simultaneously and returns PettingZoo's observation,
reward, termination, truncation, and information dictionaries. All agents remain
active together because they control one inseparable physical body.

The public environment validates an externally supplied action dictionary once,
before any state change. The simulation separately validates the XML schema once
at construction. Observation and reward calculations receive trusted data and do
not repeat either check. Tests own deliberately invalid inputs; production code
does not import test fixtures.

One public transition holds the joint action for 200 MuJoCo integration steps.
The frozen `0.0001 s` physics timestep therefore gives a 20 ms environment step
and 50 Hz agent decision frequency. Episode duration and reported time use this
simulated interval, not wall-clock runtime.

## Physical snapshot

After physical reset and after every 20 ms transition, the simulation takes one
snapshot of the complete centipede. This is a temporary, immutable-by-convention
copy of the values needed by the environment. It is not an image, checkpoint,
rollout buffer, or separate observation for every agent.

For each segment, the snapshot contains:

- body height and world-relative orientation quaternion;
- six leg-joint angles and six leg-joint angular velocities;
- body linear and angular velocity in that segment's local frame;
- planar world position of the body center for reward displacement calculations;
- left-foot, right-foot, body-ground, and leg-leg contact booleans.

The complete snapshot also contains the world position of the `head_tip` site.
The target belongs to the public environment and is not physical snapshot data.
The observation builder selects only the fields and neighboring blocks each agent
may see; reward-only planar positions are never added to policy observations.

Snapshot arrays are copied from MuJoCo and retained as `float64` internally. The
environment keeps only the previous and current snapshot. Every environment
replica owns its own simulation, snapshots, target, and random state.

Contact flags describe the final MuJoCo state at the snapshot, rather than any
contact that occurred during one of the 200 internal integration steps. A contact
that remains present therefore keeps its flag active on every environment step.
The simulation alone interprets MuJoCo geometry contacts and assigns their segment
owners; reward code does not inspect the raw contact array.

After each transition, the simulation checks the complete physical state and
snapshot for non-finite values and checks MuJoCo's numerical-instability warnings.
Numerical invalidity raises a clear exception and stops the run. It is not clipped,
replaced, silently reset, or reported as ordinary termination or truncation. A
finite fall or contact remains valid task behavior.

## Seeded physical reset

The nominal head, body, root, spine, and overall body velocities are unchanged.
Only the 48 controlled leg coordinates receive small independent variation:

- each leg-joint angle receives `Uniform(-2 degrees, +2 degrees)` around its
  nominal XML pose;
- each leg-joint velocity receives `Normal(0, 0.05 rad/s)`.

Spine angles and velocities remain nominal. A fixed seed reproduces the same
sequence of resets, while repeated `reset()` calls without reseeding advance that
sequence and produce different episodes.

The public reset derives separate deterministic child random streams for physical
variation and target sampling. Changing target sampling must not alter the
physical initial pose produced by the same public seed.

No reward is produced by `reset()`. The reset snapshot becomes the previous
snapshot used to evaluate the first transition.

## Target and episode

After the physical reset, sample one non-colliding point on the ground plane
relative to the current `head_tip` position and head yaw:

- distance is uniform from `0.010 m` through `0.020 m`;
- signed bearing is uniform from `-15 degrees` through `+15 degrees`;
- distance itself is sampled uniformly, not area within the resulting wedge.

The target is task state, not a MuJoCo body. Only the head observes it, using the
forward and lateral displacement defined in [control.md](control.md).

Arrival is checked after every transition using planar Euclidean distance from
the `head_tip` site to the target. A distance at or below `0.001 m` is arrival;
vertical distance and head orientation do not affect it. Arrival awards the
shared reward once and terminates all eight agents.

The first experiment uses an episode limit of 1,000 environment steps, or 20
seconds of simulated time. The experiment configuration may choose another
positive limit without changing the model or reward. Reaching the target on the
final allowed step takes precedence:
all agents report termination and not truncation. Otherwise the time limit
truncates all agents without an additional timeout penalty. The final observation
is returned in either case.

Body or leg contact never ends the first-version episode and never removes an
individual agent. The centipede may recover from a finite fall. The only ordinary
termination is arrival, and the only ordinary truncation is the time limit.

A PPO rollout window is separate from an episode. Reaching its fixed sample count
causes an update with critic bootstrapping and then continues from the same
environment state. It does not reset, terminate, or truncate the environment.

## First reward configuration

The reward is intentionally small and local. Arrival is its only positive term.
Every other term is a non-positive cost. There is no separate time cost,
action-effort cost, spine term, prescribed leg phase, or wave reward.

For the head:

```text
r_0 = A + E_0 - c_body * B_0 - c_leg * L_0
```

For every follower `i` from `1` through `7`:

```text
r_i = A + E_i - c_body * B_i - c_leg * L_i
```

The terms are:

- `A` is `c_arrival` on the one arrival transition and zero otherwise. The same
  value is copied to all eight independent learners.
- `B_i` is one when segment `i`'s body touches the ground in the final snapshot
  and zero otherwise.
- `L_i` is one when a leg owned by segment `i` participates in a leg-leg contact
  in the final snapshot and zero otherwise. Both owners are charged when two
  segments' legs contact. Ordinary foot-ground contact has no cost.
- `E_i` is the agent's non-positive efficiency term described below.

The first executable defaults are:

| Parameter | Initial value | Meaning |
| --- | ---: | --- |
| `c_arrival` | `1.0` | Shared arrival reward |
| `c_efficiency` | `0.003` | Cost scale; remaining stationary produces this cost |
| `c_body` | `0.010` | Local body-ground cost per active step |
| `c_leg` | `0.005` | Local leg-leg cost per active step |
| `epsilon_ratio` | `0.000001 m` | Numerical guard in distance ratios |

These values define a reproducible starting configuration, not biologically
measured constants. Stage 2 must measure zero-action noise and deterministic
motion before training; later experiments may tune the numerical guard and
reward coefficients without changing the reward's structure.

### Head efficiency

Let `d_t` be planar head-tip distance before a transition and `d_(t+1)` the
distance after it. The head efficiency term is:

```text
E_0 = -c_efficiency * (epsilon_ratio + d_(t+1))
                      / (epsilon_ratio + d_t)
```

Moving closer makes the ratio smaller than one and reduces the cost. Remaining
at the same distance gives exactly `-c_efficiency`, while moving farther makes
the cost larger in magnitude. The term is always negative, so arrival remains
the only positive reward component. It combines time pressure and target
progress: faster progress incurs less cost per step and reaching the target in
fewer steps avoids future costs.

### Local path-following efficiency

For follower `i`, let the fixed local destination for transition `t` be the
planar body-center position occupied by its immediate predecessor at the start
of that transition:

```text
q_i(t) = position_(i-1)(t)
```

Measure the follower's distance to that fixed point before and after the
transition:

```text
g_before = norm(position_i(t)   - q_i(t))
g_after  = norm(position_i(t+1) - q_i(t))
```

The follower efficiency term is:

```text
E_i = -c_efficiency * (epsilon_ratio + g_after)
                      / (epsilon_ratio + g_before)
```

Approaching the predecessor's start-of-transition position reduces the cost;
remaining equally distant gives `-c_efficiency`; moving away increases the cost.
The destination is frozen while the transition is evaluated, so the follower is
directed toward a point its predecessor has already occupied rather than toward
the predecessor's new position. Applying the same rule successively from segment
1 through segment 7 lets a straight or turning path propagate down the chain with
one-step local delays. The first transition is evaluated normally because the
reset snapshot already supplies every predecessor position.

This is a deliberately limited approximation of trail following. It does not
store a long trajectory or prescribe an exact curve, speed, leg sequence, body
phase, or wave. The physical joints constrain segment spacing, while the reward
only asks each follower to reduce distance to one local historical point.
Mechanical coupling may still satisfy part of this condition without active leg
work, so evaluation must measure each segment's contribution rather than infer
cooperation from reward alone. On an arrival transition, ordinary efficiency and
contact terms are calculated before adding the shared arrival reward.

The follower term does introduce one intentional preference: body centers should
advance through locally preceding positions, like linked units following a
leader. That preference is the cooperative behavior required by this task, but
it does not determine the joint motions used to achieve it. In particular, the
reward contains no desired gait waveform. Whether a wave emerges remains an
evaluation result rather than a built-in solution.

## Information returned for diagnostics

`info` is not an observation and is never passed to a policy or PPO update. It
exposes already-computed task facts so evaluation does not decode observation
indices or reimplement environment rules.

Each agent's ordinary information dictionary contains
`left_foot_ground_contact`, `right_foot_ground_contact`,
`body_ground_contact`, and `leg_leg_contact`, plus the four scalar components
`reward_arrival`, `reward_efficiency`, `reward_body_contact`, and
`reward_leg_contact`. At reset, only the head receives `target_distance_m` in
`info`; other reset information dictionaries are empty. Each step, the head
additionally reports its current `target_distance_m`, planar
`head_step_distance_m`, and `target_reached`. Summing the step distances gives
the head's traveled path length; subtracting final from initial target distance
gives net progress toward the target. At episode end, the head also reports `episode_end` as
`arrival` or `time_limit`, together with `episode_steps` and `episode_time_s`.
Raw snapshots and unrestricted MuJoCo state are not published.

## Later experiments and evaluation

The first reward aims at basic navigation and local cooperation. It does not
prove walking quality or biological resemblance. Evaluation must separately
measure target success, time to arrival, net head progress, follower-condition
frequency and cost, contacts, motion produced by each segment, and visually
inspected locomotion.

After the baseline works, controlled experiments may enlarge target distance and
bearing, place targets behind the body, assign a new target without resetting,
increase observation radius, expose or control spine state, change target sensing,
or introduce an explicit gait objective. Change one factor at a time and keep the
baseline as the comparison.

Wave-like motion remains the second project goal, but no wave term belongs in the
first reward. Its eventual metrics, phase relationships, and acceptance criteria
must be defined from observed locomotion rather than treating arrival or reward
growth as evidence of a natural gait.
