# Environment

The environment is the problem the agents must solve: the centipede's body, a
target for its head, what each segment observes, what reward it receives, and
when an episode ends. It sits on layer 2 of the
[architecture](architecture.md), next to the agents, and is the only component
that uses the physics simulation. It knows nothing about learning and never
stores data for training.

## Interface

The environment runs several independent copies of the centipede at once, called
**worlds**. Below, `W` is the number of worlds and `N` the number of segments.
`N` is never written in code or configuration: it is read from the model (eight
in v1). Everything passed in and out is a PyTorch tensor.

| Operation | Takes | Returns |
| --- | --- | --- |
| `reset(seed)` | An optional seed | Observations `(W, N, observation_size)` |
| `step(joint_action)` | Actions `(W, N, 6)` | Observations, rewards `(W, N)`, terminated `(W,)`, truncated `(W,)`, final observations |

The environment also reports `segment_count` and `observation_size`, which the
experiment uses to create the agents. When an episode ends in a world, `step`
resets that world by itself: the observations it returns are already the new
episode's, and the final observations hold, for that world, the observation the
episode ended with. A seed given to `reset` makes the starting poses and targets
repeatable.

## Files

The environment lives in `src/centipede/environment/`.

| File | Covers |
| --- | --- |
| `environment.py` | Front file: `reset()` and `step()`, in two sections: episode state and coordination |
| `observation_builder.py` | What each segment observes |
| `reward_function.py` | Each segment's reward |
| `settings.py` | Environment settings |
| `diagnostics.py` | The environment's [diagnostics](diagnostics.md) |
| `simulation/simulation.py` | Front file of the physics simulation: loads the model, chooses the backend, offers `reset()`, `step()`, and the physical state |
| `simulation/model_mapping.py` | Finding each segment's parts in the model by name, and checking them |
| `simulation/physical_state.py` | The physical state handed to the environment |
| `simulation/diagnostics.py` | The simulation's [diagnostics](diagnostics.md#simulation-facts): every world's positions and the physics health values |
| `simulation/constants.py` | Fixed values both backends share: physics steps per action, reset noise |
| `simulation/cpu_backend.py` | Physics on the CPU with MuJoCo |
| `simulation/gpu_backend.py` | Physics on the GPU with MuJoCo Warp |
| `simulation/settings.py` | Physics simulation settings |

## How a step works

The front file has two jobs. **Episode state:** for every world it remembers the
target, the step count, and the previous positions that rewards need, and it
decides which worlds must be reset. **Coordination:** on every `step`, it runs the
other parts in order:

1. Keep the current body positions as the "previous" positions.
2. The physics simulation applies the actions and advances 20 ms.
3. Each world is checked for arrival or the time limit.
4. The reward function computes each segment's reward; the arrival reward
   needs the result of step 3.
5. The observation builder builds the next observations, and the diagnostics
   are updated.
6. Worlds whose episodes ended are reset, get a new target, and get fresh
   observations; their final observations are returned as well.

## Physics simulation

The physics simulation moves the body: it loads the model, applies actions,
advances the physics, resets worlds, and reports the physical state. It knows
nothing about targets, rewards, or episodes.

### Loading the model

The simulation accepts any model that follows the **segment contract**, checked
once when it starts:

- segments are numbered from 0 (the head) to `N − 1` (the rear), with no gaps;
- every segment has a body, two legs with three motors each, and two feet;
- there are `N − 1` spine connections, each with one yaw motor;
- the head carries a `head_tip` site.

It never relies on the order of elements in the XML file. It finds every motor by
name (`segment_{id:02d}_{side}_{role}_motor`) and stores them in a table of shape
`(N, 6)`, one row per segment in [action order](#actions-and-timing). It then
checks that each motor's name matches the owning segment stored in the XML, that
it drives the right joint and accepts commands from −1 to 1, and that every motor
is found exactly once with none left over (`6N + N − 1`, which is 55 for v1). It
also checks that every collision shape records its owning segment and its
category (floor, body, leg, left foot, right foot, or membrane; see
[model.md](model.md#contacts-and-friction)).

If any check fails, the program stops with a clear error; after that, the
mapping is trusted. A model with a different number of segments needs no code
change, only a configuration pointing to it.

### Actions and timing

Each segment controls only its own two legs. Its action is six numbers between
−1 and 1, always in this order:

| Index | Motor |
| ---: | --- |
| 0 | Left shoulder sweep |
| 1 | Left shoulder lift |
| 2 | Left knee |
| 3 | Right shoulder sweep |
| 4 | Right shoulder lift |
| 5 | Right knee |

All segments act at the same time and the body moves once for all of them.
Actions are never clipped, because the executed action must be exactly the one
the agent learns from. The spine yaw motors are not controlled in this first
version: they always receive zero, leaving the spine to bend passively against
its springs.

An action is held for **20 ms**, or 50 decisions per second: as many physics
steps as the model's timestep fits into it, 134 steps of 0.149 ms for model v3.
The simulation computes that number from the model and rejects a model whose
timestep does not divide 20 ms evenly. On the GPU, an action's physics steps
are replayed from a CUDA graph recorded on the first step, so the CPU does not
wait for the GPU between them. Time is measured by counting steps, never
by the simulator's clock, which drifts in 32-bit arithmetic.

### When the physics fails

After every step, the simulation checks every world for positions or velocities
that are not finite numbers and, on the GPU, for contacts or constraints that did
not fit in the reserved memory. Any of these stops the run with a clear error;
they are never hidden or treated as the end of an episode. A solver that stops
before fully converging is not an error, and a fall is ordinary behaviour.

### Backends

The configuration chooses the backend. Both offer the same operations and the
same physical state, so nothing outside the simulation can tell which is running.

- **CPU:** MuJoCo, with one MuJoCo state per world. Its NumPy arrays become
  tensors without copying.
- **GPU:** MuJoCo Warp, which simulates all worlds at once. Its arrays also
  become tensors without copying, so the data stays in graphics memory.

Known differences of the GPU backend:

- It computes in 32-bit floats instead of 64-bit, so it is compared with the CPU
  backend within tolerances, not bit for bit. Leg impacts amplify these small
  differences, so under random actions the two backends drift apart after a
  single step; they are compared while the body settles with motors off, where
  positions agree within about 0.03 mm.
- Memory for contacts and constraints is reserved in advance. MuJoCo Warp's
  defaults (48 contacts, 64 constraint rows per world) are too small: the model
  at rest already reaches 40 contacts and 256 constraint rows.
- Its default Newton solver needs a GPU of the Volta generation or newer, such
  as Kaggle's T4; the local MX330 and Kaggle's P100 cannot compile it. With
  models v2 and v3, the MX330 can run the GPU backend using the conjugate-gradient
  solver, slowly and less converged, which is enough for local functional
  tests.

## Physical state and observations

### The physical state

After every `step` and `reset`, the simulation fills the physical state of every
world. It is a set of tensors overwritten in place; a reader that needs a value
later copies it. Numbers are 32-bit floats and contact flags are booleans.

Most fields also form a segment's **observation block**: 27 values describing one
segment, in the order given by the last column.

| Field | Shape | Meaning | In the block |
| --- | --- | --- | --- |
| `body_height` | `(W, N)` | Height of the segment's centre, m | 0 |
| `body_quaternion` | `(W, N, 4)` | Orientation as `(w, x, y, z)` | 1–4 |
| `leg_joint_position` | `(W, N, 6)` | Leg angles in action order, rad | 5–10 |
| `body_linear_velocity` | `(W, N, 3)` | Velocity of the centre in the segment's own frame, m/s | 11–13 |
| `body_angular_velocity` | `(W, N, 3)` | Turning speed in the segment's own frame, rad/s | 14–16 |
| `leg_joint_velocity` | `(W, N, 6)` | Leg angular velocities in action order, rad/s | 17–22 |
| `left_foot_ground_contact` | `(W, N)` | Left foot touches the ground | 23 |
| `right_foot_ground_contact` | `(W, N)` | Right foot touches the ground | 24 |
| `body_ground_contact` | `(W, N)` | Body touches the ground | 25 |
| `leg_leg_contact` | `(W, N)` | One of its legs touches another leg | 26 |
| `body_planar_position` | `(W, N, 2)` | World `x`, `y` of the centre | Not observed; rewards only |
| `head_tip_position` | `(W, 3)` | World position of the head's tip | Not observed; rewards and target only |

In the block, contact flags become 0 or 1. A segment never observes its position
in the world, and the spine is not observed in this first version.

Contact flags describe the state at the end of the 20 ms step, not contacts that
happened along the way:

| Contact between | Sets |
| --- | --- |
| Ground and a segment's body | `body_ground_contact` of that segment |
| Ground and a segment's left or right foot | That side's foot flag |
| Ground and a leg part that is not a foot | Nothing |
| Two leg or foot parts, even of the same segment | `leg_leg_contact` of both owners |
| A leg and a body | Nothing |

### What each segment sees

Each segment sees its own block and its neighbours up to `observation_radius`
segments away (`k`, first value 1). Every observation has the same layout:

1. its own block;
2. the blocks of the `k` segments ahead, nearest first;
3. the blocks of the `k` segments behind, nearest first;
4. two target values: the target's position relative to the head's tip, turned
   into the head's own direction, in metres. **Forward** is positive in front of
   the head; **sideways** is positive to its left.

That is `(2k + 1) × 27 + 2` values, **83** at radius 1 and 29 at radius 0, for
every segment and any number of segments. A neighbour that does not exist is filled with zeros, and
only the head receives real target values:

| Segment | Ahead | Behind | Target values |
| --- | --- | --- | --- |
| Head (0) | zeros | segment 1 | real |
| Middle (`i`) | segment `i − 1` | segment `i + 1` | zeros |
| Rear (`N − 1`) | segment `N − 2` | zeros | zeros |

The observation builder makes all observations with one indexing operation,
using a table of shape `(N, 2k + 1)` built once, with missing neighbours pointing
to an all-zero block. Changing the radius changes only this table.

The zeros are harmless. Each segment has its own network, and an input that is
always zero for that segment never changes it, so no "neighbour present" flag is
needed. Such a flag would only matter if networks were shared between segments.
Normalising observations is the agents' job, not the environment's.

## Episodes

**Start.** A reset returns the body to the pose stored in the model, varying
the legs: each leg angle moves by a random amount between −2° and +2°, and each
leg joint starts with a random speed (mean 0, spread 0.05 rad/s). With
`start_heading_range_deg` above 0, the whole body is then turned about the
vertical by a random angle within that range, by turning the free joint of the
head's body, the root of the body tree. Each world has its own random sequence,
and a fixed seed repeats it; the heading is drawn after the legs, so the legs
start the same with or without it.

The heading matters because each segment observes its orientation in the
world, which includes the direction it faces. Targets are placed relative to
the head, so with a fixed starting heading the direction a segment faces
carries no information about the task, yet the networks only ever see the few
directions near the start: a centipede that turns far from it gives its
networks inputs they were never trained on. Random headings make the facing
direction uninformative everywhere, so the networks learn to ignore it. The
starting position on the floor is not varied: no segment observes it, and the
floor is the same everywhere. Targets use a separate
random sequence, so changing how targets are chosen never changes starting
poses. A reset gives no reward, and its pose counts as the "previous" state for
the first step.

**Target.** The head gets a point on the ground in front of it, chosen relative
to its tip and direction: distance uniformly between 10 and 20 mm, direction
uniformly within 15° left or right of straight ahead. The target is a point
remembered by the environment, not an object in the simulation.

The head's **forward** direction is its body's x axis, which points from its
centre to its tip, laid flat on the ground; **left** is 90° anticlockwise from
it, the side of the left legs. Targets are placed, and the two target values
observed, in these directions.

**End.** An episode ends in one of two ways, always for all segments together:

- **Arrival** (terminated): the head's tip comes within 1 mm of the target,
  measured flat on the ground.
- **Time limit** (truncated): the episode reaches `max_episode_steps`, with no
  extra penalty.

If both happen on the same step, arrival wins. Nothing else ends an episode: the
centipede may fall or touch the ground with its body and recover.

## Rewards

The reward was rebuilt on 2026-10-08 from a small set of rules, agreed with
the user, so that every weight follows from them instead of being tuned by
hand. This section gives the terms, the rules, what they produce, and the
reward used before, which earlier runs keep.

### The terms

Every segment `i` receives, on every step:

```text
r_i = A × arrived + w_progress × P_i − w_step − w_body × B_i − w_leg × L_i
```

| Term | Its value on a step | Who receives it |
| --- | --- | --- |
| **Arrival** | 1 on the step the head comes within the arrival radius of the target, otherwise 0 | Every segment, the same |
| **Progress** `P_i` | `log2(distance before ÷ distance after)`: how many times the segment halved its distance to its goal; negative when it moved away, 0 when it did not move | The head toward the target; every other segment toward the spot where the segment ahead of it was at the start of the step, counted at `follower_progress_ratio` of the head's weight |
| **Step cost** | 1 on every step | Every segment |
| **Body contact** `B_i` | 1 when the segment's body touches the ground | That segment |
| **Leg contact** `L_i` | 1 when one of its legs touches another leg; both segments involved pay | That segment |

Feet on the ground cost nothing, and nothing rewards a particular gait. The
design is a **leader and followers**: only the head sees the target, and its
progress is what pulls the body toward it; each follower is rewarded, weakly,
for following the path of the segment ahead. Because a follower aims at a place
its leader has already been, a straight or turning path passes down the body
from segment to segment. Since the body is connected, part of a follower's
progress can come from being pulled along, so evaluation must measure what each
segment actually contributes.

**Why progress counts halvings.** A halving has no unit, so the same rules hold
for any target distance, and it is worth the same at 16 mm as at 2 mm. Over an
episode the head's progress adds up to `log2(start distance ÷ end distance)`,
whatever path it took: wiggling back and forth earns nothing, and walking past
the target gives back what was earned on the way. Distances closer than the
arrival radius count as the radius, so an episode that arrives earns exactly
`log2(start distance ÷ arrival radius)`, and landing nearer the target's centre
earns nothing more. Moving away is penalised only by the logarithm of how far
the head goes: ending twice as far as it started costs one halving.

Progress is part of the goal, not only a guide to it: it says that ending closer
is better than not trying (rule R2 below). Pure "potential-based shaping" (Ng,
Harada & Russell, 1999) leaves the best behaviour unchanged by construction, so
it could not make an attempt better than standing still.

### The rules

Every weight is a proportion of the arrival reward `A` (1 by default; learning
only sees proportions, since each update rescales its advantages). A cost is
given by its **share** of a whole episode of `T` steps (`max_episode_steps`): a
cost with share `s` costs `s × A / T` on each step, so the rules hold for any
episode length. The shares are σ for the step cost, β for body contact, and λ
for leg contact.

- **R0. The discount fits the episode length.** The agents' discount is
  `γ = 2^(−1/T)`, so an arrival on an episode's last step is worth half of `A`
  seen from its first step. The configuration sets it when a file sets no
  `discount` (0.9973 for 256-step episodes).
- **R1. The worst arrival balances to zero.** An episode that pays every cost on
  every step (body on the ground, legs touching) and arrives on its last step
  adds up to zero, seen from its start with that discount:
  `γ^(T−1) = (σ + β + λ) / T × (1 − γ^T) / (1 − γ)`. This gives the **cost
  budget** `σ + β + λ = T × (2^(1/T) − 1)`, about ln 2 = 0.69 for any long
  episode. R1 counts only the costs: every arrival also earns the progress of
  its approach, which depends only on where it started, and asking the costs to
  cancel that too would make them larger than the arrival itself.
- **R2. Standing still is a little worse than trying badly.** A centipede that
  halves its distance with its legs touching all episode must end better than
  one that stands still, upright: one halving must be worth more than λ. A
  halving is worth one whole episode of step cost (`head_progress_ratio = 1`,
  the progress weight per halving equal to σ × A), so the attempt ends at −λ and
  standing still at −σ = −2λ: worse by the mildest cost.
- **R3. The order of the costs.** Body contact weighs most, then the step cost,
  then leg contact: the budget is split **3 : 2 : 1** (β : σ : λ).
- **R4. The followers follow but do not push.** A follower's progress weight is
  `follower_progress_ratio` = 1/10 of the head's. Its goal is only one segment
  (3.8 mm) away, so a step of walking halves a good share of that distance: at
  the 0.17 mm per step measured in the first runs, a whole episode of walking
  makes about 16 halvings, which at 1/10 earns 0.38 `A`, below the cost budget.
  Following then never outweighs posture and time. Above about 1/5 it would, and
  above 1/2 it would outweigh the arrival itself, making walking forward the
  followers' goal: they would push the head straight past its target.

Two earlier candidates turned out to be consequences rather than rules. A full
approach from 20 mm earns `log2(20) × σ ≈ 1.0 A`, about the arrival; every
arrival still sits a whole `A` above a near miss, and progress cannot be farmed.
Wandering away is penalised only gently, by the logarithm.

### What the rules give

With `A = 1` and 256-step episodes:

| Term | Weight | From the rules |
| --- | --- | --- |
| Arrival | 1, once | The unit |
| Step cost | 0.00090 per step | σ = 2/6 of the budget (0.231), over 256 steps |
| Body contact | 0.00136 per step | β = 3/6 of the budget (0.347) |
| Leg contact | 0.00045 per step | λ = 1/6 of the budget (0.116) |
| Progress, head | 0.231 per halving | σ × `head_progress_ratio` |
| Progress, followers | 0.0231 per halving | The head's × `follower_progress_ratio` |
| Discount | 0.9973 | 2^(−1/256) |

Doubling the episode length halves the per-step costs and sets the discount to
0.9987; nothing else changes. Every training run prints these weights under its
settings. Totals per segment over one episode, without discount, in units of
`A`, for targets 10 to 20 mm away:

| Outcome | Total |
| --- | ---: |
| Clean arrival from 15 mm in 90 steps | +1.82 |
| Worst arrival from 10 mm: on the last step, body on the ground and legs touching throughout | +1.07 |
| Clean near miss, ending 2 mm from the target | +0.44 |
| Legs touching throughout, halving its distance | −0.12 |
| Standing still, upright | −0.23 |
| Body on the ground throughout, halving its distance | −0.35 |
| Clean walk straight past the target, ending 40 mm away | −0.56 |
| Worst failure: body down, legs touching, ending 58 mm away | −1.14 |

Seen from the start through the discount, the worst arrival comes to about +0.5
(zero for its costs, plus its progress) and standing still to −0.17. Per step at
walking speed, the head's progress is about three times the body cost at 15 mm
and grows near the target, so a much faster crawl could outweigh an upright
walk; the reports' body values show whether it does.

**Switching a cost off.** A cost of 0 parts is off. The budget is split into
`cost_budget_parts` equal parts, by default the costs' parts together; setting
it higher than their sum leaves part of the budget unused, so that switching
costs off does not change the others. For example, a run that ignores posture
keeps the step cost and progress of the full reward with
`body_contact_cost_parts = 0`, `leg_contact_cost_parts = 0`, and
`cost_budget_parts = 6`.

### The reward used before 2026-10-08

Earlier runs used per-step weights and one **efficiency** term in place of the
step cost and progress:

```text
r_i = A × arrived + E_i − c_body × B_i − c_leg × L_i
E_i = −c_efficiency × (ε + distance after) / (ε + distance before)
```

The efficiency term held the step cost (`−c_efficiency` when standing still)
and the progress (`c_efficiency × gain ÷ distance`) under one weight. Its
progress part was about 1% of it at the targets' distance, and about a
thousandth of the arrival over a whole approach: in the runs of 2026-10-08 the
head's progress part averaged zero, and no run learned to steer. A
configuration that sets any of `efficiency_cost`, `body_contact_cost`, or
`leg_contact_cost` uses this reward, so runs saved before keep it when they are
continued or evaluated; a file that also sets the proportions is refused.

### Changing the reward

The reward function is a sum of named **terms**, each multiplied by the weight
`RewardSettings.weights` works out. A new term is placed in the rules before it
is added: a cost takes its parts of the budget, so that the worst arrival still
balances to zero, and anything positive is weighed against the arrival and the
costs as progress was. A changed formula is added as a new term rather than by
editing an existing one, so that earlier runs' settings keep meaning the same
reward.

## Diagnostics

The environment measures what happens on every step and how each episode ends:
reward parts, contacts, each segment's progress, posture, and the head's
distance and direction to the target. It also passes on the physics
simulation's measurements. The full list is in [diagnostics.md](diagnostics.md).
Policies never see these values.

## Settings

These are the keys of the environment sections of the configuration file (see
[configuration.md](configuration.md)). Each first value is also the default.

| Setting | First value | Meaning |
| --- | ---: | --- |
| **`[environment]`** | | |
| `max_episode_steps` | 8,192 | Time limit, in 20 ms steps (about 164 s) |
| `observation_radius` | 1 | Neighbours seen on each side; 0 means only itself; must be less than `N` |
| **`[environment.simulation]`** | | |
| `model_path` | Required | Model file to load |
| `backend` | Required | `cpu` or `gpu` |
| `world_count` | 1 | Number of worlds simulated at once |
| `gpu_solver` | `newton` | Constraint solver, GPU only: `newton`, or `cg` for GPUs older than Volta |
| `contacts_per_world` | 128 | Reserved contact memory, GPU only |
| `constraints_per_world` | 512 | Reserved constraint memory, GPU only |
| `start_heading_range_deg` | 0 | A reset turns the whole body about the vertical by a random angle within ± this many degrees; 180 allows any heading |
| **`[environment.target]`** | | |
| `distance_range_m` | 0.010 to 0.020 | Distance of a new target from the head's tip |
| `bearing_range_deg` | −15 to 15 | Direction of a new target from straight ahead |
| `arrival_radius_m` | 0.001 | Distance that counts as arrival |
| **`[environment.rewards]`** | | |
| `arrival_reward` (`A`) | 1.0 | Shared reward for reaching the target: the unit of every other weight |
| `step_cost_parts` | 2 | The step cost's parts of the cost budget |
| `body_contact_cost_parts` | 3 | Body contact's parts of the cost budget |
| `leg_contact_cost_parts` | 1 | Leg contact's parts of the cost budget |
| `cost_budget_parts` | The three together | How many equal parts the budget is split into; more than the costs' parts leaves some unused |
| `head_progress_ratio` | 1 | What one halving of the head's distance is worth, in whole episodes of step cost |
| `follower_progress_ratio` | 0.1 | A follower's progress weight, as a share of the head's |
| `distance_ratio_epsilon_m` (`ε`) | 0.000001 | Keeps the efficiency ratio of the earlier reward defined |
| `efficiency_cost`, `body_contact_cost`, `leg_contact_cost` | Not set | Per-step weights of the reward used before 2026-10-08 (0.003, 0.010, and 0.005 when any is set); setting one selects that reward |

The episode length is a hyperparameter like the others: 8,192 steps is the
default, and 4,096 is the main alternative considered. The GPU memory values
leave headroom over the 40 contacts and 256 constraint rows measured at rest;
they are confirmed in motion on the GPU. The reward's proportions come from
the rules in [Rewards](#rewards), not from measured biology. Reward experiments
change these proportions or the reward function, and nothing else.

## Later experiments

Left for after the first version works, one change at a time:

- Targets farther away, at wider angles, or behind the head.
- A new target after arrival, without a reset.
- More variation in the starting pose than the heading.
- Agents controlling and observing the spine.
- A larger observation radius, which is only a configuration change.
- A body with more segments: a new model version. The code adapts, but trained
  agents do not carry over, because each segment has its own network.
- A reward term for a wave-like gait, only if one does not emerge on its own.
- A shared progress term (proposed 2026-10-08, kept for later by the user):
  every segment receives `w × (d_before − d_after)`, the head's progress toward
  the target, like the shared arrival reward. Its total over an episode is
  `w × (start distance − end distance)`, so wiggling earns nothing and walking
  past the target gives back what was earned (potential-based shaping, Ng,
  Harada & Russell, 1999). It changes the design from a leader that pulls and
  followers that follow to every segment actively helping the head, so it is
  tested as its own path, not combined with the leader–follower efficiency.
