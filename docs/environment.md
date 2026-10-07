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

**Start.** A reset returns the body to the pose stored in the model, varying only
the legs: each leg angle moves by a random amount between −2° and +2°, and each
leg joint starts with a random speed (mean 0, spread 0.05 rad/s). Each world has
its own random sequence, and a fixed seed repeats it. Targets use a separate
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

The reward is small and local. Reaching the target is the only positive reward;
everything else is a cost, and nothing rewards a particular gait. Every segment
`i` receives:

```text
r_i = A + E_i − c_body × B_i − c_leg × L_i
```

- **A**, arrival: `c_arrival` on the step the target is reached, the same for
  every segment; otherwise zero.
- **B_i**, body contact: 1 if the segment's body touches the ground.
- **L_i**, leg contact: 1 if one of its legs touches another leg. Both segments
  involved pay; feet on the ground cost nothing.
- **E_i**, efficiency: a cost that shrinks with progress.

For the **head**, progress is toward the target. With `d_before` and `d_after`
the flat distance from its tip to the target before and after the step:

```text
E_0 = −c_efficiency × (ε + d_after) / (ε + d_before)
```

For **every other segment**, progress is toward the spot where the segment ahead
of it was at the start of the step, with `g_before` and `g_after` its distances to
that spot:

```text
E_i = −c_efficiency × (ε + g_after) / (ε + g_before)
```

Standing still costs exactly `c_efficiency`; moving closer costs less and moving
away costs more. The cost is always negative, so it pushes the head to reach the
target quickly. Because each follower aims at a place its leader has already
been, a straight or turning path passes down the body from segment to segment.
This prescribes no speed, leg motion, or wave. Since the body is connected, part
of a follower's progress can come from being pulled along, so evaluation must
measure what each segment actually contributes.

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
| **`[environment.target]`** | | |
| `distance_range_m` | 0.010 to 0.020 | Distance of a new target from the head's tip |
| `bearing_range_deg` | −15 to 15 | Direction of a new target from straight ahead |
| `arrival_radius_m` | 0.001 | Distance that counts as arrival |
| **`[environment.rewards]`** | | |
| `arrival_reward` (`c_arrival`) | 1.0 | Shared reward for reaching the target |
| `efficiency_cost` (`c_efficiency`) | 0.003 | Efficiency cost when making no progress |
| `body_contact_cost` (`c_body`) | 0.010 | Cost per step with the body on the ground |
| `leg_contact_cost` (`c_leg`) | 0.005 | Cost per step with legs touching |
| `distance_ratio_epsilon_m` (`ε`) | 0.000001 | Keeps the distance ratios well defined |

The episode length is a hyperparameter like the others: 8,192 steps is the
default, and 4,096 is the main alternative considered. The GPU memory values
leave headroom over the 40 contacts and 256 constraint rows measured at rest;
they are confirmed in motion on the GPU. Reward values are starting points for
the first experiments, not measured biology. Reward experiments change these
values or the reward function, and nothing else.

The reward function is a sum of named **terms**, each multiplied by its weight
from `[environment.rewards]`; a weight of zero switches a term off. To keep
earlier runs' configurations meaning the same reward, a new term is added with a
default weight of zero, and a changed formula is added as a new term rather than
by editing an existing one.

## Later experiments

Left for after the first version works, one change at a time:

- Targets farther away, at wider angles, or behind the head.
- A new target after arrival, without a reset.
- More variation in the starting pose.
- Agents controlling and observing the spine.
- A larger observation radius, which is only a configuration change.
- A body with more segments: a new model version. The code adapts, but trained
  agents do not carry over, because each segment has its own network.
- A reward term for a wave-like gait, only if one does not emerge on its own.
