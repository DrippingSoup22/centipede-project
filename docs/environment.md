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
| `step(joint_action)` | Actions `(W, N, action_size)`: six per segment (twelve with leg clocks), plus a spine command and a clock tempo where it has them | Observations, rewards `(W, N)`, terminated `(W,)`, truncated `(W,)`, final observations |
| `set_target_ranges(distance_range_m, bearing_range_deg)` | Two `(low, high)` pairs | Nothing; new targets are drawn from these ranges from now on |

The environment also reports `segment_count`, `observation_size`,
`segment_action_sizes` (how many actions each segment takes), and
`action_size` (the joint action's width), which the experiment uses to create
the agents; `tempo_columns` and `motor_columns`, which of each segment's actions
set clock tempos and which sets each of its motors, for the agents' learning
diagnostics; and `clocks`, the [clocks](#clocks) of the segments or the legs
when they have them. When an episode ends in a world, `step`
starts the next one by itself, resetting the world or, after an arrival, perhaps
only giving it a new target (see [Episodes](#episodes)): the observations it
returns are already the new episode's, and the final observations hold, for that world, the observation the
episode ended with. A seed given to `reset` makes the starting poses and targets
repeatable.

## Files

The environment lives in `src/centipede/environment/`.

| File | Covers |
| --- | --- |
| `environment.py` | Front file: `reset()` and `step()`, in two sections: episode state and coordination |
| `observation_builder.py` | What each segment observes |
| `reward_function.py` | Each segment's reward |
| `clocks.py` | The [clocks](#clocks), one per segment or one per leg, and the legs' step shape |
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

1. Keep the current body positions and joint angles as the "previous" ones.
2. With leg clocks, every leg's clock turns and gives the leg's joints their
   targets on the step shape ([Actions and timing](#actions-and-timing)). The
   physics simulation applies the actions and advances 20 ms. With a clock
   per segment, every clock then turns at the tempo its segment chose. Every
   foot that touched the ground before and after the step is measured for
   slip.
3. Each world is checked for arrival or the time limit.
4. The reward function computes each segment's reward; the arrival reward
   needs the result of step 3.
5. The observation builder builds the next observations, and the diagnostics
   are updated.
6. Worlds whose episodes ended get a new target and fresh observations, and
   are reset, clocks included, unless they walk on after an arrival; their
   final observations are returned as well.

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
`(N, 6)`, one row per segment in [action order](#actions-and-timing), and the
spine motors (`segment_{id:02d}_yaw_motor`) in a table of `N − 1` entries,
entry `i` for the joint between segments `i` and `i + 1`, with their joints'
addresses. It then
checks that each motor's name matches the owning segment stored in the XML, that
it drives the right joint and accepts commands from −1 to 1, and that every motor
is found exactly once with none left over (`6N + N − 1`, which is 55 for v1). A
leg motor may instead take target angles, as in model v4: a MuJoCo position
actuator with gear 1 that accepts exactly its joint's range. The leg motors
must all take torques or all take angles, and the simulation records which. It
also checks that every collision shape records its owning segment and its
category (floor, body, leg, left foot, right foot, or membrane; see
[model.md](model.md#contacts-and-friction)). From the head's body shape it
measures the head's **outline** seen from above, the rectangle that encloses
it, measured from the head's centre site: how far it reaches behind and ahead,
and its half-width. The environment reads it from the simulation
(`head_outline`) to decide arrival and the closest distance progress counts.

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

**Torques or target angles.** What a leg action asks for depends on the
model's leg motors. In models v1 to v3 they are torque motors, and an action is
a share of the joint's maximum torque. In model v4 they are position actuators
([model.md](model.md#model-v4)): the simulation maps each action linearly onto
its joint's range, −1 to the lower limit and 1 to the upper, so 0 is the middle
of the range, and the actuator pulls the joint toward that target angle at
every physics step. The model's pose, every leg angle at 0, is the action 0 for
the sweep, −0.231 for the lift, and −0.429 for the knee. The mapping is one
tensor operation per step in the simulation's front file, the same for both
backends, and clips nothing: MuJoCo keeps each target within its joint's range.

With **spine control** (`spine_control = true`), every segment but the rear
also commands the spine yaw motor of the joint **behind** it, as a seventh
action (index 6): the head bends its neck, and each later segment bends the
segment behind it. The rear has no joint behind it and keeps six actions; the
joint action is seven wide, and the rear's seventh column is padding that goes
to no motor. The model stores each spine motor with the unit it sits in, the
one behind the joint ([model.md](model.md#motors)); which agent commands it is
the environment's choice. Without spine control, the spine motors always
receive zero and the spine bends passively against its springs.

**Why the joint behind.** Only the head sees the target, so it is the head
that should be able to turn the front of the body: commanding its neck, it can
point itself at the target directly. Each later segment then sets how the
segment behind it follows, and it sees what that needs with its neighbours'
blocks: the bend ahead of it, read from its leader's orientation and its own,
and its own joint. Commanding the joint ahead instead, as the model's
ownership would suggest, would leave the head without a spine motor and give
its neck to segment 1, which cannot see the target. A real centipede's body
follows its head; nothing here tells the segments to, only that they can.

**A passive follower spine.** With `passive_follower_spine` as well, only the
head commands a spine joint, its neck, as its seventh action; the followers'
joints bend passively against their springs. The head steers, and the body
follows the bend it makes: in models and robots of centipedes, the body's
bending can come from its own mechanics rather than from muscles (Aoi, Egi &
Tsuchiya, Phys. Rev. E, 2013), so the followers need no motor to follow.
Every segment still observes the joint behind it, which tells a follower how
the body bends at its place, and on which side a turn's inside lies.

**The clock's tempo.** With a [clock](#clocks) per segment, every segment
takes one more action, its last: its clock's tempo.

**Leg clocks.** With a clock per leg (`leg_clocks = true`), a segment no longer
commands its leg motors directly. Each leg's clock drives the leg through a
fixed **step shape**, and the segment sets, for each leg and at every step,
the clock's tempo, the size of the steps, and the joints' centres. Its leg
actions are twelve, six per leg, the left leg's first; with spine control,
the spine command follows them, as index 12:

| Index, left leg | Index, right leg | Action |
| ---: | ---: | --- |
| 0 | 6 | Tempo of the leg's clock |
| 1 | 7 | Sweep amplitude |
| 2 | 8 | Lift amplitude |
| 3 | 9 | Sweep centre |
| 4 | 10 | Lift centre |
| 5 | 11 | Knee centre |

At every step, each leg's three joints get their targets, in action units,
from its clock's hand φ:

```text
sweep target = sweep centre + sweep amplitude × cos φ
lift target  = lift centre  + lift amplitude  × max(0, −sin φ)
knee target  = knee centre
```

The simulation maps the targets onto the joints' ranges like any leg action,
so leg clocks need a model whose legs take angles, model v4; a model whose
legs take torques is refused. A positive sweep moves the foot forward on
either side, so over the first half of a turn, φ from 0 to π, the foot sweeps
from front to back with the leg down: the stance, which pushes the body
forward. Over the second half the foot comes forward again, lifted by up to
the lift amplitude: the swing. The amplitudes and centres are raw actions
from −1 to 1, neither clipped nor mapped onto a positive range: a negative
sweep amplitude steps backward, and the agents learn what each value does.

The centres set the legs' posture and, differing between the two sides, can
turn the body. They follow their actions slowly, as a first-order lag with
the time constant `centre_time_constant_s` (τ, 0.5 s): each step covers
`1 − e^(−0.02 s / τ)` of the way, so that a sudden change of a centre action
is 63% covered after τ. Slow centres set posture and turning but cannot make
a rhythm by themselves, which is the clocks' job. On the first step after a
restart, the centres start at their actions, so that an episode starts in
the posture the agent asks for; they keep running through an arrival when
the body walks on.

This arrangement is CPG-RL's (Bellegarda and Ijspeert, IEEE Robotics and
Automation Letters, 2022): oscillators drive the legs through a fixed foot
trajectory, which the joints' controllers track, and the policy, which sees
the oscillators' phases, sets each leg's frequency and amplitude at every
step. Here the position actuators of model v4 track the targets
([model.md](model.md#model-v4)). With amplitudes of 0.5, the centres at the
model's pose, and a wave from the head to the tail, the step shape walks the
body forward open-loop at 10 mm/s, as model.md's walking check measured with
the same shape.

All segments act at the same time and the body moves once for all of them.
Actions are never clipped, because the executed action must be exactly the one
the agent learns from.

An action is held for **20 ms**, or 50 decisions per second: as many physics
steps as the model's timestep fits into it, 134 steps of 0.149 ms for models v3
and v4.
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
- Its default Newton solver needs a GPU of the Volta generation (compute
  capability 7.0) or newer, such as a T4; older GPUs such as the MX330 or the
  P100 cannot compile it. With models v2 to v4, the MX330 can run the GPU
  backend using the conjugate-gradient solver (`gpu_solver = "cg"`), slowly and
  less converged, which is enough for functional tests.

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
| `foot_planar_position` | `(W, N, 2, 2)` | World `x`, `y` of each segment's left and right foot | Not observed; rewards only |
| `spine_yaw_position` | `(W, N)` | Angle of the spine joint behind the segment, rad; 0 for the rear | With spine control, after the target values |
| `spine_yaw_velocity` | `(W, N)` | Its angular velocity, rad/s; 0 for the rear | With spine control, after the target values |

In the block, contact flags become 0 or 1. A segment never observes its position
in the world. Without spine control the spine is not observed; with it, each
segment sees its own spine joint, the one it commands, but not its neighbours',
which it can read from their orientations.

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
   the head; **sideways** is positive to its left;
5. with spine control, the angle and speed of the spine joint behind it;
6. with clocks, its own clock: `cos φ` and `sin φ` of its hand, and its tempo
   action (see [Clocks](#clocks)); with leg clocks, these three values for
   each leg's clock, the left leg's first;
7. with neighbour clocks, the clocks of the neighbours it sees, relative to
   its own, three values each (six with leg clocks), in the order of the
   blocks: the `k` ahead, nearest first, then the `k` behind, which are
   zeros with `coupling = "ahead"`.

That is `(2k + 1) × 27 + 2` values, **83** at radius 1 and 29 at radius 0, for
every segment and any number of segments, two more with spine control
(**85** at radius 1), three more with clocks (**88** at radius 1), and
`6k` more with neighbour clocks (**94** at radius 1). Leg clocks add six and
`12k`: at radius 2 with spine control and neighbour clocks, 137 + 2 + 6 + 24 =
**169**. A segment's clocks are never in the blocks its neighbours see; they
see them only through the neighbour clocks. The spine's values come last so
that a run with spine control can start from agents trained without it: their
inputs keep their
places, and the new ones are added at the end (see
[agents.md](agents.md#checkpoints)). A neighbour that does not exist is filled with zeros, and
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
to its tip and direction: distance uniformly between 30 and 60 mm, about one to
two body lengths, and direction uniformly within 30° left or right of straight
ahead. The target is a point remembered by the environment, not an object in
the simulation. These ranges are `[environment.target]`'s; during training the
experiment's curriculum may change them between update cycles with
`set_target_ranges`, up to targets all around the head (see
[configuration.md](configuration.md#the-curriculum)). Targets already placed
stay until their episode ends.

The head's **forward** direction is its body's x axis, which points from its
centre to its tip, laid flat on the ground; **left** is 90° anticlockwise from
it, the side of the left legs. Targets are placed, and the two target values
observed, in these directions.

**End.** An episode ends in one of three ways, always for all segments
together:

- **Arrival** (terminated): the target lies under the head, seen from above:
  inside the rectangle that encloses the head's body shape, which the
  simulation reads from the model when it loads (7.2 mm long, from 2.2 mm
  behind the head's centre to its tip, and 8 mm wide, for model v3).
- **Time limit** (truncated): the episode reaches `max_episode_steps`, with no
  extra penalty.
- **Leaving the range circle** (truncated, like the time limit): the head's tip
  is farther from the target than 2.5 times its distance at the start
  (`range_circle_ratio`), plus an optional margin (`range_circle_margin_m`),
  75 to 150 mm for these targets without one. Every episode starts with a
  new target placed from the head, so the head always starts inside its
  circle, 1.5 times the target's distance (plus the margin) from its edge.
  The margin gives room to turn: a near target beside or behind the head
  needs a turn that may first take the head away from it, by roughly a body
  length (an estimate, not a measurement).

If arrival and a cut happen on the same step, arrival wins. Nothing else ends
an episode: the centipede may fall or touch the ground with its body and
recover.

**After an arrival.** With `after_arrival = "restart"`, every end restarts the
world from the starting pose. With `"new_target"`, an arrival only gives the
head a new target, placed from where the head is and drawn from the current
ranges, and the body walks on from the pose and speed it arrived with; the time
limit and leaving the range circle still restart the world. Either way the
arrival ends the episode, so its returns and the arrival share are counted as
before, and the step count starts again for the new target. Walking on gives
the agents starts in motion and in every posture a walk passes through, which
restarts never give: from a standing start, each episode first practises
starting to walk, and a centipede that must turn toward a new target never has
to turn while walking. There is no limit to how many targets a body may reach
in a row. The arrival still ends the episode for learning: each agent learns
to reach every target as fast as it can, not to arrive in a state that suits
the next target, which is drawn at random. Since the starting position is not
varied and no segment observes its position on the floor, a body far from
where it started faces the same task.

**Why the head, and why so permissive.** The task is to walk to the target,
however the body gets there, not to aim the head's tip precisely, so the target
only has to come under the head. Only the head sees the target and its distance
makes the progress, so the head, not any segment, decides arrival: counting the
whole body would also count the body sweeping or swinging over targets the
head missed, and with targets in front the head gets there first anyway.

The head is 8 mm wide, so a near target is reached by walking straight ahead,
without steering. The targets are placed far enough that this luck reaches few
of them: at 30 to 60 mm and within 30° of straight ahead, a straight walk
reaches 18%, and an approach is worth 3.5 halvings of the distance (see
[Rewards](#rewards)):

| Targets, with arrival under the head | Straight-walk luck at ±15° | ±30° | ±45° | Halvings per approach |
| --- | ---: | ---: | ---: | ---: |
| 10–20 mm | 94% | | 36% | 1.9 |
| 30–60 mm | 35% | 18% | 12% | 3.5 |
| 40–80 mm | 26.5% | | 8.8% | 3.9 |

**Why the range circle.** Without it, a centipede that walks past its target or
away from it spends the rest of the episode far beyond it, where a step changes
the halvings very little and the target's position lies far outside what the
networks usually see. Leaving the circle restarts the world with a new target,
so those steps become new approaches. It ends the episode like the time limit,
not as a failure: the agents' critic values the state the episode was cut in,
as at the time limit, so the reward and what is best stay unchanged, and
walking away can never become a way to stop paying the costs. At 2.5 times the
start distance, a centipede that overshoots can still walk one and a half times
its first distance past the target and turn back before it is cut. The circle
grows with the target's distance, so it suits any range of targets.

## Clocks

A **clock** is a hand that turns at a tempo an agent chooses. There are two
kinds, and a run has at most one.

**A clock per segment** (`clocks = true`) gives every segment a rhythm of its
own, which it can learn to keep in step with its neighbours'. It is a
designed memory: a segment's networks see only the present, and the clock is
the one thing a rhythm must remember, where the segment is in its cycle. The
clock moves nothing: the legs move with it only as far as the segment learns
to move them so, which the legs-off-tempo cost asks for ([Walking
costs](#walking-costs)), and a neighbour can read the rhythm only from those
legs.

**A clock per leg** (`leg_clocks = true`) drives its leg through the step
shape of [Actions and timing](#actions-and-timing), so the leg moves with its
clock by construction, and the body carries each leg's rhythm to the others
from the first step. Oscillators that walk robots work this way: in CPG-RL
they drive the legs while the policy modulates them (Bellegarda and Ijspeert,
IEEE Robotics and Automation Letters, 2022), and in the quadruped of Owaki
and Ishiguro (Scientific Reports, 2017) they drive the legs while the body's
load pushes their phases, as the load feedback below can.

Both kinds follow the same rules:

- **Tempo.** A tempo action `a`, from −1 to 1, sets its clock's tempo to
  `middle_tempo_hz × 2^(a × tempo_range_octaves)` turns per second: 1 to 4
  with the defaults, so that a turn takes 12.5 to 50 steps.
- **Phase.** Every step, the hand `φ` moves on by `360° × tempo × 0.02 s`.
  Nothing else moves it but load feedback: a clock changes its phase against
  another's only by running faster or slower for a while, as coupled
  oscillators do (Kuramoto, 1984; for locomotion, Ijspeert, Neural Networks,
  2008).
- **Start.** Every restart draws each hand at random, every leg's on its own,
  and sets the middle tempo, so no agreement between clocks comes for free;
  the clocks keep running through an arrival when the body walks on. Hands
  have their own random sequence, seeded with the starting poses' seed.
- **What a segment sees.** Its own hand, as `cos φ` and `sin φ` so that the
  end of a turn and the start of the next look alike, and its tempo action;
  with leg clocks, these three values for each leg, the left leg's first.
  Without `neighbour_clocks`, it never sees another segment's clock: it can
  only read it from that segment's legs.
- **What a segment sees of its neighbours.** With `neighbour_clocks = true`,
  it also sees the clock of each neighbour within its observation radius,
  relative to its own: for neighbour `n` of segment `i`, `cos(φ_n − φ_i)`,
  `sin(φ_n − φ_i)`, and the tempo difference in octaves over the largest
  possible, `(tempo_octaves_n − tempo_octaves_i) ÷ (2 × tempo_range_octaves)`,
  from −1 to 1. With leg clocks, each of the neighbour's legs is seen
  relative to the segment's leg on the same side: six values. A neighbour
  that does not exist gives zeros, like its block. A segment can keep an
  offset to a neighbour's phase only if it can tell where that phase is. The
  policy of CPG-RL observes the phase of every oscillator it coordinates, and
  decentralised leg controllers that learn to walk together observe their
  neighbouring legs (Schilling et al., 2020, arXiv:2005.11164).
- **Coupling.** `coupling` decides which neighbours' clocks count for a
  segment. With `"both"`, the default, a segment sees, and is compared with,
  the neighbours ahead of it and behind it. With `"ahead"`, only those ahead:
  the values of the neighbours behind are zeros, in the same layout, and the
  [out-of-tempo cost](#walking-costs) compares a segment only with the clocks
  ahead of it. The head, with nothing ahead, sees no other clock. Information
  about the rhythm then flows from the head backward, one segment per step at
  radius 1, and the head leads by construction, since nothing behind it pulls
  its tempo. The segment controllers of Chen, Wang and Revzen
  (arXiv:2603.09147) are coupled this way, each taking its input from the
  segment in front, and in insects the walking rhythm of the legs' circuits
  is started and steered by signals descending from the head (Bidaye et al.,
  Journal of Neurophysiology, 2018).

**Load feedback.** With leg clocks, `load_feedback_rad_per_s` (σ, 0 by
default: off) lets the body push the phases. Owaki and Ishiguro's oscillators turn
at `dφ/dt = ω − σ N cos φ`, with `N` the leg's load, which pulls a loaded
leg's phase toward the middle of its stance, at 3π/2 in their convention.
Here stance is the first half of the turn, centred at π/2, so the rule
becomes `dφ/dt = ω + σ N cos φ`, applied once per step as
`φ ← φ + (ω + σ N cos φ) × 0.02 s`, with `ω` the tempo's, in rad/s, and `N` 1
when the leg's foot touched the ground at the end of the last physics step,
else 0. Before mid-stance a loaded leg's hand hurries toward it; after it,
the hand is held back, so a leg that bears the body stays longer in stance
while the others take their turn. Owaki and Ishiguro's quadruped, whose
oscillators interact only through its body, changed by itself from walking
to trotting to galloping as its speed rose.

σ must stay below `ω` at the slowest tempo,
`2π × middle_tempo_hz ÷ 2^tempo_range_octaves` (6.28 rad/s with the
defaults), and the settings refuse more. With `σ N ≥ ω`, a loaded hand stops
where `cos φ = −ω / (σ N)`, near the end of stance, until the load leaves.
But `N` here says whether the foot touches the ground, and the foot leaves
the ground only in swing, which a stopped hand never reaches: the leg would
stay in stance for good. Model v4's open-loop wave
([model.md](model.md#model-v4), Walking), run through the step shape at 2 Hz
(`ω` = 12.57 rad/s) for 5 s, measured:

| σ (rad/s) | Forward speed (mm/s) | Feet on the ground (of 16) | Planted foot-steps sliding faster than 10 mm/s |
| ---: | ---: | ---: | ---: |
| 0 | 10.2 | 8.8 | 44% |
| 6.0 | 13.1 | 9.2 | 33% |
| 6.3 | 13.4 | 9.2 | 34% |
| 12.6 | 0.8 | 15.8 | 2% |

Below `ω` the feedback reshapes the hand-set wave, and the body walks faster
and slips less. At 12.6 rad/s, just above `ω`, every leg's clock stopped in
stance early in the run, after half a turn on average, and the body stood
still on all its feet. 6.3 rad/s still walks at 2 Hz, but at the slowest
tempo, 1 Hz, it could stop the clocks too: 6.3 and 12.6 rad/s lie above the
limit, and the settings refuse them.

The environment's `clocks` holds every hand, `clock_phase` `(W, N, C)` in rad
from 0 to 2π, with `C` one clock per segment or two, the left leg's then the
right's, and the tempo of the last step, `clock_tempo_hz` and
`clock_tempo_octaves` (`log2(tempo ÷ middle tempo)`). `phase`, `tempo_hz` and
`tempo_octaves` `(W, N)` are each segment's own clock, or with leg clocks its
left leg's, which the [rhythm diagnostics](diagnostics.md#rhythm) read as the
segment's. Like the physical state, they are overwritten in place by every
`reset` and `step`.

A clock per segment fixes nothing about the movement itself: it only gives a
segment a rhythm to keep, and two costs judge the rhythm ([Walking
costs](#walking-costs)): whether the legs repeat their movement with the
clock, and whether the clocks agree on a tempo. With leg clocks, only the
shape of a step is fixed; the tempos, the size of the steps, the posture,
and how the legs keep time with one another are learnt, and only the second
cost applies.

**Open questions.** `N` is a contact flag, where Owaki and Ishiguro's is the
load the leg bears: a measured load would let the clock of a lightly loaded
foot move on, and would allow σ above the slowest tempo's `ω`; whether that
is worth having is not settled. With `coupling = "ahead"` the head sees no
other clock; a head that also sees the clocks behind it, without paying for
them, is a possible later study.

## Rewards

The reward follows from a small set of rules, so that every weight is derived
from them instead of being tuned by hand. This section gives the terms, the
rules, and the weights they produce, then two optional costs, off by default,
and the earlier forms of the reward that the program still computes.

### The terms

Every segment `i` receives, on every step:

```text
r_i = A × arrived + w_progress × P − w_step − w_body × B_i − w_leg × L_i
```

| Term | Its value on a step | Who receives it |
| --- | --- | --- |
| **Arrival** | 1 on the step the head arrives at the target, otherwise 0 | Every segment, the same |
| **Progress** `P` | `log2(distance before ÷ distance after)` of the head's tip and the target: how many times the head halved its distance; negative when it moved away, 0 when it did not move | Every segment, the same (`follower_progress_share` = 1) |
| **Step cost** | 1 on every step | Every segment |
| **Body contact** `B_i` | 1 when the segment's body touches the ground | That segment |
| **Leg contact** `L_i` | 1 when one of its legs touches another leg; both segments involved pay | That segment |

In the code, the reward function is a sum of named terms, each multiplied by
the weight that `RewardSettings.weights` works out from the settings.

Feet on the ground cost nothing, and nothing rewards a particular gait. The
design is a **team led by the head**: only the head sees the target, but every
segment is paid for the head's progress, so each one pushes, steers, or holds
back as far as that brings the head closer, and walking past the target costs
all of them what it earned. The head leads by what it knows; how hard the
others push, and when they stop, has to come from training. Posture stays each
segment's own: the contact costs, paid by the segment that touches, are the
only terms that differ between segments.

**Why the followers share the head's progress.** A follower could instead be
paid for its own progress: for halving its distance to the spot where the
segment ahead had been. But the segments are joined by hinges at a fixed
distance ([model.md](model.md), Spine), so a follower cannot lag behind its
leader or catch up with it: its distance to that spot shrinks by exactly how
far the body moved forward on the step. Such a term pays every follower for the
body's speed, wherever it goes, and since the spot moves on with every step,
nothing is ever given back: seven segments paid to push would outweigh the one
trying to steer. Matching the leader's velocity fails for the same reason: with
the distance fixed, two neighbours' velocities differ only when the body turns,
so it would be blind to speed and would penalise the turns the head needs.
Sharing the head's progress is a shared reward copied to each agent, which
keeps every agent's networks, data, and learning separate.

**Why progress counts halvings.** A halving has no unit, so the same rules hold
for any target distance, and it is worth the same at 16 mm as at 2 mm. Over an
episode the progress adds up to `log2(start distance ÷ end distance)`,
whatever path it took: wiggling back and forth earns nothing, and walking past
the target gives back what was earned on the way. Distances closer than half
the head's width (4 mm) count as 4 mm: within the head's reach nothing pays
more, so there is nothing to gain from aiming the tip exactly, and an approach
from 45 mm is worth `log2(45 ÷ 4)` = 3.5 halvings. Moving away is penalised
only by the logarithm of how far the head goes: ending twice as far as it
started costs one halving.

Progress is part of the goal, not only a guide to it: it says that ending closer
is better than not trying (rule R2 below). Pure "potential-based shaping" (Ng,
Harada & Russell, 1999) leaves the best behaviour unchanged by construction, so
it could not make an attempt better than standing still.

### The rules

Every weight is a proportion of the arrival reward `A` (1 by default; learning
only sees proportions, since each update rescales its advantages). A cost is
given by its **share** of the **cost horizon**, a span of `H` steps: the whole
episode, `T` steps (`max_episode_steps`), unless `cost_horizon_steps` sets a
shorter one (see "Why a cost horizon" below). A cost with share `s` costs
`s × A / H` on each step, so the rules hold for any episode length.
The shares are σ for the step cost, β for body contact, and λ for leg contact.

- **R0. The discount fits the episode length.** The agents' discount is
  `γ = 2^(−1/T)`, so an arrival on an episode's last step is worth half of `A`
  seen from its first step. The configuration sets it when a file sets no
  `discount` (0.9973 for 256-step episodes).
- **R1. The worst arrival balances to zero.** An approach of `H` steps that
  pays every cost on every step (body on the ground, legs touching) and
  arrives on its last step adds up to zero, seen from its start with that
  discount: `γ^(H−1) = (σ + β + λ) / H × (1 − γ^H) / (1 − γ)`. With the whole
  episode as the horizon, this gives the **cost budget**
  `σ + β + λ = T × (2^(1/T) − 1)`, about ln 2 = 0.69 for any long episode; over
  a shorter horizon the discount weighs less, and the budget is a little
  larger (0.84 for 256 steps of a 512-step episode). R1 counts only the
  costs: every arrival also earns the progress of its approach, which
  depends only on where it started, and asking the costs to cancel that too
  would make them larger than the arrival itself.
- **R2. Standing still is a little worse than trying badly.** A centipede that
  halves its distance over the horizon with its legs touching all along must
  end better than one that stands still, upright: one halving must be worth
  more than λ. A halving is worth one whole horizon of step cost
  (`head_progress_ratio = 1`, the progress weight per halving equal to σ × A),
  so the attempt ends at −λ and standing still at −σ = −2λ: worse by the
  mildest cost.
- **R3. The order of the costs.** Body contact weighs most, then the step cost,
  then leg contact: by default the budget is split **3 : 2 : 1** (β : σ : λ).
- **R4. Every segment shares the head's progress.** Each follower receives the
  head's progress at `follower_progress_share` = 1. The share must make helping
  the head worth more than a follower's own posture over an episode, or it
  would rather keep still: a full approach from 45 mm, the middle of the
  targets' range, is worth about 0.8 `A`, more than a whole episode of leg
  contact (λ = 0.116) above a share of 0.14, and of body contact (β = 0.347)
  above 0.43. Nothing limits it from above: a follower gains only when the head
  comes closer and loses when it moves away, so its interest and the head's
  never diverge, and with a share of 1 every segment values the approach
  equally.

**Why a cost horizon.** The time limit is a safety net, long enough for the
slowest approach still worth finishing, while most arrivals take a fraction
of it. Measured at the time limit, R1 ties every cost to it: doubling the
episode length halves every cost per step, though nothing about posture
changed, and a typical arrival then pays only a small part of the budget.
With `cost_horizon_steps`, the rules are measured over a typical approach
instead, and the time limit only sets the discount (R0) and cuts episodes. An
arrival later than the horizon that paid every cost all along may then add up
to less than zero, as intended: a slow, untidy arrival should not pay. This is
safe because nothing but an arrival ends an episode early: at the time limit
and at the range circle the agents' critic values what would have followed,
so a negative return can never become a reason to stop. R2 likewise holds
over the horizon: over an episode longer than it, a centipede that touches
its legs throughout must halve its distance more than once to beat standing
still.

Two more properties follow from these rules and need no rule of their own.
Progress cannot be farmed, since it depends only on where the head starts and
ends, so an arrival always sits a whole `A` above a near miss; a full approach
from the farthest targets, 60 mm, earns `log2(60 ÷ 4) × σ ≈ 0.9 A`, about as
much as the arrival itself. And wandering away is penalised only gently, by
the logarithm.

### What the rules give

With `A = 1` and 256-step episodes:

| Term | Weight | From the rules |
| --- | --- | --- |
| Arrival | 1, once | The unit |
| Step cost | 0.00090 per step | σ = 2/6 of the budget (0.231), over 256 steps |
| Body contact | 0.00136 per step | β = 3/6 of the budget (0.347) |
| Leg contact | 0.00045 per step | λ = 1/6 of the budget (0.116) |
| Progress | 0.231 per halving of the head's distance, to every segment | σ × `head_progress_ratio`, shared at `follower_progress_share` |
| Discount | 0.9973 | 2^(−1/256) |

Doubling the episode length halves the per-step costs and sets the discount to
0.9987; nothing else changes. A cost horizon keeps the costs where the
horizon puts them: for example, 512-step episodes with
`cost_horizon_steps = 256` and the costs split 2.5 : 2 : 1.5 (body : step :
legs) give a budget of 0.837, and per step 0.00136 for body contact, 0.00109
for the step cost, and 0.00082 for leg contact, with 0.279 per halving of
progress. Every training run prints these weights under its settings.
Totals per segment over one episode of 256 steps, without discount,
in units of `A`, for targets 30 to 60 mm away; they are the same for every
segment, with its own contacts:

| Outcome | Total |
| --- | ---: |
| Clean arrival from 45 mm in 150 steps | +1.67 |
| Worst arrival from 30 mm: on the last step, body on the ground and legs touching throughout | +0.98 |
| Clean near miss from 45 mm, ending 8 mm from the target | +0.35 |
| Legs touching throughout, halving its distance | −0.12 |
| Standing still, upright | −0.23 |
| Body on the ground throughout, halving its distance | −0.35 |
| Clean walk away, cut by the range circle after 200 steps | −0.49 |
| Worst failure: body down, legs touching, cut by the circle on the last step | −1.00 |

The two cut episodes stop at the circle: what would have followed is left to
the critic's estimate, as at the time limit.

Seen from the start through the discount, the worst arrival comes to about +0.5
(zero for its costs, plus its progress) and standing still to −0.17. Per step,
the progress grows as the head nears the target, so near it a much faster
crawl, with the body on the ground, could earn more than an upright walk.

**Switching a cost off.** A cost of 0 parts is off. The budget is split into
`cost_budget_parts` equal parts, by default the costs' parts together; setting
it higher than their sum leaves part of the budget unused, so that switching
costs off does not change the others. For example, a run that ignores posture
keeps the step cost and progress of the full reward with
`body_contact_cost_parts = 0`, `leg_contact_cost_parts = 0`, and
`cost_budget_parts = 6`.

### Walking costs

Five more costs judge how a segment walks, each from 0 to 1 per step, paid by
every segment for its own legs, and each switched on by giving it parts of the
budget. None prescribes a gait, though both feet down favours standing on
one foot at a time.

- **Foot slip** `S_i` (`foot_slip_cost_parts`): each foot that touches the
  ground at both ends of a step pays how fast it slid along it, in units of
  `foot_slip_unit_m_per_s` (10 mm/s), at most 1; the mean over the two feet.
  A foot standing still on a still body costs nothing, so a segment may stand
  still; a foot dragged along by the moving body costs. When the head walks,
  its neighbours are dragged unless they walk along, and so on down the body:
  this is what makes a walking head pull the others into walking. Foot slip
  costs are common in legged robots (Hwangbo et al., Science Robotics, 2019).
- **Legs off tempo** `Q_i` (`legs_off_tempo_cost_parts`, with a clock per
  segment only; refused with leg clocks, whose legs follow their clocks by
  construction): each leg angle against its angle when the segment's clock
  last passed the same
  point of its turn; the mean over the six leg angles of the difference
  squared, in units of `legs_off_tempo_unit_deg` (20°), at most 1. It asks the
  legs to repeat their movement with the clock, whatever the movement is, so
  that a neighbour can read the clock from them. A leg standing still repeats
  too. The clock remembers the legs at 64 points of its turn, the centres of
  64 equal parts: at each point its hand passes during a step, it takes the
  legs to be between where they were before and after the step, in
  proportion to how far the hand had come, compares them with what it
  remembered there one turn earlier, and remembers them in their place. The
  hand passes every point once per turn, so each comparison is with the last
  turn, and only the first turn after a restart costs nothing.
- **Out of tempo** `D_i` (`out_of_tempo_cost_parts`, with clocks): the mean,
  over the neighbours the segment sees (its observation radius), of the tempo
  difference in octaves divided by the largest possible, 2 octaves. Both
  segments of a pair pay; the head pays `head_tempo_share` (0.25) of its own,
  so that the others follow its tempo more than it follows theirs. Two clocks
  at the same tempo keep their offset forever, so this cost keeps every offset
  steady without choosing it; which offsets suit the legs is left to the leg
  and body contact costs. With leg clocks, each leg's tempo is compared with
  its sibling leg's and with the same side's legs of those neighbours, and
  the segment pays the mean of its two legs. With `coupling = "ahead"`
  ([Clocks](#clocks)), a segment is compared only with the clocks ahead of
  it: each pair's difference is paid by the segment behind alone, so the head
  pays nothing for its neighbours (with leg clocks, only its own legs'
  difference, in full), and `head_tempo_share` has no role: a file that sets
  it is refused.
- **No support** `U_i` (`no_support_cost_parts`): 1 on a step after which
  neither of the segment's feet touches the ground, else 0. A segment carried
  by its neighbours pays; one standing on its feet pays nothing, even
  standing still, so the cost asks every segment to bear its part of the
  body's weight without asking it to move. The legs of a few segments can
  carry the whole body, three with model v4
  ([model.md](model.md#model-v4), Carrying), so without this cost a segment
  can rest its legs while its neighbours carry it.
- **Both feet down** `F_i` (`both_feet_down_cost_parts`): 1 on a step after
  which both of the segment's feet touch the ground, else 0: the counterpart
  of no support. Together they set a rule for a segment's support: one foot
  down is free, both down cost a little, and none down cost more. Both feet
  down is a normal part of slow walking, which is why its cost is kept small:
  model v4's hand-set wave ([model.md](model.md#model-v4), Walking) has 8.8 of
  its 16 feet down on average, so each foot is down 55% of a turn, and with
  the two sides half a turn apart both are down 10% of the time.

**Why support costs on both sides.** No support alone made hanging the
cheaper choice for a segment that cannot step yet. In three training runs
with leg clocks and no support at 1.5 parts (one seed each, model v4), it was
the largest cost: the head's feet were down on only 20 to 27% of steps, the
head hanging from its neck, and segments 3 and 4 paid it on half their steps.
Hanging cost at most 0.00070 per step, while standing on feet that the moving
body drags cost up to 0.00093 of foot slip (2 parts), plus legs touching. A
run can give no support more parts than foot slip, so that hanging is never
cheaper than standing on dragged feet, and both feet down a small share, so
that a segment standing on both feet gains by lifting one; the user chose
this pair of costs on 2026-10-10.

**The head's task alone.** A reward may also give the task to the head alone:
`follower_arrival_share = 0` and `follower_progress_share = 0` leave the
followers only their costs, which their own observations show, and the head
leads by walking. Without a step cost (`step_cost_parts = 0`), a halving is
given in parts of the budget (`progress_parts`), and `arrival_payout` pays the
arrival a share of `A`, the unit the rules measure every cost in: at 0.5 the
costs weigh twice as much against the arrival, so that the head cares how it
walks even when it arrives often. A budget split into fewer parts than the
costs take (`cost_budget_parts`) makes them overspend it, and the worst
arrival then ends below zero; it still pays once its progress is counted.

With the head's task alone, these proportions keep walking worth more than
standing still. Over the horizon, one halving of the distance (3 parts) is
worth more than legs touching (2 parts) or legs off tempo (1.5 parts) alone,
but a little less than both together: a careless walk at the slowest pace
still loses a little, a careful one gains, and standing still gains nothing.
Body contact weighs most (2.5), then legs touching and foot slip (2), legs off
tempo (1.5), and out of tempo least (1). With leg clocks, no support takes
the 1.5 parts of legs off tempo, so that every other cost keeps its weight.

### Optional costs

Two more costs can be added to the reward, each charged to every segment for
its own joints. Both are off unless a configuration sets them:

```text
r_i = … − w_move × M_i − w_command × C_i
```

- **Movement** `M_i` (`movement_cost_parts`): how far the joints the segment
  commands moved on the step, squared and averaged over them, in units of how
  far random commands move a joint in one step (`random_command_movement_deg`,
  25° for model v3); at most 1. It takes its parts of the cost budget like the
  other costs, so rule R1 still holds. The square makes a sudden movement
  expensive and a calm one cheap: ten steps of 1° cost a tenth of one step of
  10°, and holding still costs nothing. It is measured on the joints the agent
  observes, so an agent can tell from its own observation what a choice will
  cost.
- **Command** `C_i` (`command_cost_ratio`): the segment's commands for the
  step, squared and averaged over the joints it commands: 0 for none, 1 for
  full commands. It charges what the segment asks for, exploration noise
  included, whether or not the body follows. It is the control cost of
  Gymnasium's Ant (Towers et al., 2024), with a weight that is a multiple of
  the step cost's, `w_command = command_cost_ratio × w_step`. It lies outside
  the cost budget, so with it the worst arrival no longer balances to zero
  (rule R1).

### Earlier forms of the reward

The program also computes two earlier forms of the reward, so that runs
configured with them can still be continued and evaluated.

**The efficiency reward.** A file that sets any of `efficiency_cost`,
`body_contact_cost`, or `leg_contact_cost` uses per-step weights and one
efficiency term in place of the step cost and progress:

```text
r_i = A × arrived + E_i − c_body × B_i − c_leg × L_i
E_i = −c_efficiency × (ε + distance after) / (ε + distance before)
```

The efficiency term holds the step cost (`−c_efficiency` when standing still)
and the progress (`c_efficiency × gain ÷ distance`) under one weight, so the
progress is only a small part of it; the rules give progress a weight of its
own instead. A file that also sets the rules' proportions is refused.

**The followers' own progress.** A file that sets `follower_progress_ratio`
pays each follower, at that share of the head's weight, for halving its own
distance to the spot where the segment ahead had been, and gives it no share
of the head's progress unless it also sets `follower_progress_share`. As
explained under [The terms](#the-terms), this pays for the body's speed rather
than for approaching the target.

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
| `spine_control` | false | Every segment but the rear commands the spine joint behind it and observes its angle and speed; when false, the spine motors receive zero |
| `passive_follower_spine` | false | With spine control, only the head commands a spine joint, its neck; the followers' joints are passive. Every segment still observes the joint behind it |
| `clocks` | false | Every segment has a [clock](#clocks): one more action, its tempo, and three more observed values |
| `leg_clocks` | false | Every leg has a [clock](#clocks) that drives it through a [step shape](#actions-and-timing): twelve leg actions per segment instead of six, and six observed clock values; needs a model whose legs take angles (v4); excludes `clocks` |
| `neighbour_clocks` | false | Every segment also observes the clocks of the neighbours it sees, relative to its own: three more values per neighbour (six with leg clocks), `6 × observation_radius` in all (`12 ×` with leg clocks); needs clocks of either kind |
| **`[environment.clock]`** | | |
| `middle_tempo_hz` | 2 | The tempo at a tempo action of 0, in turns per second |
| `tempo_range_octaves` | 1 | How far a tempo action of ±1 moves the tempo, in octaves; the fastest tempo, plus `σ / 2π` with load feedback, must stay below 12.5 Hz |
| `load_feedback_rad_per_s` | 0 | With leg clocks, σ of the [load feedback](#clocks): a foot on the ground holds its clock in stance; 0 is off. Must stay below 2π times the slowest tempo, 6.28 rad/s with the defaults, or a clock could stop for good |
| `centre_time_constant_s` | 0.5 with leg clocks, else 0 | With leg clocks, the time constant with which each joint's centre follows its action ([Actions and timing](#actions-and-timing)); 0 follows at once |
| `coupling` | `"both"` | Which neighbours' clocks a segment sees and is compared with: `"both"`, those ahead and behind; `"ahead"`, only those ahead, so that the rhythm passes from the head backward ([Clocks](#clocks)) |
| **`[environment.simulation]`** | | |
| `model_path` | Required | Model file to load |
| `backend` | Required | `cpu` or `gpu` |
| `world_count` | 1 | Number of worlds simulated at once |
| `gpu_solver` | `newton` | Constraint solver, GPU only: `newton`, or `cg` for GPUs older than Volta |
| `contacts_per_world` | 128 | Reserved contact memory, GPU only |
| `constraints_per_world` | 512 | Reserved constraint memory, GPU only |
| `start_heading_range_deg` | 0 | A reset turns the whole body about the vertical by a random angle within ± this many degrees; 180 allows any heading |
| **`[environment.target]`** | | |
| `distance_range_m` | 0.030 to 0.060 | Distance of a new target from the head's tip; with a curriculum, its level 0 |
| `bearing_range_deg` | −30 to 30 | Direction of a new target from straight ahead; with a curriculum, its level 0 |
| `arrival` | `"head"` | `"head"`: the target must lie under the head's outline; `"tip"`: the head's tip must come within `arrival_radius_m` |
| `arrival_radius_m` | Not set | With `arrival = "tip"` (0.001 when unset): the distance that counts as arrival, and the closest distance progress counts. A file that sets it uses the tip |
| `range_circle_ratio` | 2.5 | The range circle's radius around the target, as a multiple of the head's distance at the start; leaving it ends the episode like the time limit. 0: no circle. With the tip, no circle unless the file sets one |
| `range_circle_margin_m` | 0 | Added to the range circle's radius: room to turn, whatever the target's distance |
| `after_arrival` | `"restart"` | `"restart"`: an arrival restarts the world, like every other end; `"new_target"`: the body walks on from where it arrived, toward a new target ([Episodes](#episodes)) |
| **`[environment.rewards]`** | | |
| `arrival_reward` (`A`) | 1.0 | Shared reward for reaching the target: the unit of every other weight |
| `arrival_payout` | 1 | What an arrival pays, as a share of `A`; the costs keep the weights `A` gives them |
| `follower_arrival_share` | 1 | The share of the arrival each follower receives as well |
| `step_cost_parts` | 2 | The step cost's parts of the cost budget |
| `body_contact_cost_parts` | 3 | Body contact's parts of the cost budget |
| `leg_contact_cost_parts` | 1 | Leg contact's parts of the cost budget |
| `foot_slip_cost_parts` | 0 | [Foot slip](#walking-costs)'s parts of the cost budget; 0 is off |
| `legs_off_tempo_cost_parts` | 0 | [Legs off tempo](#walking-costs)'s parts; needs a clock per segment, and is refused with leg clocks |
| `out_of_tempo_cost_parts` | 0 | [Out of tempo](#walking-costs)'s parts; needs clocks of either kind |
| `no_support_cost_parts` | 0 | [No support](#walking-costs)'s parts; 0 is off |
| `both_feet_down_cost_parts` | 0 | [Both feet down](#walking-costs)'s parts, the counterpart of no support; 0 is off |
| `foot_slip_unit_m_per_s` | 0.010 | Foot slip's unit: the sliding speed that costs 1 |
| `legs_off_tempo_unit_deg` | 20 | Legs off tempo's unit: the difference that costs 1 |
| `head_tempo_share` | 0.25 | The share of its out-of-tempo cost the head pays; with `coupling = "ahead"` the head pays nothing for its neighbours, and a file that sets it is refused |
| `movement_cost_parts` | 0 | The [movement cost](#optional-costs)'s parts of the cost budget; 0 is off |
| `random_command_movement_deg` | 25 | The movement cost's unit: how far a joint moves in one step under random commands (the root of the mean square, measured for model v3). It was measured with v3's torque motors and does not hold for v4, whose leg actions are target angles |
| `command_cost_ratio` | 0 | The [command cost](#optional-costs)'s weight `w_command`, in step costs: each step a segment pays this many times the step cost's weight times `C_i`; 0 is off. Outside the cost budget, so rule R1 no longer holds when it is set. Gymnasium's Ant charges 0.5 times the sum of its 8 squared commands against a reward of 1 per healthy step, a ratio of 4 |
| `cost_budget_parts` | The costs' parts together | How many equal parts the budget is split into; more than the costs' parts leaves some unused, fewer makes the costs overspend it |
| `cost_horizon_steps` | `max_episode_steps` | The cost horizon: the rules are measured over an approach of this many steps ("Why a cost horizon" under [The rules](#the-rules)); shorter than the episode, it makes every cost and the progress weigh more against the arrival |
| `head_progress_ratio` | 1 | What one halving of the head's distance is worth, in whole cost horizons of step cost |
| `progress_parts` | Not set | Instead of `head_progress_ratio`: what one halving is worth, in parts of the budget; a reward without a step cost needs it |
| `follower_progress_share` | 1 | The share of the head's progress each follower receives as well (0 in a file that sets `follower_progress_ratio`) |
| `follower_progress_ratio` | Not set | Each follower's [own progress](#earlier-forms-of-the-reward) toward the spot where the segment ahead had been, as a share of the head's weight |
| `distance_ratio_epsilon_m` (`ε`) | 0.000001 | Keeps the efficiency reward's ratio defined |
| `efficiency_cost`, `body_contact_cost`, `leg_contact_cost` | Not set | Per-step weights of the [efficiency reward](#earlier-forms-of-the-reward) (0.003, 0.010, and 0.005 when any is set); setting one selects that reward |

The episode length is a hyperparameter like the others. The GPU memory values
leave headroom over the 40 contacts and 256 constraint rows measured at rest,
and hold in motion. The reward's proportions come from the rules in
[Rewards](#rewards), not from measured biology.
