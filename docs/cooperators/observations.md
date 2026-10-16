# Observations

**Status (2026-10-16): agreed with the user and built; the joint forces
added with the chain ([The joints around it](#the-joints-around-it-agreed-2026-10-15)),
and from cycle 6 the 46-value block of cycle 5 is the only one.**

## Decided

- **Only the neighbourhood.** When it acts, a segment sees only itself and
  its neighbours within the observation radius (the constraints in
  [README.md](README.md)). The head also sees the target, which only it can
  know.
- **Radius 2 to start.** A segment sees five segments: itself, two ahead
  and two behind. The tail-ward wave of a real centipede's legs spans a few
  legs, about five in the Tohoku model (Yasui et al., Frontiers 2022), so a
  window of five covers about one wavelength around each segment: what it
  needs to sense where it sits in a wave. A coincidence of scales, not
  proof, but it points the same way. A missing neighbour (ahead of the
  head, behind the rear) is all zeros, as on main.
- **The critics see a little more, in training only:** what their actor
  sees, plus the neighbours' actions of the same step
  ([critic.md](critic.md)).

## What describes a segment (agreed)

Each segment's **block** is what that segment senses of itself; an actor
sees five blocks. For v4, with its eight leg joints:

| Part | Values | What it is | Why |
| --- | ---: | --- | --- |
| Gravity's direction | 3 | The downward direction in the segment's own frame | Its tilt, without its heading in the world, which means nothing for the task ([task.md](task.md#the-starting-heading-agreed-2026-10-14)); legged_gym's "projected gravity" |
| Body speed | 3 | The segment's velocity in its own frame | Its own motion, the cause of its path progress (the pairing rule) |
| Turning rate | 3 | Its angular velocity in its own frame | Turning and tilting |
| Leg angles | 8 | Every leg joint's angle | Where its legs are: proprioception |
| Leg speeds | 8 | Every leg joint's angular velocity | How they move; with the torques, the work its motors do |
| Spine | 2 | The yaw angle and speed of the joint behind it | The joint it commands; across the five blocks, the body's local bend, which the trail follows |
| Foot loads | 4 | Each foot's force on the ground: pressing down, and pushing fore or aft | **Load sensing**, the signal biology puts at the root of the wave (below) |
| Belly load | 1 | The body's force on the ground | Resting or dragging, which energy prices |
| Leg-on-leg force | 1 | The force with which its legs press on other legs | The cause of pain, so the critic can see it |
| Last action | 9 | The commands it sent last step | What it is doing; across the blocks, the neighbours' last actions (the hexapod; legged_gym) |
| **Block** | **42** | | |

**The head also sees the target:** its position relative to the head's
tip, in the head's own frame, forward and sideways (2 values, as on main).

| Input | Size |
| --- | ---: |
| An actor: five blocks and the target's two values (zeros for the followers) | 212 |
| A critic: the actor's input and the four neighbours' actions of the same step | 248 |

Forces are in units of one unit's weight, so the inputs mean the same on
the 21-segment model. Each agent normalises its own inputs, as on main.

**Why load sensing.** In every source on how a gait emerges without a
central clock, the signal is a leg's load. Real centipedes: a leg moves
again when the leg ahead touches the ground, and the model's legs react to
their own and their neighbours' ground forces (Yasui et al., PLOS ONE 2017).
Stick insects: the load on a leg coordinates its neighbours (Dallmann et
al., Proc. R. Soc. B 2017). A quadruped robot: local load sensing alone
gave walk, trot and gallop (Owaki and Ishiguro, 2017). The oscillators
found the same: their wave formed only with load feedback (runs 15 and 20
against 16 and 25). Main gives only a contact flag per foot; a force says
how much the leg carries, which is what the neighbours need to know.

**What changes from main's block** (31 values,
[environment.md](../environment.md#the-physical-state)):

| Main | Here |
| --- | --- |
| Orientation as a quaternion, heading included | Gravity's direction only |
| Body height | Left out: no animal senses it directly; leg angles and loads tell it on flat ground, and legged_gym leaves it out |
| Foot, belly and leg-leg contact flags | Forces, graded |
| Spine only with spine control, outside the block | In the block, since every segment but the rear commands a spine joint |
| No actions | The last action |

**A perfect sense of speed.** How fast a segment moves is the agents'
choice, learned like everything else; what comes from the simulator is
only the measurement a segment receives of its own speed, exact, like a
perfect speedometer. A real animal has no such sense: it estimates its
speed from what it sees and from its legs, and robots estimate theirs from
an inertial sensor and leg motion. The project is a simulation, so the
exact reading is a choice of simplicity, not of realism; legged_gym's
policies read the simulator's body velocity the same way.

## The joints around it (agreed, 2026-10-15)

With the chain ([reward.md](reward.md#the-chain-push-and-pull-between-neighbours-the-users-design)),
each segment pays for the force on its two joints, so by the pairing rule
it must see them. **Each block holds one more value: the force of the
segment ahead on its segment, along the body**, in unit weights, positive
when it pulls (the segment lags), negative when the segment pushes; zero
for the head. Since a segment's rear joint is the front joint of the
segment behind, the five blocks show every joint around it once: its own
front joint in its own block, its rear joint in the block behind, and
three more, two ahead and one behind. From the difference between two
neighbouring joints a segment can tell which segment is pushing or lagging,
and so react to what is coming before its own joints feel it: the user's
aim of anticipation, through what a segment sees rather than through what
it pays. Only the part along the body is seen, as only it is priced.

| Input | Size |
| --- | ---: |
| A block, with the joint force | 43 |
| An actor: five blocks and the target's two values | 217 |
| A critic: the actor's input and the four neighbours' actions | 253 |

**Cycle 4: the offset and the net joint force** (the user's design,
2026-10-16; [reward.md](reward.md#cycle-4-the-trail-and-carrying-oneself-the-users-design)).
With the trail, each block holds its segment's offset from its trail, signed,
positive to the left, in radians; with carrying oneself, the net force its
two joints put on it, along the body and up, in unit weights. The pairing
rule asks for both: a segment pays its own offset and that of the segment
behind it, which it steers, and both lie within its radius; and the net
force is what it pays for carrying itself. Seeing the offsets of the
segments ahead also previews the trail it will walk: a turn shows in their
offsets before it reaches its own.

| Input | Size |
| --- | ---: |
| A block, with the offset and the net joint force | 45 |
| An actor: five blocks and the target's two values | 227 |
| A critic: the actor's input and the four neighbours' actions | 263 |

**Cycle 5: the forces it pays, averaged** (the user's design, 2026-10-16;
[reward.md](reward.md#cycle-5-coordination-first-the-users-design)). With
coordination, each block holds both the front joint's force and the net
joint force, averaged over 0.2 s as the reward prices them, so that each
critic sees the very numbers its coordination cost is made of: its own,
and within its radius those of the segments it pushes or pulls. The
offset comes from the averaged trail.

| Input | Size |
| --- | ---: |
| A block, with the front joint's force, the net joint force and the offset | 46 |
| An actor: five blocks and the target's two values | 232 |
| A critic: the actor's input and the four neighbours' actions | 268 |

**Cycle 6 keeps it** ([reward.md](reward.md#cycle-6-distance-on-the-trail-the-squeeze-its-price-the-users-design)).
Every term a segment is paid or charged is in its own block or its
neighbours': its front joint's squeeze in its own block, the head's push
from behind in its net joint force, its hanging in its net joint force and
belly load, the offset of the segment it steers in the block behind, and
its distance in its own velocity. The settings that chose the earlier
blocks are gone: the block is always this one.

**Cycle 7: the head's line** (the user's design, 2026-10-16;
[reward.md](reward.md#cycle-7-the-heads-line-the-users-design)). The trail
offset gives way to three values of the segment's tail on the head's line
(`HeadLine.view`): its distance from the line, signed, positive to the
left; the angle of its body from the line's direction there; and its
distance from the line it has not erased, which its goal charges. The
pairing rule holds: what a follower is paid and charged, the line it
erases and the distance it is left behind, lies in its own block. The line
is the head's, but a follower sees only its own tail's place on it, inside
its sphere, never the head; and since its front end is the tail of the
segment ahead, its five blocks give five samples of the line around it, so
that a turn shows before it arrives. The head's block holds its neck's
values, which nothing pays.

| Input | Size |
| --- | ---: |
| A block, with the head's line (3) in place of the offset | 48 |
| An actor: five blocks and the target's two values | 242 |
| A critic: the actor's input and the four neighbours' actions | 278 |

## As built

The block's values in order: gravity's direction (3), velocity (3), angular
velocity (3), the eight leg angles, the eight leg speeds, the spine joint's
angle and speed, the feet's forces pressing down (left, right) and along the
segment (left, right), the belly's force, the force of its legs pressed by
others, the front joint's force, the net joint force (along, up), both
averaged over `averaging_s` as the reward prices them, the tail's place on
the head's line (signed distance, angle, distance from the line not yet
erased), and the last action (9; the rear's ninth is always zero),
as the policy sent it, before the spine's scale. The
forces are means over the step's physics steps (40 on v4), in unit weights of
0.857 mN; after a reset, the starting pose's own. The target comes last,
forward and to the left of the head's tip, in metres, zeros for the
followers. Each agent normalises its inputs with running statistics, as on
main.

## Later experiments

The radius changed, to show how much seeing the neighbours matters for
cooperation to emerge; then the final model of 21 segments with several
radii ([README.md](README.md#the-plan)).
