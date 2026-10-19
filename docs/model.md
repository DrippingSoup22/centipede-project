# Model

The model is the centipede's body: MuJoCo XML files describing its shape,
joints, motors, masses, and contacts. It contains no program logic and is not
part of the pipeline; the configuration names the file, and the physics
simulation loads it (see the [architecture](architecture.md)).

## Status

The newest model is **v5**, the body of a real species, *Scolopendra
heros*: a head without legs and 21 leg-bearing units at the animal's
measured length and mass, with v4's physics scaled to its size (see
[Model v5](#model-v5)). It comes in two lengths, the whole animal and its
first ten units, a tester that screens a design quickly before the whole
body trains it. **v4** has twelve body units, and legs of three
links with a fourth joint, the ankle, with v3's torque motors, leg armature,
and a 0.5 ms physics timestep (see [Model v4](#model-v4)). **v3**, which the
example configurations use, is v2 with a longer physics timestep (see
[Model v3](#model-v3)). v2 has exactly the body and physics of the original
**v1 flat-ground baseline**. v2 and v1 are kept unchanged for reference. All
five are frozen: training work may change observations, rewards, targets,
and learning, but never a model. A physical
change requires a new, separately named version with its own validation. The
two exceptions, each made before any training had used the model, by the
user's decision: v4's physics were revised in place on 2026-10-15
([Physics of v4](#physics-of-v4)), and v5's simulation settings on
2026-10-19, when its file was split into the two lengths
([Physics of v5](#physics-of-v5)).

**MuJoCo version.** The program runs on MuJoCo 3.16.0 and MuJoCo Warp
3.16.0, which v5 was validated with ([Validation of v5](#validation-of-v5)).
v1 to v4 were validated with MuJoCo 3.12.0, and their validations have not
been repeated under 3.16.0. Their automated tests pass under it, and it
compiles them to the same bodies but for one part: v4's tarsus, a 0.6 mg
link, gets another inertia, its smallest principal moment ten times
smaller. A model of v1 to v4 is validated again before it is trained on.

| File | Purpose | SHA-256 |
| --- | --- | --- |
| [`models/assembly_v5_full.xml`](../models/assembly_v5_full.xml) | **v5**, *Scolopendra heros*: a head and 21 leg-bearing units, 15 physics steps per action | `8d2dd3e433feb5d710b176cd73c3b0dfc35c18bca57ca0d367de86293d280cec` |
| [`models/assembly_v5_half.xml`](../models/assembly_v5_half.xml) | **v5's half**, its first ten units: the tester | `13773151d9468b5778ea65f0b67b472539a100bf67616b03e4cdff8ca04f55f1` |
| [`models/assembly_v4.xml`](../models/assembly_v4.xml) | **v4**, twelve units with four-joint legs, 40 physics steps per action | `ae43527f871ca116af51b4befc21f62b710c509442f0cf15c8c681243fca95e0` |
| [`models/assembly_v3.xml`](../models/assembly_v3.xml) | **v3**, eight units with three-joint legs, used by the example configurations | `8a693cc5e15e4a468a6cafb293f67be7fab6460f7b76f0ca948546e59cb336fe` |
| [`models/assembly_v2.xml`](../models/assembly_v2.xml) | v2, the same body with a 0.1 ms timestep | `eb3a89fcb640cdb7b75045c8848e1b9c4fde2c230d0c5a7e8115bf89f1d8a14a` |
| [`models/assembly.xml`](../models/assembly.xml) | v1, the original baseline | `92143cf54b030856e436a1de6f4333c327ffb9a5f812a0e4a0b4c4aed107d50e` |
| [`models/segment.xml`](../models/segment.xml) | One isolated trunk unit, for reference | |
| [`models/head.xml`](../models/head.xml) | The isolated head, for reference | |

v2 changes three things and nothing else:

1. **Which shapes may collide.** Bodies collide only with the floor; legs and
   feet collide with the floor and with each other. Legs never reached a body
   and bodies never reached each other in v1's validation, so these pairs only
   cost time. Removing them also removes every pair that needs MuJoCo Warp's
   slow general collision code (see [Contacts](#contacts-and-friction)).
2. **Separate categories for left and right feet**, so every shape's role can be
   read from the model without looking up names.
3. **Solver tolerance 1e-6 instead of 1e-10**, the value MuJoCo Warp enforces,
   so the CPU and GPU solve to the same standard. The unused generator comment
   and configuration block were also removed.

## The body at a glance

This section and those up to [Validation](#validation) describe the body of v1
to v3. [Model v4](#model-v4) lists what v4 changes; everything it does not list
is as described here.

- Eight body units: one head and seven trunk units.
- Sixteen two-link legs, one left/right pair on every unit.
- Each leg has three hinges: shoulder sweep, shoulder lift, and knee.
- Seven flexible connections between units, each with a yaw and a pitch hinge;
  rolling between units is locked.
- A free root, so the whole body can move and rotate in 3D. It is the free
  joint of the head's body, `segment_00`, a child of the world (v5's is its
  middle unit's): turning it turns the whole centipede, which is how a reset
  varies the starting heading.
- 69 position coordinates, 68 velocity coordinates, and 55 motors.
- Total mass 643.4 mg.

This is a simplified centipede. It keeps the body chain, paired legs, the main
leg motions, and sideways body bending, but omits antennae, forcipules, extra
leg sections, soft tissue, a specialised tail unit, and the real number of
segments. Eight units keep the first cooperative problem at eight agents; the
reference species has 19 leg pairs.

## Body

The rigid body is about **33.8 mm long, 8 mm wide, and 3.4 mm high**, without
legs.

| Property | Value |
| --- | ---: |
| Trunk unit length | 4.4 mm |
| Body width / height | 8.0 / 3.4 mm |
| Distance between unit centres | 3.8 mm |
| Overlap between neighbouring units | 0.6 mm |
| Extra length at the front of the head | 2.8 mm |

Each unit is a convex rounded box with an arched back, flatter belly, and
bevelled ends. The head has the same rear shape as a trunk unit plus a rounded
front. The last unit is an ordinary trunk unit.

The width and leg-length ratio follow measurements of *Scolopendra polymorpha*.
Unit length, height, and head extension are engineering choices, because no
complete 3D body scan was available.

## Legs

Every unit carries two identical, mirrored legs. Each leg has an upper link, a
lower link, and a spherical foot.

| Part | Value |
| --- | ---: |
| Upper leg length / radius | 4.6 / 0.30 mm |
| Lower leg length / radius | 5.8 / 0.22 mm |
| Total leg length relative to body width | 1.30 |
| Foot radius | 0.32 mm |

The shoulder has two hinges at the same point. **Sweep** moves the leg forward
and backward; **lift** raises and lowers it. The **knee** is one hinge standing in
for several real leg joints.

| Joint | Range | Maximum torque | Damping | Armature |
| --- | ---: | ---: | ---: | ---: |
| Shoulder sweep | ±40° | ±8 µN·m | 2e-7 N·m·s/rad | 2e-11 kg·m² |
| Shoulder lift | −25° to +40° | ±18 µN·m | 2e-7 N·m·s/rad | 2e-11 kg·m² |
| Knee | −20° to +50° | ±10 µN·m | 1e-7 N·m·s/rad | 2e-12 kg·m² |

The upper link starts 10° below horizontal, so the lift range corresponds to
roughly −35° to +30° of real elevation. The two links start 25° apart, giving a
knee bend of about 5° to 75°.

The sweep range is based on measured leg motion in *S. polymorpha*. No suitable
3D measurements were found for lift or the knee, so those ranges are engineering
approximations.

## Spine

Each unit after the head is attached to the unit ahead of it by two hinges.

| Hinge | Range | Stiffness | Damping | Motor |
| --- | ---: | ---: | ---: | --- |
| Yaw (sideways bending) | ±15° | 2e-5 N·m/rad | 3e-7 N·m·s/rad | ±15 µN·m |
| Pitch (up-down bending) | ±8° | 1e-4 N·m/rad | 5e-7 N·m·s/rad | None, passive |

Both hinges have a weak spring pulling them back to straight, and an armature of
2e-11 kg·m². At their limits, the springs give about 5.2 µN·m (yaw) and
14.0 µN·m (pitch).

The yaw range is based on sideways bending measured in *Scolopendra heros*:
typical amplitudes of 4° to 8.5°, with peaks near 15°. The pitch range is a
smaller engineering approximation.

## Motors

All 55 motors take a command between −1 and 1, which the model scales to the
maximum torques above. The head owns its six leg motors; every other unit owns
its six leg motors and the yaw motor connecting it to the unit ahead, giving
6 × 8 + 7 = 55. Pitch has no motor. Each motor's owning unit is also stored in the
XML (`actuator_user`), so ownership can be checked without relying on the order
of motors in the file. Ownership says which unit a motor sits in; which agent
commands it is the environment's choice: with spine control, each yaw motor is
commanded by the unit ahead of its joint. How motors become agent actions is
described in [environment.md](environment.md#actions-and-timing).

## Mass and inertia

| Part | Mass |
| --- | ---: |
| Head body | 90 mg |
| Each trunk body | 73 mg |
| Each upper leg | 1.50 mg |
| Each lower leg | 1.00 mg |
| Each foot | 0.15 mg |
| Head unit with legs | 95.3 mg |
| Trunk unit with legs | 78.3 mg |
| **Whole model** | **643.4 mg** |

MuJoCo computes each part's inertia from its shape and the masses above; no
inertia values are written by hand. A general length-to-mass formula for this
group of centipedes predicts about 0.55 g for a body this size, which serves only
as a sanity check.

## Contacts and friction

Each collision shape stores two numbers in the XML (`user="owner category"`):
the segment that owns it (−1 for the floor) and its category. The category
decides what it may collide with:

| Category | Shapes | Collides with |
| ---: | --- | --- |
| 0 | Floor | Everything except membranes |
| 1 | Segment bodies | Floor only |
| 2 | Leg links (upper and lower) | Floor, legs, and feet |
| 3 | Left feet | Floor, legs, and feet |
| 4 | Right feet | Floor, legs, and feet |
| 5 | Visual membranes | Nothing |

This is set with MuJoCo's collision bitmasks: floor `contype 1, conaffinity 1`,
bodies `2, 1`, legs and feet `4, 5`. Two shapes collide when either one's
`contype` shares a bit with the other's `conaffinity`. MuJoCo also never tests a
body against its own parent body. Leg-leg contact is deliberately on, so that
avoiding it is part of what the agents must learn.

All remaining pairs have fast dedicated collision routines on both CPU and GPU:
body meshes meet only the floor plane, and everything else is spheres, capsules,
and the plane.

| Contact | Sliding | Torsional | Rolling |
| --- | ---: | ---: | ---: |
| With the ground | 0.8 | 1e-4 m | 1e-5 m |
| Between body parts | 0.8 | off | off |

These are engineering values, not measured centipede properties. Real traction
also depends on surface roughness, claws, hairs, and soft feet, which the model
omits.

As a check, pushing the body sideways for 0.1 s with 20% of its weight moved the
head 1.101 mm with sliding friction 0.2, and only 0.0177 mm with the chosen 0.8.
All 16 feet stayed on the ground in both cases.

## Simulation settings

| Setting | Value |
| --- | ---: |
| MuJoCo version | 3.16.0 (v1 to v4 were validated with 3.12.0, see [Status](#status)) |
| Integrator | `implicitfast` |
| Physics timestep | 0.149 ms, 134 steps per 20 ms action (v1 and v2: 0.1 ms) |
| Gravity | 9.81 m/s² |
| Contact response time | 0.3 ms |
| Joint-limit response time | 0.8 ms |
| Contact damping ratio | 1.0 |
| Solver tolerance | 1e-6 (v1: 1e-10) |

For v1, halving the timestep to 0.05 ms changed body positions by at most
0.006 mm and spine angles by at most 0.002°, so 0.1 ms was accurate enough. v3
lengthens it to 0.149 ms for speed (see [Model v3](#model-v3)). Agents act far
less often than every physics step (see
[environment.md](environment.md#actions-and-timing)).

## Visual membranes

Seven dark ellipsoids cover the seams between units, purely for appearance. They
have no mass, no collisions, and no joints, so they do not affect the physics.
Small gaps can show at strongly bent seams. This does not matter on flat ground,
but a future task with small obstacles would need to revisit the seams.

## Validation

Before it was frozen, the model passed these checks:

| Check | Result |
| --- | --- |
| Standing still, 2 s | All 16 feet down, no body contact, 0.004 mm deepest ground penetration |
| Scripted body bend, 2 s | All 16 feet down, no body contact, 9.78° yaw, no limit exceeded |
| Spine release | Bent spine returns to within 0.1° of straight in 0.25 s |
| Random full-strength motor commands, 2 s | 0.176 mm deepest penetration, joint limits exceeded by at most 4.23° (soft limits), no solver warning |
| Spine at its hard limits, 18 poses | No body-body or body-leg contact |
| Legs at their limits, 672 poses | No body-body or body-leg contact; leg-leg contact allowed |
| Membranes on and off | Identical mass, inertia, and contacts |
| Half timestep | All standing, bending, and pushing checks still pass |

These checks show the model is stable and mechanically capable. They do not show
that it can walk; that is what learning has to achieve.

With every motor off, the body cannot hold itself up: it settles onto its belly
within about 33 ms, in both versions. Holding the body off the ground is the
first thing the agents have to learn.

v2 was checked against v1 with these results. The automated tests in
`tests/models/test_model_v2.py` repeat the first three checks and one of the
random-command runs:

| Check | Result |
| --- | --- |
| Body, joints, motors, masses, inertia, friction | Identical to v1 |
| Collision rules and categories | As in the table above; only mesh–plane pairs involve a mesh |
| Settling for 2 s with motors off | Same contacts as v1, body positions within 0.016 mm |
| Random full-strength motor commands, 2 s, 12 runs | No leg ever closer than 0.33 mm to another segment's body |
| Random leg commands, 2 s, with v1 collisions removed but v1 tolerance | Identical motion to v1 |

Random flailing is chaotic: even changing the tolerance from 1e-10 to 1e-9 moves
the body by several millimetres after two seconds. Versions and backends are
therefore compared by checks like these and over short horizons, never by long
trajectories.

## Model v3

v3 is v2 with one change: the physics timestep is 0.149 ms (exactly 20 ms /
134) instead of 0.1 ms, so one 20 ms action takes 134 physics steps instead of
200. The agents still act every 20 ms; only the substeps inside an action
change. It was adopted because, after the GPU backend replays an
action's physics steps as a CUDA graph, the GPU's work per physics step limits
training speed.

**Why this timestep.** The contacts respond in 0.3 ms, and MuJoCo never lets a
constraint respond faster than twice the timestep. Up to 0.15 ms the contacts
therefore keep exactly the stiffness of v2; above it they would become softer.
0.149 ms is the longest timestep below that bound that divides 20 ms into a
whole number of steps.

**What was compared** (`benchmarks/physics_options.py`, a T4 GPU, Newton,
1,024 worlds for speed and 64 for the checks, every option changed in memory
only). The time of one action, replayed as a CUDA graph, against v2:

| Option | Speed | Outcome |
| --- | ---: | --- |
| v2 as it is (0.1 ms) | 1.00x (2.55 s) | Reference |
| Conjugate-gradient solver | 0.31x | Slower; reaches its 80-iteration limit |
| Solver tolerance 1e-5 or 1e-4 | 0.94x to 0.96x | No gain: Newton needs only 9 to 11 iterations |
| Line search limited to 10 iterations | 0.91x | No gain |
| Pyramidal friction cones | 1.34x | A world became non-finite while settling |
| Timestep 0.125 ms | 1.10x | Sound |
| **Timestep 0.149 ms (v3)** | **1.32x** | **Sound; contacts unchanged** |
| Timestep 0.2 ms | 1.74x | Softer contacts: deeper penetration, larger limit excess |

**Validation.** Under random full-strength leg commands (64 worlds, 2 s), v3's
deepest penetration was 0.097 mm against v2's 0.096 mm, and its largest
joint-limit excess 4.55° against 4.25° (v1's validation: 0.176 mm and 4.23°);
the mean height and speed of the body stayed within 3% of v2's, about the
run-to-run spread. The automated tests in `tests/models/test_model_v3.py`
check that only the timestep differs from v2, that no contact is softened,
that settling for 2 s with motors off ends with the same contacts and body
positions within 0.013 mm of v2's (deepest penetration 0.020 mm against
0.018 mm), and that no leg passes through a body under random commands.

## Model v4

v4 is a longer body whose legs have a fourth joint, with faster physics. Its
contacts and friction, head, body width and height, spine joints and motors,
and the joints its legs share with v3 (ranges, damping, and maximum torques)
are exactly v3's, and so are the integrator and the solver. It changes:

| Property | v3 | v4 |
| --- | ---: | ---: |
| Body units, each with a pair of legs | 8 | 12 |
| Distance between unit centres | 3.8 mm | 4.3 mm |
| Trunk unit length / mass | 4.4 mm / 73 mg | 4.9 mm / 81.3 mg |
| Body length without legs | 33.8 mm | 54.8 mm |
| Leg links | upper 4.6 mm, lower 5.8 mm | upper 4.6 mm, lower 3.2 mm, tarsus 2.6 mm |
| Leg joints | sweep, lift, knee | sweep, lift, knee, ankle |
| Coordinates (position / velocity) | 69 / 68 | 125 / 124 |
| Motors | 55 | 107 |
| Whole model | 643.4 mg | 1047.9 mg |
| Physics timestep | 0.149 ms, 134 steps per action | 0.5 ms, 40 steps per action |
| Contact / joint-limit response time | 0.3 / 0.8 ms | 1 / 1 ms |
| Leg joints' armature | 2e-11 (sweep, lift), 2e-12 (knee) kg·m² | each leg joint's transient 2 ms ([Physics of v4](#physics-of-v4)) |

The trunk unit is longer by the same amount as the spacing, so neighbouring
units still overlap by 0.6 mm, and its mass grows with its length. The spacing
and the number of units are engineering choices, still short of the reference
species' 19 to 21 leg pairs.

**Legs.** v3's lower link is split into a lower link and a **tarsus**, joined
by the **ankle**, a hinge parallel to the knee. The leg keeps its total length,
1.30 body widths, and its mass: 1.50 mg upper, 0.55 mg lower, 0.45 mg tarsus,
and the 0.15 mg foot. The lower link is 0.22 mm in radius, as before, and the
tarsus 0.18 mm.

| Joint | Range | Maximum torque | Damping |
| --- | ---: | ---: | ---: |
| Ankle | −30° to +40° | ±5 µN·m | 1e-7 N·m·s/rad |

The ankle damps like the knee, and its maximum torque is half the knee's, as
it moves only the tarsus; its range and torque are engineering choices. In
the model's pose the leg arches down and out: the upper link is horizontal,
the lower link 35° below horizontal, and the tarsus 60° below. The lift range
is therefore 25° below to 40° above horizontal, the knee bends between 15° and
85°, and the ankle between −5° and 65°. The body's centres start 5.06 mm above
the floor, with the feet just clear of it.

**Motors.** Every unit owns eight leg motors, in leg action order the left
leg's sweep, lift, knee, and ankle, then the right leg's, and every unit after
the head the yaw motor joining it to the unit ahead: 8 × 12 + 11 = 107. The
model's file is otherwise built like v3's, with the same naming, ownership and
shape categories, and eleven visual membranes.

### Physics of v4

v4's physics were revised on 2026-10-15, before any training had used it, so
that the GPU, whose work per physics step limits training, takes 40 physics
steps per action instead of 134.

**Why v3 needed 134 steps.** MuJoCo makes contacts and joint limits soft
constraints: each responds like a damped spring whose response time
(`solref`) the model sets, and MuJoCo never lets a response be faster than
twice the timestep. How far a constraint gives under a load is the response
time squared times the acceleration the load would cause without it
([MuJoCo's documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)).
The legs' outer links weigh 0.15 to 1.5 mg, so a motor's full torque would
accelerate them by up to 1.2 million rad/s² (the ankle): their limits and
contacts held only with the very short responses of v1 to v3, 0.3 ms for
contacts and 0.8 ms for limits, and so with a timestep under 0.15 ms.

**What changed.** Every leg joint gets **armature**, an inertia added to the
joint, as a robot's motor rotor adds to its joint: enough that the joint's
inertia, armature included, over its damping is 2 ms. Damping, not inertia,
governs the legs: a leg's top speed is its torque over its damping (40 to
100 rad/s), and the armature leaves it unchanged; the leg reaches it in 2 ms
instead of 0.04 to 0.5 ms, a tenth of an action. The accelerations the limits
and contacts must stop fall up to twelvefold, so they hold with slower
responses: **1 ms** for both, twice the new **0.5 ms timestep**.

| Leg joint | Inertia without armature | Armature | Transient before / after |
| --- | ---: | ---: | ---: |
| Sweep | 7.0e-11 kg·m² | 3.3e-10 kg·m² | 0.45 / 2.0 ms |
| Lift | 7.7e-11 kg·m² | 3.2e-10 kg·m² | 0.48 / 2.0 ms |
| Knee | 1.6e-11 kg·m² | 1.8e-10 kg·m² | 0.18 / 2.0 ms |
| Ankle | 2.1e-12 kg·m² | 2.0e-10 kg·m² | 0.04 / 2.0 ms |

Small bodies need short timesteps: a fruit fly trained with reinforcement
learning in MuJoCo walks on a 0.2 ms timestep (Vaxenburg et al., flybody),
legged robots of several kilograms on 4 to 5 ms (Rudin et al., CoRL 2021; the
Go1 of MuJoCo Playground). Scaled by the square root of size, as body motion
under gravity is, both point to about 0.5 to 0.7 ms for legs of this length.

**What was compared** (CPU MuJoCo for the checks, an RTX 5090 with MuJoCo
Warp's Newton solver at 2,048 worlds for the speed; each variant with
contacts and limits at twice its timestep):

| Variant | Steps per action | Time per action on the GPU | Joint-limit excess, random commands | Feet's creep under a sideways push of 20% of the weight, 0.2 s |
| --- | ---: | ---: | ---: | ---: |
| v4 before the revision | 134 | 1,083 ms | 6.2° | 101 µm |
| No armature, 0.5 ms | 40 | | 16.8° | 320 µm |
| No armature, 1 ms | 20 | | 46° | 440 µm |
| **Armature 2 ms, 0.5 ms (v4)** | **40** | **337 ms** | **3.1°** | **81 µm** |
| Armature 2 ms, 0.8 ms | 25 | 228 ms | 5.1° | 127 µm |
| Armature 3 ms, 1 ms | 20 | 187 ms, but a world became non-finite | 6.1° | 120 µm |

A scripted walk (feet driven along sines, a wave over five units) moved at
37.7 mm/s, 7.4 mm high, before the revision and at 37.1 mm/s, 7.5 mm high,
after it. 0.5 ms was chosen over 0.8 ms for its margin from the point where
the GPU's 32-bit arithmetic lost a world. Not chosen: impedances close to 1
instead of armature (the solver needed three times the iterations and the
feet lost their grip), MuJoCo's `noslip` solver (MuJoCo Warp does not
implement it), and stiffer friction cones (`impratio` 10 let the feet slide
further).

**Validation.** The checks of [Validation](#validation), on v4 (CPU MuJoCo,
the model's timestep unless stated):

| Check | Result |
| --- | --- |
| Settling with every motor off, 2 s | The first body touches the floor after 38.5 ms (before the revision 36 ms); all 12 bodies and 19 of the 24 feet end on the floor; 0.055 mm deepest penetration; no solver warning |
| Every lift motor pressing down, 1 s | At 40% or 60% the lift joints reach their limit and the body stands on all 24 feet with no body contact, centres 6.41 mm high, 0.042 mm deep; at 20% it stands 5.36 mm high on 22 feet |
| Random full-strength motor commands, 2 s, 3 runs | 0.43 to 0.52 mm deepest penetration, in brief impacts (before the revision 0.16 to 0.18 mm); joint limits exceeded by at most 3.2° (before 6.4°); no solver warning |
| Leg closest to another segment's body under the random commands | 0.54 mm |
| Legs at the corners of their ranges with the spine straight or at its hard limits, 192 poses | No leg or body closer than 0.56 mm to another segment's body |
| Half timestep | Settling ends within 0.04 mm of the full timestep, standing at the same heights; joint-limit excess at most 2.9° |
| GPU (RTX 5090, MuJoCo Warp, 32-bit), 2,048 worlds, 1,000 actions of random commands | Every world finite, no memory overflow, the solver never at its iteration limit; 0.54 mm deepest penetration, joint limits exceeded by at most 3.4°; 337 ms per action |

As in v1 to v3, the body cannot hold itself up with its motors off. The
automated tests in `tests/models/test_model_v4.py` check that v3's physics,
joints, and torques carry over except the timestep, the response times, and
the leg armature, which they check too; that the body sinks with its motors
off and stands on its lift motors; and that no leg passes through a body
under random commands.

## Model v5

v5 is the body of a real species, *Scolopendra heros*, the giant desert
centipede of North America: a head without legs and 21 leg-bearing units,
at the length and mass its animals were measured at. Of the *Scolopendra*
species whose walking has been studied, it is the largest: Anderson,
Shultz and Jayne (1995) measured its length, its mass and its gait at
three speeds. Where they measured nothing, v5 follows *S. polymorpha*, a
close relative of the same deserts and the best-studied species. Its
physics are v4's, scaled to the new size.

### Two lengths

v5 comes in two files built from the same units, density and armatures by
one generator (`archive/model_v5_heros/make_model_v5.py` in the main
folder), with the same simulation settings:

| File | Units | Mass | Degrees of freedom | Motors | Use |
| --- | ---: | ---: | ---: | ---: | --- |
| `assembly_v5_full.xml` | 21, the whole animal | 11.56 g | 214 | 188 | confirming a design, and the final training |
| `assembly_v5_half.xml` | 10, the first ten | 5.71 g | 104 | 89 | the tester: screening a design quickly |

The half is the whole body cut behind its tenth unit: every body, shape,
joint and motor it has is the full body's, by name and value. It is half
the length because that is about the shortest body that still holds one
whole wave of the legs, as the scripted walks used to check v5 run (a
wave over 9 units), and *S. heros* bends its body in waves of 5 to 11
segments (Anderson et al. 1995). It costs a fourth of the full body's
work on the GPU (below). What it cannot show is what only a long body
has: 21 segments cooperating, and the whole body's wave.

### The body

| Property | v4 | v5 | Source |
| --- | ---: | ---: | --- |
| Units with legs | 12, the head's included | 21, behind a head without legs | *S. heros* has 21 leg-bearing segments (Anderson et al. 1995); neither the head nor the forcipular segment behind it carries walking legs (Kenning et al. 2017) |
| Length, the head's tip to the rear's end | 54.8 mm | 150 mm | the animals' mean length, 15.2 cm (Anderson et al. 1995) |
| Distance between leg pairs | 4.3 mm | 6.61 mm | 152 mm over 23 units, the head and the forcipular segment counted as one unit each (an estimate) |
| Body width / height | 8.0 / 3.4 mm | 14.1 / 7.8 mm | *S. polymorpha*'s width over its length, 0.8 over 8.6 cm (Pierce et al. 2026), times 152 mm; the height 0.55 of the width, as in Anderson et al.'s cross-section of *S. heros* (their Fig. 2A) |
| Leg length | 10.4 mm | 18.4 mm | 1.3 body widths, as in *S. polymorpha* (Pierce et al. 2026) |
| Leg thickness: femur, tibia, tarsus | 0.60, 0.44, 0.36 mm | 1.83, 1.41, 0.99 mm | 0.13, 0.10 and 0.07 body widths: measured on photographs of *S. polymorpha* with scale bars (Pierce et al. 2026, Figs 1A and 2A), 0.12 to 0.18 widths near the body, and on Anderson et al.'s drawing, 0.09 |
| Leg pose | the femur level, the tibia 35° and the tarsus 60° below the horizontal | the femur rising 30°, the tibia 75° and the tarsus 60° below the horizontal | between the photographs and the drawing: the knee near the top of the body, the foot about 0.75 body widths beyond its side, the belly half the body's height above the floor |
| Mass | 1047.9 mg | 11.56 g | the animals' mean, 11.56 g (Anderson et al. 1995) |
| Motors / position coordinates | 107 / 125 | 188 / 215 | |

Every unit and leg keeps v4's shapes, scaled along each axis. The mass has
one density throughout (687 kg/m³ of the convex shapes), so a part weighs
as its volume: a trunk unit's body 0.477 g, the legs 10% of the whole (v4:
6%). The units and their legs are all alike. In the animal the legs grow
longer toward the rear and the back plates alternate long and short
(Anderson et al. 1995; Pierce et al. 2026); v5 leaves this out, as the
model of Dionne, Giardina and Mahadevan (2026) does. The last pair is an
ordinary leg: in the animal it does not walk (Kenning et al. 2017), and
the agents may learn to hold it.

**The head** is part of the first unit's body (0.874 g with it): one
collision shape, v4's head shape, its front stretched 10 mm beyond the
unit's front end and narrowing to 0.85 of the width and 0.8 of the height
there (an estimate). It cannot turn on its own. The program finds its
outline and its tip, `head_tip`, on the first segment, as for v1 to v4.
That shape is drawn as two: a trunk unit like the others, carrying the
first legs, and the red head ahead of it, which has none. The antennae and
the forcipules, the venom claws under the head, are shown too. Each
antenna is 12 tapering pieces, two of *S. heros*' 24 to 26 articles each
(Shelley 2002), about a fifth of the body long (an estimate), reaching
forward and out and drooping toward its tip; each forcipule is a claw
curving forward and in to a dark point. Everything drawn and not the
collision shape weighs nothing and touches nothing: like the membranes,
these are visual shapes (category 5), which MuJoCo Warp leaves out of its
collision pairs.

**Joints and motors** are v4's: every unit owns eight leg motors and every
unit after the first the yaw motor joining it to the unit ahead, 8 × 21 +
20 = 188. The joint ranges are v4's around the new pose, but the knee's,
which bends 10° less, to +40°: folded at the corners of their ranges, and
3° past them as the soft limits allow, a foot then stays 1.05 mm clear of
its own body (at +50° it went 0.68 mm into it). Legs and bodies still do
not collide, as in v2 to v4.

### Physics of v5

v4's physics, scaled to the new size by dynamic similarity, as motion under
gravity scales (Froude): with lengths L times v4's and masses M times,
forces scale with M, torques and stiffnesses with M L, times with √L, and
damping, a torque per speed, with M L √L. L is the legs' length ratio, M a
unit's mass ratio (the whole body's mass over its units).

| Property | v4 | v5 | Scale |
| --- | ---: | ---: | --- |
| Leg length (L) | 10.4 mm | 18.4 mm | 1.767 |
| A unit's mass (M) | 87.3 mg | 550 mg | 6.304 |
| Maximum torques: sweep, lift, knee, ankle, spine yaw | 8, 18, 10, 5, 15 µN·m | 89, 201, 111, 56, 167 µN·m | M L = 11.14 |
| Spine stiffness: yaw, pitch | 2e-5, 1e-4 N·m/rad | 2.2e-4, 1.1e-3 N·m/rad | M L |
| Damping | | | M L √L = 14.81 |
| Physics timestep | 0.5 ms, 40 steps per action | 1.33 ms, 15 steps per action | twice √L (below) |
| A leg joint's transient (inertia over damping) | 2 ms | 2.66 ms | √L |
| Contacts' and joint limits' response | 1 ms | 2.67 ms | twice the timestep, as in v4 |

So the motors keep v4's strength for the body's weight, and a leg's top
speed, torque over damping, is v4's over √L, 30 to 75 rad/s. Each leg
joint's armature makes its transient 2.66 ms, given the legs' own inertia
at the model's pose; the spine's armature is v4's times M L². Friction is
v4's; the simulation settings below were revised for speed.

**Simulation settings, revised for speed (2026-10-19).** v5 first had the
scaled timestep (0.667 ms, 30 steps per action), v4's contacts and
tolerance, and the head as the root of its body tree. On a training GPU
(MuJoCo Warp's Newton solver in 32-bit arithmetic) a run of 256 update
cycles then took about 7.5 hours on an RTX 4090, and about one world-step
in 800 became non-finite early in training. A speed study, on Warp's CPU
device (which runs the GPU's code) and on GPUs, found that the solver does
84% of the work, most of it factorizing for every world a dense matrix as
wide as the body's 214 degrees of freedom, and set four things:

- **The free joint on the middle unit** (`segment_10`; the half's
  `segment_05`). MuJoCo describes the body as a tree of joints that starts
  at the body with the free joint. From the head, the chain of 21 units
  made the solver's matrix so ill-conditioned when the body lay flat that
  32-bit rounding broke its factorization: the non-finite worlds (MuJoCo
  Warp's open issue #1415). From the middle the longest chain is half as
  long: no failure in 128 flat landings without any repair, against 19,
  and none on the GPU. Every hinge, axis, limit and motor is the same; on
  the links between the middle and the head, the yaw and pitch hinges are
  read from the other side (moved into the unit ahead, in the reverse
  order, their axes negated so that every angle keeps its sign). In the
  air both trees accelerate alike to 1e-11; with contacts they differ by
  the one term MuJoCo omits for contacts, the Jacobian's rate times the
  velocity, which depends on the root (about 0.1% of a contact force).
- **A 1.33 ms timestep**, 15 steps per action, twice the scaled one, with
  contacts and limits responding in twice it. The work per action falls
  with the steps; the price is softer impacts (below).
- **Legs touch only the legs of their own side up to three units away**,
  through explicit contact pairs with the legs' own contact values, and
  the floor. In a census of v5's leg contacts, 0.02% were four units
  apart, none farther and none across the body.
- **The solver's tolerance 1e-4** instead of 1e-6: the same walk, and
  about 10% less work.

On an RTX 5090, 4,352 worlds acting like a fresh policy:

| Model | Work per action | Factorizations broken by rounding |
| --- | ---: | ---: |
| v5 as it first was | 1,936 ms | 6,427 |
| The full body | 619 ms | none |
| The half | 143 ms | none |

The price of the longer timestep, in CPU MuJoCo 3.16.0
(`archive/model_v5_heros/timestep_check.py` and `validate.py`):

| | v5 as it first was | The full body |
| --- | ---: | ---: |
| A sine walk (3.5 Hz, wave over 9) | 84.7 mm/s | 64.6 mm/s |
| Deepest penetration, random commands | 0.68 to 0.74 mm | 1.32 to 1.56 mm |
| Joint-limit excess, random commands | 2.7 to 3.1° | 6.5 to 7.0° |

A body scaled up in proportion would not have helped: every motion slows
by the square root of the size, so the timestep could grow exactly as much
as each stride lengthens, and the steps per stride stay the same (dynamic
similarity, as above).

**The GPU's reserves.** MuJoCo Warp reserves room for each world's contacts
and constraint rows once, and the program stops rather than drop any
([environment.md](environment.md)). A full-body world needs at most 126
contacts and 756 constraint rows, when the whole body lies on the floor
(each contact with the floor six rows of its friction cone), 252 in a walk
(CPU MuJoCo 3.16.0, `archive/model_v5_heros/contact_budget.py`); on the GPU
771 rows were seen, so configurations of the full body set
`[environment.simulation] constraints_per_world` to 1,024. The half needs
at most 60 contacts and 360 rows (369 on the GPU), within the default 512.

### Validation of v5

The checks of [Validation](#validation), on both lengths (CPU MuJoCo
3.16.0, the model's timestep unless stated; `archive/model_v5_heros/validate.py`
in the main folder's archive):

| Check | The full body | The half |
| --- | --- | --- |
| Settling with every motor off, 2 s | The first body touches the floor after 52 ms; all 21 bodies and 42 feet end on the floor, centres 3.89 mm high; 0.072 mm deepest penetration; no solver warning | The same with its 10 bodies and 20 feet: centres 3.88 mm high, 0.071 mm deep |
| Every lift motor pressing down, 1 s | At 20%, 40% and 60% the body stands on all 42 feet with no body contact, centres 9.7 to 10.3 mm high, 0.10 to 0.24 mm deep | At 40% and 60% on all 20 feet, centres 9.9 to 10.0 mm high, 0.14 to 0.22 mm deep; at 20% its first unit, which carries the head on two legs, touches the floor |
| Random full-strength motor commands, 2 s, 3 runs | 1.32 to 1.56 mm deepest penetration, in brief impacts; joint limits exceeded by at most 7.0°; no solver warning | 1.28 to 1.44 mm; at most 6.7°; no warning |
| Leg closest to a body under the random commands | 0.09 mm | 0.06 mm |
| Legs at the corners of their ranges with the spine straight or at its hard limits, 192 poses | Nothing closer than 0.135 mm to the body of its own or any segment within two of it. Bent to every limit at once, the 21 units coil onto themselves and the tail's legs reach 2.4 mm into the head: v5 is long enough to coil, and its bodies collide neither with one another nor with other segments' legs | Nothing closer than 0.135 mm; too short to coil |
| Half timestep | Settling ends within 0.06 mm of the model's timestep, standing within 0.07 mm of its heights; penetration 1.14 to 1.24 mm and joint-limit excess at most 5.7° under random commands | Settling within 0.07 mm; penetration 1.09 to 1.38 mm, joint-limit excess at most 5.4° |

As in v1 to v4, the body cannot hold itself up with its motors off. The
GPU backend reads v5's state as CPU MuJoCo computes it
(`tests/environment/simulation/test_gpu_backend.py`, on the half), and the
main environment runs both lengths on both backends without change. The
automated tests in `tests/models/test_model_v5.py` check the full body's
size and mass, that every torque and damping is v4's times one scale and
the transients its square root, the timestep and the contacts' and
limits' response, the root on the middle unit, the legs' contacts and the
tolerance in both lengths, that the half's parts are the full body's, that
the body sinks with its motors off and stands on its lift motors, and that
no leg passes through a body under random commands.

### Sources for v5

- Anderson, B. D., Shultz, J. W. and Jayne, B. C. (1995). Axial kinematics
  and muscle activity during terrestrial locomotion of the centipede
  *Scolopendra heros*. *Journal of Experimental Biology* 198, 1185–1195.
- Pierce, C. J., Soto, D., Erickson, E., Diaz, K., Iaschi, M., Lay, A. and
  Goldman, D. I. (2026). Legged locomotion in lattices: centipede
  traversal of obstacle-rich environments. *Annals of the New York Academy
  of Sciences* 1555, e70187.
- Kenning, M., Müller, C. H. G. and Sombke, A. (2017). The ultimate legs of
  Chilopoda (Myriapoda): a review on their morphological disparity and
  functional variability. *PeerJ* 5, e4023.
- Dionne, A., Giardina, F. and Mahadevan, L. (2026). Embodied intelligence
  solves the centipede's dilemma. arXiv:2603.08409.
- Shelley, R. M. (2002). A synopsis of the North American centipedes of the
  order Scolopendromorpha (Chilopoda). *Virginia Museum of Natural History
  Memoir* 5; *S. heros*' antennae as Myriatrix summarises it
  (myriatrix.myspecies.info, *Scolopendra heros*).
