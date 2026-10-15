# Model

The model is the centipede's body: MuJoCo XML files describing its shape,
joints, motors, masses, and contacts. It contains no program logic and is not
part of the pipeline; the configuration names the file, and the physics
simulation loads it (see the [architecture](architecture.md)).

## Status

The newest model is **v4**: twelve body units, and legs of three links with a
fourth joint, the ankle, with v3's torque motors, leg armature, and a 0.5 ms
physics timestep (see [Model v4](#model-v4)). **v3**, which the example configurations use, is v2
with a longer physics timestep (see [Model v3](#model-v3)). v2 has exactly the
body and physics of the original **v1 flat-ground baseline**. v2 and v1 are kept
unchanged for reference. All four are frozen: training work may change
observations, rewards, targets, and learning, but never a model. A physical
change requires a new, separately named version with its own validation. The
one exception: v4's physics were revised in place on 2026-10-15, before any
training had used it, by the user's decision
([Physics of v4](#physics-of-v4)).

| File | Purpose | SHA-256 |
| --- | --- | --- |
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
  joint of the head's body, `segment_00`, which must be a child of the world:
  turning it turns the whole centipede, which is how a reset varies the
  starting heading.
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
| MuJoCo version | 3.12.0 |
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
