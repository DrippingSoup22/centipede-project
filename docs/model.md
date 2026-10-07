# Model

The model is the centipede's body: MuJoCo XML files describing its shape,
joints, motors, masses, and contacts. It contains no program logic and is not
part of the pipeline; the configuration names the file, and the physics
simulation loads it (see the [architecture](architecture.md)).

## Status

The current model is **v3**, used for training. It is v2 with a longer physics
timestep (see [Model v3](#model-v3)). v2 has exactly the body and physics of the
original **v1 flat-ground baseline**. v2 and v1 are kept unchanged for
reference. All three are frozen: training work may change observations,
rewards, targets, and learning, but never a model. A physical change requires a
new, separately named version with its own validation.

| File | Purpose | SHA-256 |
| --- | --- | --- |
| [`models/assembly_v3.xml`](../models/assembly_v3.xml) | **v3**, the complete centipede used for training | `8a693cc5e15e4a468a6cafb293f67be7fab6460f7b76f0ca948546e59cb336fe` |
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

- Eight body units: one head and seven trunk units.
- Sixteen two-link legs, one left/right pair on every unit.
- Each leg has three hinges: shoulder sweep, shoulder lift, and knee.
- Seven flexible connections between units, each with a yaw and a pitch hinge;
  rolling between units is locked.
- A free root, so the whole body can move and rotate in 3D.
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
of motors in the file. How motors become agent actions is described in
[environment.md](environment.md#actions-and-timing).

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
change. It was adopted on 2026-10-07 because, after the GPU backend replays an
action's physics steps as a CUDA graph, the GPU's work per physics step limits
training speed.

**Why this timestep.** The contacts respond in 0.3 ms, and MuJoCo never lets a
constraint respond faster than twice the timestep. Up to 0.15 ms the contacts
therefore keep exactly the stiffness of v2; above it they would become softer.
0.149 ms is the longest timestep below that bound that divides 20 ms into a
whole number of steps.

**What was compared** (`benchmarks/physics_options.py`, Kaggle T4, Newton,
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

## Possible future versions

These are outside v1 and would each be a new model version:

- A fully passive spine with no yaw motors.
- A distinct tail unit.
- Antennae, forcipules, extra leg sections, soft feet, or adhesion.
- Deformable membranes between units.
- Rough terrain, obstacles, or slopes.
- A realistic number of segments.

## References

- Anderson, Shultz, and Jayne (1995), body bending in *S. heros*:
  [paper](https://jwshultz.weebly.com/uploads/4/6/2/2/46222147/j_exp_biol-1995-anderson-1185-95.pdf)
- Diaz et al. (2023), leg and body motion in *S. polymorpha*:
  [DOI](https://doi.org/10.1242/jeb.244688)
- Pierce et al. (2026), body width and leg length in *S. polymorpha*:
  [authors' PDF](https://crablab.gatech.edu/pages/publications/pdf/2026_CPierce.pdf)
- Sohlstrom et al. (2018), arthropod mass from body size:
  [DOI](https://doi.org/10.1002/ece3.4702)
- Nadein et al. (2025), arthropod joint structure:
  [DOI](https://doi.org/10.1007/s00435-025-00708-4)
- Gravish et al. (2023), foot traction on rough surfaces:
  [DOI](https://doi.org/10.1242/jeb.245261)
- MuJoCo contact parameters:
  [documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)
