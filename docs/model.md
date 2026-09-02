# Centipede model

## Status and canonical files

The eight-unit active-yaw assembly is frozen as the **v1 flat-ground physical
baseline**. Training work may change controllers, observations, rewards, and
targets, but must not silently change this model. A physical revision requires a
new version and new validation artifacts.

| Purpose | Canonical file |
| --- | --- |
| Loadable MuJoCo model | [`models/assembly.xml`](../models/assembly.xml) |
| Isolated trunk reference | [`models/segment.xml`](../models/segment.xml) |
| Isolated head reference | [`models/head.xml`](../models/head.xml) |

Current XML SHA-256:
`92143cf54b030856e436a1de6f4333c327ffb9a5f812a0e4a0b4c4aed107d50e`.

Superseded development material may be retained in the ignored local `archive/`
directory. It is not part of the repository or an active experiment input.

## General description

The model is a reduced centipede-like articulated body for cooperative
reinforcement learning in MuJoCo:

- Eight body units: one head and seven ordinary trunk units.
- Sixteen two-link legs: one left/right pair on every unit.
- One segment-agent will own each body unit's actuators.
- A free root lets the complete animal translate and rotate in 3D.
- Seven yaw/pitch spine connections make the body flexible; relative roll is
  locked.
- Each leg has shoulder sweep, shoulder lift, and knee hinges.
- The v1 model has 69 position coordinates, 68 velocity coordinates, and 55
  torque motors.

This is an engineering abstraction. It preserves the body chain, bilateral leg
pairs, principal leg motions, and lateral body bending while omitting many real
leg articles, antennae, forcipules, a specialized rear unit, deformable tissue,
and the biological number of leg-bearing segments.

## Overall body

The rigid body envelope is approximately **33.8 mm long, 8 mm wide, and 3.4 mm
high**, excluding legs. The eight-unit choice deliberately limits the initial
cooperative problem to eight agents; the biological reference used for local
proportions has 19 leg pairs.

| Property | v1 value |
| --- | ---: |
| Units / legs | 8 / 16 |
| Trunk body length | 4.4 mm |
| Body width / height | 8.0 / 3.4 mm |
| Unit center spacing | 3.8 mm |
| Straight rigid overlap | 0.6 mm |
| Head front extension | 2.8 mm |
| Approximate complete body envelope | 33.8 x 8.0 mm |

The bodies use convex rounded-rectangular meshes with an arched roof, flatter
belly, broad sides, and bevelled ends. The head retains the trunk's broad rear
interface and adds a rounded front. The last unit remains an ordinary trunk;
v1 has no distinct tail mechanics.

The width and leg-length ratio follow measurements of *Scolopendra polymorpha*.
Exact segment length, height, and head extension are engineering choices because
the available locomotion studies did not provide a complete 3D body scan.

## Segment and leg structure

Every unit carries two identical mirrored legs. Each leg has an upper link, a
lower link, and a spherical foot contact.

| Part | Dimension |
| --- | ---: |
| Upper leg length / radius | 4.6 / 0.30 mm |
| Lower leg length / radius | 5.8 / 0.22 mm |
| Total link length / body width | 1.30 |
| Foot radius | 0.32 mm |

Each shoulder is modeled by two colocated hinges. Sweep produces fore/aft motion
in plan view; lift moves the leg above and below the body plane. The knee is one
distal hinge aggregating several biological articulations.

### Leg joint limits and actuation

| Joint | Hard range | Torque limit | Damping | Armature |
| --- | ---: | ---: | ---: | ---: |
| Shoulder sweep | +/-40 degrees | +/-8 micro-N m | 2e-7 N m s/rad | 2e-11 kg m2 |
| Shoulder lift | -25 to +40 degrees | +/-18 micro-N m | 2e-7 N m s/rad | 2e-11 kg m2 |
| Knee | -20 to +50 degrees | +/-10 micro-N m | 1e-7 N m s/rad | 2e-12 kg m2 |

The upper link begins 10 degrees below horizontal, so the lift range corresponds
to an approximate physical elevation of -35 to +30 degrees. The two links begin
25 degrees apart, giving an effective bend envelope of about 5 to 75 degrees.

The +/-40-degree sweep limit is informed by measured whole-limb motion in
*S. polymorpha*. Suitable live-centipede 3D joint ranges were not found for lift
or the simplified knee, so those two limits remain explicit engineering
approximations. Real distal centipede joints are primarily uniaxial, supporting
the hinge abstraction but not its exact numerical limits.

## Spine and segment connections

Each unit after the head is directly parented to the preceding unit through two
hinges at the intersegment joint.

| Spine DOF | Hard range | Stiffness | Damping | Actuation |
| --- | ---: | ---: | ---: | --- |
| Yaw | +/-15 degrees | 2e-5 N m/rad | 3e-7 N m s/rad | +/-15 micro-N m motor |
| Pitch | +/-8 degrees | 1e-4 N m/rad | 5e-7 N m s/rad | Passive |

Both axes use zero as the neutral spring reference and 2e-11 kg m2 armature.
Yaw has weak passive centering in addition to its motor; pitch is entirely
passive. At the hard limits, the yaw and pitch springs contribute approximately
5.2 and 14.0 micro-N m, respectively.

The yaw limit is informed by adjacent-segment lateral flexion reported for
*Scolopendra heros*: mean locomotor amplitudes were roughly 4 to 8.5 degrees,
with individual traces approaching 15 degrees. The pitch range is a smaller
engineering approximation because comparable dorsoventral measurements were not
found.

Setting `spine.active_yaw` to false is supported as a future experimental
variant, but it changes the actuator set and is not the frozen v1 model.

## Mass and inertia

The complete model mass is **643.4 mg**.

| Part | Mass |
| --- | ---: |
| Head body | 90 mg |
| Each trunk body | 73 mg |
| Each upper leg | 1.50 mg |
| Each lower leg | 1.00 mg |
| Each foot | 0.15 mg |
| Head unit including legs | 95.3 mg |
| Ordinary unit including legs | 78.3 mg |

MuJoCo derives each body's inertia from its actual mesh and primitive geoms with
the explicit masses above. No manual inertia tensor is stored. Visual membranes
have zero mass and produce bit-identical body mass and inertia arrays when
toggled off.

A broad Scolopendromorpha length/width regression predicts approximately 0.55 g
for this envelope. That is only a taxonomic sanity check; no live-mass
measurement was found for the individual *S. polymorpha* used in the referenced
locomotion studies.

## Contacts and ground friction

All ordinary model geoms participate in collision. Leg-leg contacts remain
enabled so coordination is part of the learning problem. There are no exclusions
or coupled action limits added merely to prevent legs touching.

Ground and self-contact friction are stored separately:

| Contact | Sliding | Torsional | Rolling | Contact dimension |
| --- | ---: | ---: | ---: | ---: |
| Ground | 0.8 | 1e-4 m | 1e-5 m | 6 |
| Model-model | 0.8 | inactive | inactive | 3 |

The higher-priority floor supplies the exact ground triplet. Six-dimensional
ground contacts activate sliding, torsional, and rolling resistance; self-contact
retains the earlier sliding-only response. These are isotropic engineering
parameters for the modeled spherical feet, not measured centipede material
coefficients. Real arthropod traction also depends on roughness, compliance,
claws, hairs, and other omitted contact structures.

In the deterministic sensitivity check, a 0.1 s lateral force equal to 20% of
total weight produced 1.101 mm lateral head displacement at sliding friction 0.2
and 0.0177 mm at the selected 0.8 baseline. Both cases retained all 16 foot
contacts. This establishes useful initial traction, not biological fidelity or
walking ability.

## Numerical simulation

| Setting | v1 value |
| --- | ---: |
| MuJoCo | 3.12.0 |
| Integrator | `implicitfast` |
| Physics timestep | 0.1 ms |
| Gravity | 9.81 m/s2 |
| Contact response time | 0.3 ms |
| Joint-limit response time | 0.8 ms |
| Contact damping ratio | 1.0 |

The contact and limit time constants exceed MuJoCo's recommended minimum of two
physics steps. A 0.05 ms refinement changed the final per-segment pose by at most
0.00054 mm in stance, 0.00563 mm in the scripted bend, and 0.00054 mm after the
lateral perturbation. Peak bend yaw differed by 0.00215 degrees. The 0.1 ms step
is therefore retained.

The future policy interval is separate from the physics timestep. The environment
must hold each action for an agreed integer number of physics steps rather than
run neural inference at 10 kHz.

## Visual membrane

Seven dark ellipsoidal membrane geoms cover the intersegment seams. Their axial
size is derived from body overlap and the approved yaw/pitch limits; width and
height remain slightly inside the body envelope.

The membranes are visual only:

- Zero mass and no contribution to inertia.
- `contype=0` and `conaffinity=0`, so they create no contact forces.
- No extra body, joint, constraint, or dynamic degree of freedom.

Small openings can remain visible at strongly bent seams because each rigid
membrane follows the child unit instead of deforming between two bodies. This
cannot affect state-based flat-ground training. A later obstacle task must revisit
the collision surface if obstacles are small enough to enter a seam or physical
membrane contact becomes relevant.

## Control ownership

The head owns its six leg motors. Each other unit owns its six leg motors plus
the yaw motor connecting it to the unit ahead, giving 55 motors total. Pitch has
no action. All controls are normalized to [-1, 1]; motor gear values convert them
to the physical torque limits above.

Observation design, policy sharing, control frequency, and learning architecture
belong to [`control.md`](control.md). Rewards, targets, and episode behavior belong
to [`environment.md`](environment.md).

## Validation baseline

Before it was frozen, the model passed 28 regression tests plus Ruff lint and
formatting checks. That suite and its dependencies are now archived with the
model-development tooling; this section records the accepted result rather than
describing an active test suite.

| Check | Result |
| --- | --- |
| Supported stance, 2 s | 16 feet, 0 body contacts, 0.00399 mm peak penetration |
| Scripted bend, 2 s | 16 feet, 0 body contacts, 9.78 degrees yaw, no limit excess |
| Passive spine release | Alternating 8/4-degree yaw/pitch offsets return below 0.1 degrees within 0.25 s |
| Full random torque stress, 2 s | 0.176 mm peak penetration, 0.439 thickness ratio, 4.23-degree soft-limit excess, no solver warning |
| Hard-limit spine poses | 18 poses: 0 body-body, 0 body-leg, 10 with allowed leg-leg contact |
| Leg endpoint poses | 672 poses: 314 with leg-leg contact, 0 body-body, 0 body-leg |
| Membrane toggle | Identical mass, inertia, coordinates, motors, and initial contacts |
| Half-timestep comparison | Stance, bend, and perturb acceptance checks pass |

These diagnostics demonstrate numerical stability and mechanical capability.
They do not demonstrate learned walking or a natural wave-like gait.

## Reopening model development

Retired source material may exist in the ignored local archive, but it is not a
project dependency. If the physical model must change, create a separately named
development version, keep `models/assembly.xml` unchanged, and produce new
validation artifacts before accepting a replacement baseline.

## Known limitations and future variants

The following are outside the frozen v1 scope:

- A fully passive-yaw spine, evaluated as a separate model variant.
- A distinct rear unit or decorative rear cap.
- Antennae, forcipules, additional leg articles, compliant feet, or adhesion.
- Physical deformable membranes.
- Rough terrain, small obstacles, slopes, or non-planar contact validation.
- A biological number of segments and segment-specific leg morphology.

## Research basis

- Anderson, Shultz, and Jayne (1995), axial kinematics and intersegmental
  flexibility in *S. heros*: [paper](https://jwshultz.weebly.com/uploads/4/6/2/2/46222147/j_exp_biol-1995-anderson-1185-95.pdf)
- Diaz et al. (2023), 2D limb/body kinematics in *S. polymorpha*:
  [DOI](https://doi.org/10.1242/jeb.244688)
- Pierce et al. (2026), *S. polymorpha* body width and leg-length proportions:
  [authors' PDF](https://crablab.gatech.edu/pages/publications/pdf/2026_CPierce.pdf)
- Sohlstrom et al. (2018), arthropod live-mass regressions:
  [DOI](https://doi.org/10.1002/ece3.4702)
- Nadein et al. (2025), arthropod joint structure and centipede distal hinges:
  [DOI](https://doi.org/10.1007/s00435-025-00708-4)
- Gravish et al. (2023), animal foot-ground traction on rough surfaces:
  [DOI](https://doi.org/10.1242/jeb.245261)
- MuJoCo contact parameter mixing and solver guidance:
  [documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)
