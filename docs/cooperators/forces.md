# Forces

**Status (2026-10-14): checked; no model change proposed.** Once the reward
prices forces ([reward.md](reward.md)), the forces the model can make must
be believable. This file checks model v4's against biology
([model.md](../model.md#model-v4)). From cycle 18 (2026-10-18) the body is model v5, *Scolopendra heros*,
whose forces are v4's scaled by dynamic similarity, so every comparison
below holds for it in proportion ([model.md](../model.md#model-v5)).

## The body's weight

| Quantity | v4 | Check |
| --- | ---: | --- |
| Whole model | 1.048 g, weight 10.3 mN | A length-to-mass formula for this group predicts about 0.55 g for v3's 33.8 mm body ([model.md](../model.md#mass-and-inertia)); scaled to v4's 54.8 mm at the same width, about 0.9 g. Within about 20% |
| One trunk unit with its legs | 86.6 mg, weight 0.85 mN | |
| Legs | 6% of a unit's mass | No measurement found |

The mass is realistic for the body's size. Rescaling it (to 1 kg, or to
exactly 1 g) is not proposed: a kilogram centipede would live in another
physical regime, where inertia matters and muscles are relatively weaker,
since a muscle's force grows with its cross-section (length squared) and
weight with volume (length cubed). Easy ratios come from units instead:
forces in body weights, and energy as the cost of transport, energy per
unit of weight and distance, the pure number biologists use to compare
animals of every size (Tucker, 1975; Taylor, Heglund and Maloiy, 1982).
These carry over unchanged to the 21-segment model if its units keep the
same mass and strength.

## What the legs can do

In v4's pose a foot is about 8.5 mm from its shoulder, horizontally (8.52
mm, as the simulation reads it), so a shoulder torque of 1 µN·m makes about
0.12 mN at the foot, and the sweep's strongest push, which prices pain, is
0.94 mN.

| Joint | Maximum torque | At the foot | In body terms |
| --- | ---: | ---: | --- |
| Lift (shoulder) | 18 µN·m | about 2.1 mN up or down | 2.5 unit weights per leg; all 24 legs together about 5 body weights |
| Sweep (shoulder) | 8 µN·m | about 0.9 mN forward or back | About one unit weight per leg; all together about 2 body weights |
| Knee | 10 µN·m | about 2.5 mN | Not the limit |
| Ankle | 5 µN·m | about 3.8 mN | Not the limit |

- **Standing takes 20% of the lift motors' strength on all 24 feet,** as
  v4's validation measured; walking with about half the feet down takes
  about 40%. Muscles in routine locomotion work well below their maximum,
  and the reserve grows as animals get smaller: the peak limb force in fast
  running is 6.9 body weights in a 7 kg turkey, 2.8 in a horse and about
  0.8 in an elephant (Biewener, J. Exp. Biol. 2005). A 1 g animal likely has
  more reserve than ours, so v4's legs are plausible, perhaps on the weak
  side. No measured leg force of a centipede was found.
- **The sweep is weaker than the lift,** less than half at the foot. The
  strengths are engineering choices, but they follow the order measured in
  a walking insect (below). With force priced as each motor's activation
  squared ([reward.md](reward.md)), the maximum torques act as muscle
  sizes: a weaker motor pays more for the same torque.

## Compared with a walking insect of the same weight

No leg forces of a centipede were found, but the stick insect *Carausius
morosus*, 0.8 g, close to our 1.05 g, has measured joint torques in free
walking (Dallmann, Dürr and Schmitz, Proc. R. Soc. B 2016). Its leg joints
match ours: thorax–coxa is our sweep, coxa–trochanter our lift, femur–tibia
our knee.

| Stick insect, walking | Value |
| --- | --- |
| Peak lift (depression) torque, middle and hind legs | 54 to 58 µN·m |
| Peak sweep torque, hind leg | 16 µN·m |
| Lift : sweep | about 3.5 : 1 |
| Peak vertical force per leg | about half the body weight on a middle or hind leg, a fifth on a front leg |

- **The lift is the strongest joint, in insects and in v4.** The insect's
  propulsion comes from pressing the legs down at the lift joint, not from
  sweeping them back; the sweep and knee torques mostly keep the leg from
  collapsing under the load. v4's lift : sweep strengths, 18 : 8, about
  2.2 : 1, follow the same order.
- **The insect's torques are larger because it has six long legs, not
  twelve pairs of short ones.** Torque is force times reach: each of its
  legs carries up to half its weight about 11 mm out. Each of v4's 24 legs
  carries a twenty-fourth to a twelfth of the weight about 8.5 mm out, so it
  needs about 4 to 7 µN·m of lift, 20 to 40% of its strength.
- **Real joints are asymmetric; v4's motors are not.** In insects, the
  muscle that pushes the leg against the ground is the stronger of each
  pair: the lift joint presses down through all of stance, and the knee's
  flexor has twice the extensor's lever arm (Guschlbauer, dissertation,
  Cologne 2009). Every v4 motor is equally strong both ways. No centipede
  data say by how much the two directions differ, so this stays a known
  difference, not a change.

So the engineering strengths of v4 have the insect's order and a
plausible size; only their exact ratios and their symmetry are unmeasured.

## Speed limits

Each joint's damping turns its motor into a linear force–velocity curve:
full torque at rest, none at the speed where damping absorbs it all, as a
muscle's force falls with its shortening speed (Hill's relation, here
straight).

| Joint | Top speed (maximum torque ÷ damping) |
| --- | ---: |
| Sweep | 40 rad/s: a full 80° sweep in about 35 ms |
| Lift | 90 rad/s |
| Knee | 100 rad/s |
| Ankle | 50 rad/s |

These are well above a walking stride, so they bound only the fastest
movements. The motors' work includes what damping dissipates, so fast
movements cost energy.

## The spine and the ground

- **The pitch springs can carry a hanging unit.** Holding one unit's weight
  4.3 mm out takes about 3.7 µN·m, which the pitch spring gives at about 2°
  of bend; that is why followers could hang from their neighbours in the
  oscillators. Energy shared through α prices it: the neighbours pay the
  force, and the hanger with them.
- **Friction 0.8 for feet and body alike.** An engineering value; real feet
  grip with claws, and the belly's friction was not measured. Dragging the
  whole body on its belly takes about 8 mN of pushing, a third of all the
  sweeps' strength. Whether a raised walk is cheaper than a drag is checked
  before training ([reward.md](reward.md)).
- **Inertia hardly matters.** For a body this small, friction outweighs
  momentum by far (oscillators, 2026-10-12): the walk is close to
  quasi-static, as for a slow small animal.

## Verdict

v4's forces are plausible for a centipede of about a gram. Its joints'
strengths follow a walking insect's order (lift strongest, sweep weaker);
unmeasured remain their exact ratios, the asymmetry of real joints, and the
belly's friction. Without data to replace
them, the model stays as it is; a physical change would be a new model
version, agreed with the user.
