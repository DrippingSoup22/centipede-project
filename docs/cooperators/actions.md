# Actions

**Status (2026-10-14): torque motors decided; the vector proposed.**

## What the constraints fix

Each segment commands only its own part of the body and chooses its action
on its own ([README.md](README.md)).

## Decided: torque motors

The user's decision (2026-10-14): main's v4 motors, each applying a
commanded share of its joint's maximum torque. The reward prices energy as
a muscle spends it ([reward.md](reward.md)), and a torque motor is the
closest to a muscle: it makes a force, and how the joint moves follows from
the body and the ground. A position actuator would instead drive each joint
toward an angle, hiding the force inside a controller the agents do not
choose.

A useful consequence: **the action is the activation.** A command of 0.4
is 40% of that motor's strength, so the force cost of energy is simply the
square of each command, and a critic that sees its neighbours' actions sees
their effort directly.

## Proposed: the vector

| | Each segment but the rear | The rear |
| --- | --- | --- |
| Leg motors | 8: left sweep, lift, knee, ankle, then the right leg's | The same 8 |
| Spine | 1: the yaw motor of the joint behind it | None (no joint behind it) |
| Size | 9 | 8 (the ninth padded, sent to no motor) |

Every number lies between −1 and 1, a Gaussian squashed by tanh, and goes
unchanged to its motor, held for one decision ([environment.md](../environment.md#actions-and-timing)).

**Why the spine is commanded.** In *Scolopendra heros* walking on a
treadmill at 0.5 to 1.5 body lengths per second, the body's lateral muscles
fire as they shorten, in waves travelling toward the tail with the bending:
the body bends itself, actively, at every speed (Anderson, Shultz and
Jayne, J. Exp. Biol. 1995), against the older view that the legs impose
the bending and the body only resists it (Manton). Turning toward targets
behind, from level 0.55 on ([task.md](task.md)), needs the body to bend
far, and energy prices every use of it, so the agents bend only when it is
worth it. The pitch joints stay passive: no motor in v4, and no task asks
the body to arch.

**Why the joint behind.** As on main: the head can then turn its neck
toward the target it alone sees, and each later segment sets how the
segment behind follows, seeing that bend in its neighbours' orientations
([environment.md](../environment.md#actions-and-timing)).

**The decision rate: every 20 ms**, 50 a second, as main and as legged_gym
(Rudin et al.'s 50 Hz). A real centipede steps at about 6 Hz, so a
decision comes about eight times in each step. Physics costs the same per
simulated second at any rate; a faster rate would only add samples and
learning time.

## Not part of the vector

- **How actions explore** (white or coloured noise, smoothing): a choice
  of learning, made with the agents' settings.
- **Asymmetric joints**: real muscles push harder toward the ground than
  away from it; v4's motors are symmetric, a known difference without
  centipede data ([forces.md](forces.md)).
