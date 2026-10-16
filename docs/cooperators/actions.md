# Actions

**Status (2026-10-14): agreed with the user and built; from cycle 6
(2026-10-16) the spine motors receive 0.35 of their command
([The spine's scale](#the-spines-scale-cycle-6)); from cycle 8 the actions
explore with state-dependent noise, from cycle 9 its spread on a fixed
schedule ([How actions explore](#how-actions-explore)).**

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

## The vector (agreed)

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

**The spine's scale (cycle 6).** On v4 a spine motor of 15 µN·m meets the
joint's limit spring, 5.2 µN·m at 15°, so any command above about 0.35
holds the joint at its limit: two thirds of the command's range did the
same thing, and runs 09 and 10 learned to pin their spines there. Each
spine motor now receives `spine_action_scale` (0.35) times its command,
so that the whole range −1 to 1 spans the joint's. The action itself stays
the policy's, unfiltered and unclipped (the user's rule): only the motor's
gain changes, as a weaker motor would. Energy prices the command the motor
receives, and each segment observes the action it sent.

**The decision rate: every 20 ms**, 50 a second, as main and as legged_gym
(Rudin et al.'s 50 Hz). A real centipede steps at about 6 Hz, so a
decision comes about eight times in each step. Physics costs the same per
simulated second at any rate; a faster rate would only add samples and
learning time.

## How actions explore

A choice of learning, made with the agents' settings, not part of the
vector.

**Until cycle 7: noise drawn through time.** White noise was planned (the
user, 2026-10-14); pink noise (β = 1) ran from run 01 on, and from cycle 6
with a starting spread of 0.25 instead of 0.5, since at 0.5 the noise made
standing worth more than walking
([reward.md](reward.md#cycle-6-distance-on-the-trail-the-squeeze-its-price-the-users-design)).

**What runs 13 and 14 showed: the noise was the gait.** In training their
targets rose tenfold, but without noise the body stood, its legs stepping
0.2 to 0.4 times a second. Driven by their cycle-128 actors with the pink
noise at a part of its learned spread, the body moved the more the larger
the noise (`analysis/exploration_check.py` in `../Training`, 4 worlds, mm/s):

| Pink noise | Run 13: closing / head speed | Run 14: closing / head speed |
| --- | --- | --- |
| None (the mean action) | 0.00 / 1.7 | 0.02 / 2.7 |
| A quarter | 0.17 / 3.3 | 0.19 / 5.7 |
| Half | 0.46 / 7.7 | 0.74 / 10.1 |
| All | 0.60 / 14.2 | 1.22 / 16.2 |

The mean cannot learn a step the noise made. The noise is drawn through
time, unrelated to the state, and the actor has no clock; PPO moves the
mean toward the actions that went well in each state, and averaged over the
noise's random steps that keeps the direction and loses the rhythm. The
same noise is jitter that the force price charges
([reward.md](reward.md#cycle-8-only-a-lasting-force-costs-the-users-idea)).

**From cycle 8: noise that depends on the state** (the user's agreement,
2026-10-16): generalized state-dependent exploration, gSDE (Raffin, Kober
and Stulp, CoRL 2021; [research.md](research.md#exploration)).

- **How.** Each action's noise is the actor's last hidden layer, scaled to
  unit length, times an exploration matrix that each world draws and keeps
  for `state_dependent_noise_steps` steps, then draws again. The matrix's
  entries have learned spreads, one per feature and action, starting at
  `initial_action_std` (0.25). For a given state the noise is normal, so
  PPO's probabilities stay exact (RL_lib's `GaussianPolicyNetwork` and
  `SquashedGaussianPolicy`).
- **Why.** While a matrix lasts, the same state gets the same noise: each
  draw acts as a slightly different policy, and what it tries is a
  behaviour the actor can learn, such as swinging a leg forward when it is
  far back. A draw that went well moves the mean toward it.
- **Unit-length features** (our choice; Raffin et al. use the features as
  they are): an action's spread is then 0.25 while the spreads are equal,
  whatever the features' size, and does not drift as the hidden layer
  learns.
- **A tanh actor** (`actor_activation`): a held matrix adds the hold's mean
  direction of the features as a constant push. ReLU's features, never
  negative, hardly change direction from one state to the next, so held 32
  steps on run 13's states the constant part is 83% of the noise with its
  trained actors and 71% with fresh ReLU ones, against 52% with fresh tanh
  ones, as Raffin et al. and Stable-Baselines3's PPO use.
- **Held 4 steps, 80 ms.** In Raffin et al.'s PPO test (Walker2D) the
  interval mattered most: 4 scored best, then 8 and 16; 32 and longer
  explored too little; 1, unstructured noise, worse. On runs 13 and 14's
  actors (ReLU, so the held part is large) the same order:

| Run 13 / run 14, mm/s | Closing | Targets a minute | Line erased | Force cost, 0.2 s | 1 s |
| --- | --- | --- | --- | --- | --- |
| Their own pink noise | 0.60 / 1.22 | 0.9 / 2.7 | 2.3 / 3.9 | 18 / 27 | 6.5 / 9.9 |
| gSDE, held 1 step | 3.08 / 0.84 | 8.1 / 1.8 | 7.6 / 4.4 | 10 / 7 | 3.2 / 2.6 |
| gSDE, held 4 steps | 1.45 / 1.31 | 4.5 / 3.6 | 8.9 / 6.5 | 13 / 11 | 4.5 / 3.2 |
| gSDE, held 16 steps | 0.81 / 0.94 | 1.8 / 1.8 | 4.1 / 4.0 | 26 / 21 | 8.1 / 6.7 |
| gSDE, held 64 steps | 0.71 / 0.73 | 1.8 / 0.9 | 1.0 / 1.0 | 27 / 27 | 10.0 / 9.8 |

  Held 4 steps, both actors moved more than with their own pink noise
  (held 1 step, one did and one did not), at about half its force cost;
  held 16 or 64 steps, less, at its full cost, the held part acting as a
  lasting push.

**What it is in practice, and the risks.** Held 4 steps, most of a draw
is a push that stays for its 80 ms: 78% with fresh tanh features, against
44% held 64 steps (`analysis/gsde_feature_check.py` in `../Training`). So
the change is mainly noise redrawn every 80 ms instead of pink's slow
drifts, with a smaller part that follows the state; a draw is consistent
within a step's phase, about a sixth of a stride, not over a whole stride.
Longer holds follow the state more but explored worse, in Raffin et al.'s
test and in ours: the same trade-off. Whether the gait now moves into the
mean is the question runs 15 and 16 answer, each review comparing the
mean action with training. And the actor's change to tanh changes how it
learns as well as how it explores.

**What runs 15 and 16 showed: still the noise.** In training run 16 did
best of the line after run 04 (0.80 targets an episode at cycle 128, 23% in
time, the level rising), run 15 as run 13; but without noise both stood,
their legs stepping under 0.7 times a second. In the recordings (the
user's reading) the head pulls toward its target and the body cannot
follow: the legs move cautiously, one at a time, their feet held on the
ground. Two causes, besides the noise doing the walking:

- **Exploring does not pay one segment alone.** A step pays only when many
  segments step together: the head cannot drag eleven planted segments,
  and a follower stepping alone only strains its joints. Each agent judges
  its own stepping against the others' random behaviour, on which standing
  looks better: relative overgeneralization (Wei and Luke, JMLR 2016).
- **The noise shrinks before a gait is found.** Each agent learns its
  spread, and PPO took it from 0.25 to 0.19: noise costs energy and force
  and almost never pays, so exploring grew more cautious while it should
  search.

**From cycle 9: the spread on a fixed schedule** (the user's choice,
2026-10-16). The spreads are not learned: they start at 0.4 and shrink by
equal ratios to 0.05 at the last of 256 cycles (`action_std_schedule`
"log_linear"; 0.24 at cycle 64, 0.14 at 128, 0.085 at 192), as the PPO
paper set its spreads for its humanoid tasks (Schulman et al., 2017). The
body explores widely first, and the 1 s force average makes that noise
about a third as costly as in cycle 6, when a spread of 0.5 made standing
worth more than walking; and as the noise shrinks the policy itself must
carry the walk, so the training's own numbers show whether it does.
gSDE, the tanh actor and the forces stay as in cycle 8.

**Later, if needed** (the user, 2026-10-16): a cost curriculum, energy and
the force prices starting at a fraction and growing over training (as
Hwangbo et al.'s curriculum factor, Science Robotics 2019), decided from
what the schedule shows; and, only if nothing else works, noise shared
between segments, correlated by their distance and only within the
observation radius, so that what a segment cannot see never shares its
noise.

## Not part of the vector
- **Asymmetric joints**: real muscles push harder toward the ground than
  away from it; v4's motors are symmetric, a known difference without
  centipede data ([forces.md](forces.md)).
