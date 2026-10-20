# The follow-line design

This folder holds the design of the `follow-line` workspace branch, one file
per part: what is decided and why. The other documents in `docs/` describe
the program this branch started from (`main` at ec907b5); where this folder
decides differently, this folder holds on this branch.

## The job

The body learns to walk by following a line on the ground (the user's plan,
2026-10-20):

1. **A straight line**, until a wave gait forms: every leg stepping, and
   neighbouring legs in a steady phase lag.
2. **Lines that curve**, gradually, so that the wave carries the body
   through turns.

The next job, on its own branch, is `hunt-target`: a leader draws the line
toward a target, and the body follows it.

## The question and its constraints

Can independent agents, each seeing only its neighbours, learn to walk
together like a centipede? The constraints are the user's (2026-10-14,
from the cooperators study):

1. **Independent agents.** Each unit has its own actor, critics and
   experience; nothing is shared between units.
2. **A local view.** When it acts, a unit sees only itself and its
   neighbours within its sight ([observations.md](observations.md)).
3. **General rules.** The reward says what to achieve, never how to walk.
4. **The measure of success.** A real centipede gait emerges from learning.

## Principles

Each is a lesson of the cooperators study (runs 01 to 28, in its branch's
`docs/cooperators/` and its Training's results):

- **The pairing rule.** A critic must see the causes of the reward it
  predicts; a reward and its critic are one decision.
- **The goal first.** Moving along the line is worth more than what it
  costs, and no term is lowest for a body standing still (runs 07 to 10).
- **Outcomes, never form.** A term judges what happened, never how the legs
  moved (the oscillators: every rule about how to move was gamed).
- **Shared gains, own duties.** What the body achieves and spends is shared
  by all; what a unit does to its neighbours and suffers is its own (runs 23
  and 24, cycle 16: a private cost set against a shared gain teaches doing
  less).
- **A ceiling on speed.** A reward for speed without one beats every price
  (runs 04 and 24).

## The parts

| Part | File | In short |
| --- | --- | --- |
| Task | [task.md](task.md) | The headless tester on a straight line laid at a small random angle; 20 s episodes; a world that loses the line restarts; the tow at the start |
| Reward | [reward.md](reward.md) | Run 28's: the team's line, squeeze and energy; each unit's own force share, pain and hanging |
| Observations | [observations.md](observations.md) | Five units' bodies (47 values each), the trail inside the unit's sight (15), the tow (1) |
| Actions | [actions.md](actions.md) | Eight leg torques and the spine joint behind, every 20 ms |
| Critic and learning | [critic.md](critic.md) | Two critics per unit, team and own; gSDE; PPO's settings |

## The bodies

Every design is screened on a tester before the full body (the procedure in
`../Training/results/follow-line/procedure.md`). The first tester is model
v5's headless half (`models/assembly_v5_headless.xml`, [model.md](../model.md)):
ten units alike, so that walking is learned before the front units must
carry a head. The head comes later, as its own segment, with a spine that
can lift the front (the user, 2026-10-20: a separate discussion of the
model).

## Open questions

Not decided; each is one change for a later run.

- **Stage 2's curves.** A level that makes the line bend more as the walk
  allows, raised by the wave's consistency (the user's idea, 2026-10-20):
  measured on the untowed noise-free worlds, so that the noise cannot fake
  it, with a fallback if no wave forms. Stage 1 has no level.
- **The ramps.** A run without them tells whether they are needed.
- **The discount.** 0.99 against 0.998.
- **The head**, as its own segment, on the model's later version.
