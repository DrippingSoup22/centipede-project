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

1. **Independent agents.** Each unit has its own actor, critic and
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
  From run 08 a cost takes a share of the line's pay, never more: a price
  subtracted from it outgrew the pay in this branch's run 07, and the body
  stopped walking ([reward.md](reward.md)).
- **Outcomes, never form.** A term judges what happened, never how the legs
  moved (the oscillators: every rule about how to move was gamed).
- **Shared gains, own duties.** What the body achieves and spends is shared
  by all; what a unit does to its neighbours and suffers is its own (runs 23
  and 24, cycle 16: a private cost set against a shared gain teaches doing
  less).
- **Biology's costs.** From run 09 the cost is the body's energy as physics
  measures it: its motors' work and the force they hold (the user, 2026-10-21:
  "the biology has all the answers"). The pace comes from the balance of
  that energy and the task's priority, the value of its time, as an
  animal's does. Runs 05 to 08 charged the joints' speeds or jerk instead,
  measures that grow with speed by the gait's whim (run 05 sped up to 353
  mm/s, run 06 crept), and run 08 had to cap the line ([reward.md](reward.md)).

## The parts

| Part | File | In short |
| --- | --- | --- |
| Task | [task.md](task.md) | The headless tester on a straight line laid at a small random angle; 20 s episodes; a world that loses the line restarts; the tow at the start |
| Reward | [reward.md](reward.md) | From run 09: the team's line, of which the body's energy (its motors' work and holding) takes the share h / (h + E), h the task's priority, so the body walks at its cheapest pace; nothing else |
| Observations | [observations.md](observations.md) | Five units' bodies (43 values each), the trail inside the unit's sight (15), the tow (1) |
| Actions | [actions.md](actions.md) | Eight leg torques and the spine joint behind, every 20 ms |
| Critic and learning | [critic.md](critic.md) | One critic per unit (two, team and own, in runs 01 to 04 and 06 to 08); gSDE; PPO's settings |

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
- **The terms left out**, each brought back as its own run if a test shows
  it is needed: the carry (lazy legs, if holding does not stop them), the
  force share (a unit carried along on skidding feet), the squeeze, pain
  (legs crossed over their neighbours').
- **The priority and holding's weight**: 1 mJ/s and 2.7 mJ/s for a motor
  fully on.
- **The exploring noise's share.** The noise's energy lowers the pay, so
  the agents gain by narrowing it.
- **The discount.** 0.99 against 0.998.
- **The head**, as its own segment, on the model's later version.
