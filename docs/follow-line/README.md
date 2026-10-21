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
- **Biology's costs.** From run 09 the cost is the body's energy (the
  user, 2026-10-21: "the biology has all the answers"), from run 10 as a
  muscle pays it: its work, holding its force, and switching its force on
  and off, which makes quick strokes dear. The pace comes from the balance
  of that energy and the task's priority, the value of its time, as an
  animal's does. Runs 05 to 08 charged the joints' speeds or jerk instead,
  measures that grow with speed by the gait's whim (run 05 sped up to 353
  mm/s, run 06 crept), and run 08 had to cap the line ([reward.md](reward.md)).

## The parts

| Part | File | In short |
| --- | --- | --- |
| Task | [task.md](task.md) | The headless tester on a straight line laid at a small random angle; 20 s episodes; a world that loses the line restarts; the tow at the start |
| Reward | [reward.md](reward.md) | From run 10: the team's line, of which the body's energy, as a muscle pays it (work, holding, switching), takes the share h / (h + E), h the task's priority, so the body walks at its cheapest pace; each unit's own carry |
| Observations | [observations.md](observations.md) | Five units' bodies (45 values each from run 10), the trail inside the unit's sight (15), the tow (1) |
| Actions | [actions.md](actions.md) | From run 11: eight leg targets and the spine joint's behind, each a target angle its motor pulls toward as a spring and a damper (torques in runs 01 to 10), every 20 ms |
| Critic and learning | [critic.md](critic.md) | Two critics per unit, team and own (one in runs 05 and 09); how they see; gSDE; PPO's settings |

## The bodies

Every design is screened on a tester before the full body (the procedure in
`../Training/results/follow-line/procedure.md`). The first tester is model
v5's headless half (`models/assembly_v5_headless.xml`, [model.md](../model.md)):
ten units alike, so that walking is learned before the front units must
carry a head. From run 11 it is model v6 headless, the same body whose
motors pull toward targets ([actions.md](actions.md)); a full v6 body
follows if targets walk better. The head comes later, as its own segment, with a spine that
can lift the front (the user, 2026-10-20: a separate discussion of the
model).

## Open questions

Not decided; each is one change for a later run.

- **Stage 2's curves.** A level that makes the line bend more as the walk
  allows, raised by the wave's consistency (the user's idea, 2026-10-20):
  measured on the untowed noise-free worlds, so that the noise cannot fake
  it, with a fallback if no wave forms. Stage 1 has no level.
- **The terms left out**, each brought back as its own run if a test shows
  it is needed: the force share (a unit carried along on skidding feet),
  the squeeze, pain (legs crossed over their neighbours').
- **The energy's weights**: the priority, 1 mJ/s, and the switching and
  holding prices, set on the scripted walk by one measurement of human
  muscle.
- **The exploring noise's share.** The noise switches the commands too,
  about 12 mJ/s a unit: it lowers the pay, so the agents gain by narrowing
  it, and acts as a larger priority until they do.
- **The tow's walking test** counts feet lifting off after stances of 60
  ms: it misses gaits above about 8 strokes a second, and brought run 09's
  tow back once its legs buzzed.
- **The motors' stiffness.** Full strength a quarter of each joint's range
  away from its target, critically damped.
- **The discount.** 0.99 against 0.998.
- **The head**, as its own segment, on the model's later version.
