# The follow-line design

This folder holds the design of the `follow-line` workspace branch, one file
per part. The other documents in `docs/` describe the program this branch
started from (`main` at ec907b5); where this folder decides differently,
this folder holds on this branch.

## The project

A centipede gait can emerge from many agents, each seeing only its part of
the body. What may never change is in [constraints.md](constraints.md);
what the literature knows, in [research.md](research.md). Every run's
results and lessons are in the study's `Training/results/follow-line/`.

## The job

The body learns to walk by following a line on the ground (the user's plan,
2026-10-20):

1. **A straight line**, until a wave gait forms: every leg stepping, and
   neighbouring legs in a steady phase lag.
2. **Lines that curve**, gradually, so that the wave carries the body
   through turns.

The next job, on its own branch, is `hunt-target`: a leader draws the line
toward a target, and the body follows it.

## The parts

The design as runs 10 and 11 used it.

| Part | File | In short |
| --- | --- | --- |
| Task | [task.md](task.md) | The tester on a straight line laid at a small random angle; 20 s episodes; a world that loses the line restarts; a fading tow at the start |
| Reward | [reward.md](reward.md) | The team is paid its line, of which the body's energy takes the share h / (h + E); each unit's own account is charged the weight its feet do not carry |
| Observations | [observations.md](observations.md) | Five units' bodies (45 values each), the line inside the unit's sight (15), the tow (1) |
| Actions | [actions.md](actions.md) | Eight leg motors and the spine joint behind, every 20 ms: targets on v6, torques on v5 |
| Critic and learning | [critic.md](critic.md) | Two critics per unit, team and own, each seeing what its actor sees; gSDE; PPO |

## The bodies

The testers are ten units alike, without a head: model v5's headless half
(`models/assembly_v5_headless.xml`, torque motors, [model.md](../model.md))
and v6 headless, the same body with target motors
(`models/assembly_v6_headless.xml`, [actions.md](actions.md); run 11). The
full body, with a head as its own segment, comes later.

## Open questions

- **Stage 2's curves**: a level that makes the line bend more as the walk
  allows (the user's idea, 2026-10-20).
- **The head**, as its own segment, on a later model.
