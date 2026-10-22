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

The design of run 12, agreed with the user on 2026-10-21. Training goes in
stages: stage 1 from scratch at a moderate energy price, stage 2 from its
checkpoint at a larger one ([reward.md](reward.md)).

| Part | File | In short |
| --- | --- | --- |
| Task | [task.md](task.md) | The tester on a straight line laid at a small random angle; 20 s episodes; a world that loses the line restarts; a fading tow at the start |
| Reward | [reward.md](reward.md) | The team is paid near a commanded speed, 67 mm/s, and each unit charged its neighbourhood's muscle energy; each unit's own account is charged its carry: from run 14 each foot's load beyond a quarter either way of half the unit's weight, over 1 s |
| Observations | [observations.md](observations.md) | Five units' bodies (46 values each), the line inside the unit's sight (15), the tow (1) |
| Actions | [actions.md](actions.md) | Eight leg motors and the spine joint behind, every 20 ms: targets on v6, torques on v5 |
| Critic and learning | [critic.md](critic.md) | Two critics per unit, team and own, 1 : 1, each seeing what its actor sees; gSDE from 0.3; PPO, discount 0.99 |

## The bodies

The testers are ten units alike, without a head: model v5's headless half
(`models/assembly_v5_headless.xml`, torque motors, [model.md](../model.md))
and v6 headless, the same body with target motors
(`models/assembly_v6_headless.xml`, [actions.md](actions.md); run 11). The
full body, with a head as its own segment, comes later.

**v6's trunk** was made more mobile on 2026-10-22 (the user's decision;
`archive/model_v6_spine/make_model_v6.py` in the study's folder). Runs 11
and 12 used the first v6, whose links had only yaw and pitch hinges.

| Link hinge | First v6 | v6 now | Why |
| --- | --- | --- | --- |
| Yaw (sideways) | ±15°, spring 2.2e-4 N·m/rad, the unit's motor | Unchanged | *S. heros* bends 4 to 8.5° a joint running, peaks near 15 (Anderson, Shultz & Jayne 1995) |
| Pitch (up-down) | ±8°, spring 1.1e-3 | ±10°, spring 7.8e-4 (0.7 times) | A little looser: the body dips over rough ground a few degrees a joint (Ozkan-Aydin et al. 2020, Fig. 4a); the first range was an engineering guess |
| Roll (twist) | None | ±5°, spring 5.0e-3 (4.5 times the first pitch spring), damped as pitch | Centipedes twist, but locally and over several segments (Pierce et al. 2026); the stiffest spring the 1.33 ms step keeps stable |

No stiffness of a real trunk has been measured; the springs keep the first
v6's character. On the scripted body (the study's
`archive/model_v6_spine/body_tests_*.txt`):

| Test | First v6 | v6 now |
| --- | --- | --- |
| Standing, unit 5 limp: how far it sinks, what its neighbours hold | 0.28 mm, 95% | 0.31 mm, 95% |
| Standing, unit 5 on one foot: its roll | 0.1° | 0.6° |
| The scripted wave: speed, energy for each mm, the units' mean roll | 104 mm/s, 0.025 mJ, 2.2° | 80 mm/s, 0.035 mJ, 2.6° |

The pitch alone costs the scripted wave little (98 mm/s); any roll hinge,
whatever its damping or range (±1 to ±5°), costs it 12 to 23%. The script
was tuned on a body that could not twist.

## Open questions

- **Stage 2's curves**: a level that makes the line bend more as the walk
  allows (the user's idea, 2026-10-20).
- **The head**, as its own segment, on a later model.
- **The carry in turns**: the inner legs carry more in a bend, which the
  per-leg carry's even share does not allow for (the user, 2026-10-22).
