# Task

Agreed with the user on 2026-10-20.

## The body

A headless tester, v5 or v6: ten units alike, ten agents
([README.md](README.md#the-bodies)).

## A world's episode

```text
restart: the model's pose, a new line through the body (stage 2: bending beyond the head),
         a commanded speed (stage 3: drawn at its level)
   │  every 20 ms each unit acts; the team is paid near a commanded speed
   ├─ the units lose the line (their mean distance beyond the sight radius) ─┐
   └─ 20 s pass ─────────────────────────────────────────────────────────────┴─ restart
```

| Setting | Value | Why |
| --- | --- | --- |
| Start pose | The model's pose, each leg angle moved by up to ±2° and each leg joint given a small random speed (main's reset), always facing the same way. From run 25, on uneven ground: at a random point of a ground drawn from a fixed mix, facing any way, set down just above the ground ([terrain.md](terrain.md#a-start)) | The floor is the same everywhere and the agents sense no direction in the world, so a random heading would change nothing. On uneven ground where and which way the body starts decide what it meets |
| The line | Through the mean of the units' centres, pointing where the body points; it stays for the episode. Stage 1: straight. Stage 2 (from run 21): straight through the body, then bending piece by piece ([the curves](#the-curves-stage-2)) | First walk straight and stay on the line (the user, 2026-10-22, from run 16): runs 01 to 15 turned it by a random angle of up to ±10° so that every episode started with a correction, and the bodies walked turned to it, the rear units up to 9 mm off (`line_angle_range_deg`). Then follow a line that turns |
| Its direction | The line is an arrow: a step along it is paid, a step against it charged | Walking backward never pays, whichever way the body faces |
| Episode | 20 s, 1,000 decisions | legged_gym's 20 s (Rudin et al. 2021); a world stuck early waits at most 20 s |
| First episodes | In training, each world starts at a random moment of its first episode; evaluations start every world together | Otherwise all training worlds restart on the same step |
| Sight radius | 2.5 unit spacings, 16.5 mm on v5: the far ends of the two units ahead and behind, the body straight | The reach of what a unit sees ([observations.md](observations.md)) |
| A world loses the line | It restarts once its units' mean distance from the line passes the sight radius | No batch fills with steps where the line pays nothing; the same distance as the line's pay band ([reward.md](reward.md)) |

**Every ending is a truncation.** At the time limit, when it loses the
line, and when MuJoCo Warp makes it non-finite (rare, and counted), a world
restarts and the agents value what would have followed with their critics,
as if the walk went on (Pardo et al. 2018). Valuing an ending at zero would
pay an untrained body, whose steps are worth less than zero, for leaving
the line.

## The curves (stage 2)

Designed with the user on 2026-10-22, once the walk was learned (runs 16
and 17): the body must follow a line that turns, never the same, so that it
learns to follow lines in general. One setting chooses the stage:
`line_shape = "straight"` (the default, stage 1) or `"curved"`, which reads
`[environment.curves]`; a curved run's file returns to stage 1 by that word
alone, its inputs unchanged (on a straight line the command stays at
67 mm/s). Each world's curved line is laid at every restart:

- **Straight through the body** and half a body length beyond the head, so
  the body starts aligned with it, as in stage 1.
- **Then piece by piece**: each piece one to three body lengths long,
  straight or bent to the left or right, its curvature drawn evenly up to the
  level's tightest.
- **No corner**: a piece's curvature moves from the one before's in a
  straight line over the piece's first body length, so the line's heading
  never jumps and its curvature never changes faster than that; a 90° corner
  cannot happen. Roads, railways and robot paths are laid the same way, with
  bounded curvature and a bounded rate of change of curvature (Fraichard &
  Scheuer 2004).
- **Kept as points** a millimetre apart, each with its direction and
  curvature; a unit's place, distance and angle are measured at its nearest
  point, and its trail follows the curve.

**Levels**, the user's design: the lines start gentle and get sharper, with
fewer straight pieces, as the body follows them. The tightest radius stays
within what the animal bends: *S. heros* bends 4 to 8.5° a joint running
straight (Anderson, Shultz & Jayne 1995), and our joints stop at ±15°. No
turning radius of a centipede was found.

| Level | Tightest radius | Each joint bent | Straight pieces |
| ---: | --- | --- | --- |
| 1 | 5 body lengths (330 mm) | 1.1° | 1 in 2 |
| 2 | 3 (198 mm) | 1.9° | 2 in 5 |
| 3 | 2 (132 mm) | 2.9° | 1 in 3 |
| 4 | 1.5 (99 mm) | 3.8° | 1 in 4 |
| 5 | 1 (66 mm) | 5.7° | 1 in 5 |

The level rises by one, at most once in 16 update cycles, when the untowed
noise-free worlds kept the line through 90% of their episodes over those 16
cycles (their lost lines per episode's worth of their steps); it never falls.
Starting easy matters because the walk was learned on a straight line alone:
what a policy learns first shapes what follows (Hwangbo et al. 2019), and
the standard legged recipe raises its terrain's difficulty as the robots
succeed (Rudin et al. 2021). The level is kept in checkpoints. On a bend
the command falls ([reward.md](reward.md#the-commanded-speed)).

**When the curves begin** (from run 22, the user's decision, 2026-10-22;
`start_speed_mm_per_s`, `start_pair_consistency`; on uneven ground, run 25,
the speed is 33 mm/s, [terrain.md](terrain.md#the-gates-on-the-mix)). A run from scratch keeps
its lines straight until the walk has formed and the wave starts forming, so
that the bends are there while the order between neighbours sets: neither
before the wave forms nor after it froze. In runs 12 to 19, from their
training logs (the study's archive, `checks/run_12/wave_timeline_12_20.txt`):

| What | When (update cycles) | The noise then |
| --- | --- | --- |
| The walk formed: the noise-free, never-towed speed at 50 mm/s | 58 to 93 | 0.28 to 0.30 |
| A neighbour pair rose above chance: consistency 0.25 (chance about 0.2; before it the highest pair stays at 0.22 to 0.25) | 10 to 60 cycles before it ordered | |
| The first pair ordered: 0.30. The first pair to fix its direction for good did so in that very cycle in 5 of 11 runs, within 11 cycles of it in 9 | 63 to 165 | 0.24 to 0.30 |
| The other pairs fixed, one by one: 80% of them | 176 to 383 | 0.13 to 0.28 |

So the first level begins once, over the last 8 update cycles, the untowed
noise-free worlds walk at least 50 mm/s and some pair's consistency is at
least 0.25. Before it the level is −1 (straight lines), in the log and in
checkpoints.

**Mixed levels** (from run 22, `mixed_levels`). Each world's line is laid
at a level drawn evenly from the first to the one reached, so the easier
lines and their straight pieces stay in training: the legged recipe sends
the robots that solve its top level back to a random one, against
forgetting (Rudin et al. 2021), and a policy trained on new states loses
skills on those it no longer visits (Wołczyk et al. 2024). The level is
judged by the worlds laid at it alone.

## The speeds (stage 3)

Designed with the user on 2026-10-23, once the body walked and followed
curving lines (run 22): the commanded speed changes, so that the body finds
an economical gait at every speed and learns to move between them. A head
will choose the speed in a later stage; until then the task draws it and
every unit sees it ([observations.md](observations.md#the-command-up-to-2-values-stages-2-and-3)),
as an animal's descending command reaches every segment: the lamprey's
reticulospinal cells start, keep and stop locomotion along the whole cord
(Daghfous et al. 2016), and a fly's descending neurons start its forward
walking (Bidaye et al. 2020). `[environment.speeds]` sets the levels;
without them the command stays at 67 mm/s.

- **Holds.** At each restart a world draws a speed at its level and holds
  it for the level's hold, then draws again; a hold of 0 keeps it the whole
  episode. The speed is drawn evenly in the level's range, or 0, a stop,
  with the level's stop share; stops belong only to levels whose range
  starts at 0 (the user, 2026-10-23).
- **The pay is the same** ([reward.md](reward.md#the-commanded-speed)): the
  body's speed and the command, each averaged over 1 s, so that the lag
  after a change counts the same on both sides; a restart is a change from
  rest. In bends the command falls as in stage 2, whatever the speed.

| Level | The speed holds for | Range | Stops | What it teaches |
| ---: | --- | --- | --- | --- |
| 1 | the whole episode | 50 to 85 mm/s | none | a good gait near the old command |
| 2 | the whole episode | 30 to 100 mm/s | none | a good gait at every speed |
| 3 | the whole episode | 0 to 100 mm/s | one episode in five | standing when told |
| 4 | 10 s | 0 to 100 mm/s | one hold in five | moving between speeds, slowly |
| 5 | 7 s | 0 to 100 mm/s | one hold in five | |
| 6 | 5 s | 0 to 100 mm/s | one hold in five | |
| 7 | 3 s | 0 to 100 mm/s | one hold in five | moving between speeds, often |

**Speeds first, then changes** (the user's agreement, 2026-10-23): the body
finds an economical gait for each speed before it must move between them,
so the changing levels teach only the moves; with mixed levels the easier
ones stay in training.

**The range's top, 100 mm/s**, 1.5 body lengths a second, the top of *S.
heros*' 0.5 to 1.5 ([reward.md](reward.md#numbers-fixed-by-measurement)).
The scripted wave reaches 135 mm/s at 7 strokes a second, but spends 0.053
mJ for each mm there, against 0.030 to 0.037 from 46 to 99 mm/s (the study's
archive, `ledger/speed_steps.txt`).

**The holds leave time to learn, not to move.** A coordinated walk reaches a
new speed within a stroke: after the largest jumps, from standing to 135
mm/s and from 46 to 115, the scripted wave's speed, averaged over 1 s,
stays within 6.5 mm/s of the command averaged alike, and within 3.6 mm/s
after a jump between neighbouring speeds (0, 81, 135 mm/s) (the same file). What the body needs time for is finding an
economical gait, which the level's judgement waits for.

**A level rises** by one, at most once in 16 update cycles, when the untowed
noise-free worlds drawn at it, over those cycles:

- kept their speed within **8 mm/s** of the command on the mean, both
  averaged over 1 s, as in stage 2's sign;
- spent at most **0.05 mJ for each mm** while the command was above 0: a
  unit's energy over its distance, which changes little with the speed (the
  scripted wave spends 0.030 to 0.041 from 46 to 115 mm/s, run 22's learned
  walk 0.035 to 0.044). A quick, tapping gait that holds the speed spends
  several times as much (0.27 in run 11, against the scripted wave's 0.025
  on v6; 0.10 to 0.17 in run 12) and does not pass: a body pushed through
  the levels before it finds an economical gait would learn to tap (the
  user, 2026-10-23);
- kept the line through **90%** of their episodes.

Legged robots widen their commands' range where they succeed in the same
way (Margolis et al. 2022). Each judgement is logged: the speed error,
within 2 s of a change and after, the energy for each mm and the lines kept.
The level is kept in checkpoints. With `start_after_curves` the levels begin
once the curves reach their last level; before, the level is −1 and the
command 67 mm/s.

## The tow

Training wheels that fade (`docs/cooperators/tow.md` on that branch).
Without them a unit's own step moves nothing while its neighbours stand, so
exploring finds nothing to learn from.

- **The push.** Each unit is pushed forward along its heading by its world's
  tow level times its weight, less as it speeds up: max(0, 1 − v / 33 mm/s)
  of it, none at the help speed.
- **Each world its own level**, drawn at every restart between half the top
  and the top; each unit observes it.
- **The top fades.** It starts at 0.2 (`[environment.tow] start`, off by
  default; run 01 sets it). After every update cycle, once 128
  steps have passed since it last moved, it falls by 0.02 if the untowed
  noise-free worlds walked (their line paid at least 2.7 mm/s a unit, their
  speed below the commanded one, and their feet lifting off at least once a
  second), and rises by 0.02 otherwise, up to its start.
- **Noise-free worlds.** Besides the 4,096 that learn, 2 × 128 worlds act
  with the agents' mean actions and are never learned from: one group at the
  tow's top, one never towed, the honest measure of the walk. Evaluations
  never tow.
