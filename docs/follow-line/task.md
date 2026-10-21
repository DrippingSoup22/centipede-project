# Task

Agreed with the user on 2026-10-20.

## The body

A headless tester, v5 or v6: ten units alike, ten agents
([README.md](README.md#the-bodies)).

## A world's episode

```text
restart: the model's pose, a new straight line at a small random angle
   │  every 20 ms each unit acts; the team is paid near a commanded speed
   ├─ the units lose the line (their mean distance beyond the sight radius) ─┐
   └─ 20 s pass ─────────────────────────────────────────────────────────────┴─ restart
```

| Setting | Value | Why |
| --- | --- | --- |
| Start pose | The model's pose, each leg angle moved by up to ±2° and each leg joint given a small random speed (main's reset), always facing the same way | The floor is the same everywhere and the agents sense no direction in the world, so a random heading would change nothing |
| The line | Straight, through the mean of the units' centres, pointing where the body points turned by a random angle of up to ±10°; it stays for the episode | The body starts on the line; the small angle makes every episode start with a correction, so that following, not only walking straight, is practised all along (the user's worry, 2026-10-20). Curves are stage 2 |
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
