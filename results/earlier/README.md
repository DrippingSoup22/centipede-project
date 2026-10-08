# Earlier runs

Runs from 2026-10-08, before the reward study, that the report may use. All
start from the standing prototype (`runs/2026-10-07_2221_baseline_stage2`),
with 64 worlds, rollout windows of 64 steps, 256-step episodes, seed 1 unless
the name says otherwise, and one evaluation of 32 worlds with the policy's
mean action. The full account is in the development plan (local).

## Far targets and the movement cost

Task: targets 30–60 mm away within ±30°, arrival when the target lies under
the head, range circle 2.5 × the start distance; a straight walk reaches 18%
of these targets by luck. Reward: arrival 1, the head's progress shared by
every segment, and a step cost, body contact, and leg contact (budget
2 : 3 : 1), plus a movement cost where stated.

| Run | Machine | Body | Movement cost | Cycles |
| --- | --- | --- | --- | --- |
| `2026-10-08_1603_far_01_stand_far_256` | RTX 3080 | spine passive | none | 64 |
| `2026-10-08_2116_spine_02_movement_1` | Kaggle T4 | spine control | 1 part | 32 |
| `2026-10-08_2147_spine_03_movement_2` | Kaggle T4 | spine control | 2 parts | 32 |

First → last quarter of the first 32 cycles:

| Measure | far_01 | spine_02 | spine_03 |
| --- | --- | --- | --- |
| Arrived | 14% → 23% | 12% → 24% | 12% → 25% |
| Head speed in training (mm/s) | 21.5 → 24.3 | 24.4 → 25.1 | 24.5 → 26.6 |
| Joints' movement (random commands = 1) | – | 0.68 → 0.67 | 0.68 → 0.67 |
| Legs' action spread | 0.451 → 0.434 | 0.450 → 0.432 | 0.451 → 0.430 |
| Spine's action spread | – | 0.438 → 0.431 | 0.438 → 0.436 |
| Legs touching | 8.0% → 8.1% | 11.8% → 11.5% | 11.4% → 11.0% |
| Travel turning toward the target (recordings) | 55% → 57% | 58% → 62% | 60% → 65% |
| Head within 10 mm of the target | 30% → 26% | 24% → 26% | 27% → 34% |
| Mean action, evaluation | 8 mm in 5 s, 0% arrived (cycle 64) | 10 mm in 5 s, 0% | 8 mm in 5 s, 0% |

Chance for the turning share is 50%; with about 250 one-second pieces per
recording, one standard deviation is about 3%. Joint movement of 0.64 was
measured separately for far_01's training walk.

**What they showed:** the centipede walks forward with the exploration noise
(about 25 mm/s in training) but not with its mean action, and arrivals stay
near a straight walk's luck. The movement cost, at 1 and 2 parts of the cost
budget, changed neither the joints' movement nor the noise in 32 cycles: per
step it was about −0.0002 per segment, small next to the arrival and progress
rewards (about +0.001 each). Spine control left the spine bending at random
(about 8°) and raised leg contact by about 3 points.

## The followers' share of the progress (older task)

Task: targets 10–20 mm away within ±15°, arrival within 1 mm of the head's
tip, no range circle; a straight walk reaches about 26.5% of these targets by
luck. 64 update cycles.

| Run | Followers' progress | Seed |
| --- | --- | --- |
| `2026-10-08_1304_rules_01_stand_rules` | Each follower's own distance to its leader's spot, at a tenth | 1 |
| `2026-10-08_1330_rules_02_stand_rules_seed2` | The same | 2 |
| `2026-10-08_1446_shared_01_stand_shared` | The head's progress, shared by every segment | 1 |
| `2026-10-08_1513_shared_02_stand_shared_seed2` | The same | 2 |
| `2026-10-08_1540_shared_03_stand_no_follower_progress` | None | 1 |

**What they showed:**

- *Own spot (rules_01, rules_02):* arrivals 21–23%; the heading error rose to
  142–145°; no episode that ran out of time ended closer than it started; the
  head's path per episode grew from 123 to 157–167 mm while the final distance
  grew from 34 to 70–81 mm. Each follower's progress reward rose to +0.0045
  per step, five times the step cost, while the head's fell to −0.0016: the
  segments are hinged at a fixed distance, so a follower's distance to its
  leader's last spot shrank by the body's forward step, which rewarded speed
  whatever the direction.
- *Shared progress, seed 1 (shared_01):* the overshoot stopped: the head's path
  per episode stayed near 120 mm, the final distance fell from 26 to 19 mm, no
  episode ended beyond 50 mm (79% in rules_01), the heading error fell from
  110° to 91°, and 13% of the episodes that ran out of time ended closer than
  they started. Arrivals 22–25%; critic accuracy 0.33–0.39; with the mean
  action the head moved 8 mm in 5 s.
- *Shared progress, seed 2 (shared_02):* the overshoot stopped too (head path
  near 125 mm, final distance 27–32 mm, nothing beyond 50 mm), but the
  heading error stayed at 110–117°, no episode that ran out of time ended
  closer, arrivals were 23–27%, and legs touching rose from 9% to 15%.
- *No follower progress (shared_03):* stopped at cycle 42 of 64 when its queue
  was stopped; not evaluated and not reviewed.
