# Night tests, 2026-10-09

On the night of 2026-10-09 the user left the assistant to run short tests
alone, one at a time on the RTX 3080 (at most 5 hours, each run at most 1
hour) and on Kaggle's T4s, starting with the two smoothness components
discussed for the reward study. These are screening tests, not the reward
study's two-seed conditions: most run one seed, and a condition gets a second
seed only when it looks promising.

## Protocol

- **Starting point:** the reward-study baseline
  (`configs/reward_study/01_baseline.toml`): model v3 with spine control,
  targets 30-60 mm within 30 degrees, 256-step episodes, the baseline reward,
  the default networks (64 x 64) and PPO settings, from the standing prototype.
- **Changes from it:** 256 worlds instead of 64 (minibatch 2048, so each cycle
  still makes 32 gradient steps per agent), tested first (test 01), then one
  change at a time on top. Configurations: `configs/night/`.
- **Goal, set by the user:** the policy's mean action reaches the target.
  Judged by the evaluation (mean action, no exploration noise; seed 101, 32
  worlds, each world's first episode): arrivals, the share of the start
  distance closed, and the head's path. Smoothness and the training measures
  break ties.
- **Measures:** the training quarters and the evaluation as in
  [the reward study](../reward_study/README.md); the legs' sweep joints from
  the recordings: amplitude (standard deviation), change per 20 ms frame,
  rhythm (the autocorrelation's first peak after its first zero crossing),
  left-right correlation, and the correlation with the segment behind.

## Tests

| # | Change from the reward-study baseline | Seeds | Machine | Status |
| --- | --- | --- | --- | --- |
| 01 | 256 worlds, minibatch 2048 | 1 | 3080 | Done |
| 02 | 01 + colored noise, beta 0.5 | 1 | 3080 | Done |
| 03 | 01 + pink noise, beta 1 | 1, 2 | Kaggle | Ran; results still on Kaggle (login expired) |
| 05 | 03 for 96 cycles | 1 | 3080 | Running |
| 04 | 01 + command cost, ratio 4 | 1 | 3080 | Planned |

## Results

The reference is the reward-study baseline on Kaggle (64 worlds, two seeds;
[01_baseline](../reward_study/01_baseline/README.md)): its mean action closed
8% of the start distance and arrived in none of the 32 evaluation episodes,
while the training arrivals ended at 19-20%.

**Evaluation, mean action** (seed 101, 32 worlds, 256 steps):

| Test | Arrivals | Distance closed | Head's path | Legs touching | Sweep change per frame | Rhythm |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline, seed 1 (64 worlds) | 0% | 8.6% | 8.1 mm | 0.3% | 0.95° | 0.01 |
| 01 (256 worlds) | 0% | 8.4% | 22.9 mm | 22.5% | 1.25° | 0.01 |
| 02 (beta 0.5) | 0% | **24.6%** | 20.7 mm | 6.1% | 2.26° | 0.03 |

**Training**, first → last quarter of 32 cycles:

| Test | Arrivals | No closer | Speed | Toward | Joint movement | Legs' spread |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | 16% → 22% | 14% → 22% | 24.6 → 26.1 mm/s | 4.4 → 3.7 mm/s | 0.68 | 0.450 → 0.431 |
| 02 | 11% → 26% | 9% → 33% | 23.3 → 27.0 mm/s | 4.4 → 3.0 mm/s | 0.54 → 0.60 | 0.450 → 0.426 |
| 03, seeds 1 and 2, cycles 1-22 | 0% → 1-2% | | 16-18 mm/s | 2-3 → 4-6 mm/s | | |

**Readings so far:**

- Four times more worlds (test 01) cost 25% more time per cycle (30 s
  against about 24 s on the 3080) and changed little in 32 cycles.
- Colored noise reduces the jitter that moved the body in training: with
  beta 1 the training arrivals fell from about 20% to 0% at first and the head
  slowed from 25 to 16 mm/s; with beta 0.5 the arrivals dipped to 11% and
  recovered within 20 cycles. The white noise had been moving the body like a
  vibrating brush, while the mean action only held a posture.
- With beta 0.5 the mean action closed three times as much of the distance as
  with white noise (25% against 8%), one seed. It still arrived nowhere and
  its legs have no rhythm yet.
