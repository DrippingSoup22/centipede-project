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
| 05 | 03 for 96 cycles | 1 | 3080 | Done |
| 04 | 01 + command cost, ratio 4 | 1 | 3080 | Done |
| 07 | 05 + command cost, ratio 4 | 1 | 3080 | Done |
| 08 | 05 + movement cost aimed at the buzz (2 parts, 10° unit) | 1 | 3080 | Running |

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
| 05 (beta 1, 96 cycles) | **44%** | 4.1% | 146.9 mm (174 steps) | 2.3% | 10.68° | 0.15 |
| 04 (command cost 4) | 0% | 9.1% | 11.6 mm | 16.1% | 1.19° | 0.00 |
| 07 (beta 1 + command cost 4, 96 cycles) | **41%** | −24.3% | 155.3 mm (168 steps) | 7.5% | 10.73° | 0.16 |

**Buzz** of the legs' sweep joints: the share of the frame-to-frame change's
power above 5 Hz, and the correlation of each change with the next (−1: the
leg flips back and forth every 20 ms step):

| Recording | Power above 5 Hz | Change-to-next correlation |
| --- | --- | --- |
| Baseline seed 1, mean action | 48% | +0.46 |
| 01, mean action | 56% | +0.32 |
| 02, mean action | 68% | +0.09 |
| 05, training cycles 1-4 / 45-48 / 93-96 | 70% / 83% / 92% | +0.23 / −0.06 / −0.28 |
| 05, mean action | **93%** | **−0.32** |
| 07, mean action | **95%** | **−0.39** |
| Random commands | 88% | −0.11 |

By joint kind, test 05's mean action changes its legs' lift and knee joints by
29° and 32° per frame (99% of it above 5 Hz), as much as random commands do
(24° and 32°), and the sweep joints by 11° (random: 17°); the baseline's mean
action, by 1.7°, 2.1° and 1°. The legs tap up and down at the control rate,
like a vibrating brush.

**Training**, first → last quarter of 32 cycles:

| Test | Arrivals | No closer | Speed | Toward | Joint movement | Legs' spread |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | 16% → 22% | 14% → 22% | 24.6 → 26.1 mm/s | 4.4 → 3.7 mm/s | 0.68 | 0.450 → 0.431 |
| 02 | 11% → 26% | 9% → 33% | 23.3 → 27.0 mm/s | 4.4 → 3.0 mm/s | 0.54 → 0.60 | 0.450 → 0.426 |
| 04 | 18% → 23% | 14% → 29% | 24.6 → 26.7 mm/s | 4.5 → 3.2 mm/s | 0.68 | 0.450 → 0.428 |
| 03, seeds 1 and 2, cycles 1-22 | 0% → 1-2% | | 16-18 mm/s | 2-3 → 4-6 mm/s | | |

Test 05 by quarters of its 96 cycles: arrivals 1% → 19% → 27% → 30%, no
closer 8% → 18% → 41% → 49%, speed 16.9 → 22.6 → 30.4 → 35.5 mm/s, toward
3.0 → 4.2 → 1.8 → 0.6 mm/s, heading error 33° → 62° → 91° → 102°, joint
movement 0.32 → 0.46 → 0.65 → 0.78, legs' spread 0.445 → 0.395.

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
- With beta 1 for 96 cycles (test 05) the mean action reached the target in
  44% of the evaluation's episodes, where every earlier mean action reached
  none: the policy now moves the body itself. But it moves it by buzzing:
  each leg flips back and forth at the control rate, more than under random
  commands, and the buzz grew over the training. Since pink noise cannot
  buzz, the policy learned to buzz with its mean action. Its aim also got
  worse as it got faster (heading error 33° → 102°): of the evaluation's 32
  episodes, 14 arrived, 14 ran out of time no closer than they started, 2
  ended closer, and 2 left the range circle.
- The command cost alone (test 04, white noise) changed nothing in 32 cycles,
  like the movement cost on 2026-10-08: it charged about 0.00065 per step, 70%
  of the step cost (a mean squared command of 0.18), but the commands' size
  barely moved (0.186 → 0.177) and neither did the spreads. While the noise
  does the walking, a cost on it reaches the policy only through the slowly
  learned spreads.
- With pink noise, the command cost (test 07) did not stop the buzz either:
  the mean action arrived in 41% of the episodes (13 of 32; 10 left the range
  circle) and buzzed even more (95% above 5 Hz), its mean squared command rose
  from 0.18 to 0.22, and its spreads fell further (0.436 → 0.376, against
  0.395 without the cost). The cost, about 0.0008 per step, is less than half
  of what the arrivals pay per step by the end (0.0018): buzzing still pays.
  Its training followed test 05's almost exactly (arrivals 1% → 19% → 26% →
  30%, speed 16.5 → 36.2 mm/s, heading error 33° → 99°, joint movement 0.32 →
  0.84).
