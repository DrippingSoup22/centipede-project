# Pink-noise study

Decided with the user on 2026-10-09, after the night tests
([`night_2026-10-09/`](../night_2026-10-09/README.md)) showed that pink
exploration noise made the policy's mean action reach the target, by a buzz of
the legs. Pink noise is the base; each test changes one thing.

## Protocol

- **Reference:** night test 05: the reward-study baseline with pink
  exploration noise (β = 1), 256 worlds, minibatch 2,048, 96 update cycles,
  starting from the standing prototype (`runs/2026-10-07_2221_baseline_stage2`).
- **Changes:** one per test, in training settings, networks, or reward
  weights. Configurations: `configs/pink_study/`.
- **Seeds:** one per screening test; a change that does better than the
  reference gets a second seed.
- **Evaluation:** the policy's mean action on 128 episodes, seeds 102 and 103
  with 64 worlds each (`configs/pink_study/evaluation.toml`), because 32
  episodes leave about ±9 points of uncertainty on an arrival rate. The
  reference's own runs were evaluated on 32 episodes (seed 101) only; its
  128-episode evaluation is still to be run.
- **Measures:** as in the night tests: evaluation endings, the share of the
  start distance closed, the legs' buzz and gait from the evaluation's
  recordings (`claude-notes/analysis/`), steering, and the training quarters.

## Tests

| # | Change from test 05 | Why | Seeds | Machine | Status |
| --- | --- | --- | --- | --- | --- |
| 01 | GAE λ 0.9 | Recommended by Andrychowicz et al. (ICLR 2021); with weak critics a follower's advantage reaches over fewer future steps | 1 | Kaggle T4 | Done |
| 02 | Progress worth twice as much (`head_progress_ratio` 2) | Test 05's aim got worse as the arrivals rose | – | – | Planned |
| 03 | 1,024 worlds, minibatch 8,192 | The same 32 gradient steps per update cycle, each from four times the samples (Rudin et al., CoRL 2021) | – | – | Planned |
| 04 | Hidden layers 256 × 256, widened from the prototype | The network's capacity | 1 | Kaggle T4 | Done |

## Results

**Evaluation, mean action** (128 episodes; the reference on 32):

| Run | Episodes | Arrived | Left the range circle | Out of time | Distance closed | Episode length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 05 seed 1 (reference) | 32 (seed 101) | 44% | 6% | 50% | 4.1% | 174 steps |
| 05 seed 2 (reference) | 32 (seed 101) | 28% | 41% | 31% | −73.2% | 184 steps |
| 01, GAE λ 0.9 | 128 | 27% (25%, 28%) | 17% | 56% | −45.8% | 201 steps |
| **04, 256 × 256** | 128 | **81% (80%, 83%)** | **0%** | 19% | **90.2%** | 154 steps |

Values in brackets are evaluation seeds 102 and 103.

**Legs and steering**, from the evaluation's recordings (mean action):

| Measure | 05 seed 1 | 01 | 04 |
| --- | ---: | ---: | ---: |
| Sweep change's power above 5 Hz (buzz) | 93% | 94% | 93% |
| Change-to-next correlation (−1: flips every step) | −0.32 | −0.35 | −0.40 |
| Lift / knee change per 20 ms frame | 29° / 32° | | 29° / 36° |
| Spine yaw change per frame | 5.4° | | 10.6° |
| Rhythm (0: none; 1: clean oscillation) | 0.15 | | 0.16 |
| Left-right correlation (alternating legs: negative) | +0.11 | | +0.14 |
| Wave to the segment behind | +0.28 at lag 0 | | +0.31 at lag 0 |
| Travel turning toward the target (1 s pieces) | 45% | | 75% |
| Head within 10 mm of the target | 66% | | 74% |

**Training**, by quarters of the 96 update cycles:

| Run | Arrivals | Ended no closer | Speed toward the target | Heading error | Joint movement |
| --- | --- | --- | --- | --- | --- |
| 05 seed 1 | 1% → 19% → 27% → 30% | 8% → 49% | 3.0 → 0.6 mm/s | 33° → 102° | 0.32 → 0.78 |
| 05 seed 2 | 1% → 18% → 26% → 27% | 7% → 55% | 3.1 → −1.3 mm/s | 32° → 105° | 0.31 → 0.80 |
| 01 | 1% → 19% → 28% → 28% | 6% → 55% | 3.2 → −1.8 mm/s | 33° → 103° | 0.31 → 0.85 |
| 04 | 2% → 22% → 33% → 45% | 6% → 14% | 3.2 → 6.5 mm/s | 34° → 81° | 0.33 → 0.87 |

In the last quarter: critic explained variance of the seven followers 0.17-0.33
for 04, 0.04-0.18 for 05; KL per update about 0.022 and clip fraction about
0.28 for 04, 0.013 and 0.18 for 05; reward per step from arrivals and
progress +0.0023 and +0.0022 for 04, +0.0014 and +0.0008 for 05. On a Kaggle
T4 a cycle took 42 s collecting and 2 s learning.

**Readings:**

- The wider networks (test 04) changed the outcome the most of any test so
  far: the mean action arrived in 81% of 128 episodes, never left the range
  circle, and closed 90% of the start distance. Unlike every earlier
  pink-noise run, its aim improved as it got faster: in the last quarter of
  training it ended no closer in 14% of the episodes (05: about 50%) and moved
  toward the target at 6.5 mm/s (05: 0.6). Its travel turned toward the target
  in three of four one-second pieces.
- It moves the same way as test 05, by buzzing: the legs' lift and knee
  joints flip by about 30° every 20 ms step, with no rhythm, no left-right
  alternation, and no wave along the body. It is a better-aimed buzz, not a
  gait.
- The wider critics predict the followers' returns better (explained
  variance about twice test 05's), though the followers still cannot see the
  target. The wider actors also change more per update at the same learning
  rate.
- GAE λ 0.9 (test 01) changed nothing measurable: training followed test 05's
  curves, and the evaluation's 27% arrivals are within test 05's range.
- One seed each; the comparison with test 05 mixes 128 and 32 evaluation
  episodes.

## Open questions

- Test 04's second seed, and test 05's evaluation on the same 128 episodes.
- Whether more samples per update cycle (test 03, or more) improve on test
  04.
- How to keep test 04's aim and arrivals without the buzz, for a gait: CAPS
  (night test 09 stopped the buzz with 64 × 64 networks but did not yet walk),
  or another smoothness measure.
