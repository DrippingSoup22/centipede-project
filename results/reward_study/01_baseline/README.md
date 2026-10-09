# 01 — Baseline

**Configuration:** `configs/reward_study/01_baseline.toml`, evaluated with
`configs/reward_study/evaluation.toml`.

```powershell
python scripts/kaggle_gpu.py run configs/reward_study/01_baseline.toml --seeds 1 2
```

**Change from the baseline:** none; this is the reference for every other
condition.

**Expected:** close to the first 32 cycles of
`earlier/2026-10-08_1603_far_01_stand_far_256` (the same reward with the spine
passive: arrivals 14% → 23%, against a straight walk's 18%), with the spine's
random bend (about 8°) and somewhat more leg contact, as in the movement-cost
runs `earlier/2026-10-08_2116_spine_02_movement_1` and `_2147_spine_03_movement_2`.

**Runs:** one Kaggle session (two T4s), 2026-10-08 22:33, code `f02b45e`:
`seed1/` = `runs/2026-10-08_2233_reward_01_baseline_seed1`, `seed2/` =
`runs/2026-10-08_2233_reward_01_baseline_seed2`. About 16 minutes of
training each, then the evaluation.

## Results

**Training**, by quarter of the 32 cycles (first → last; episode-weighted
endings):

| Measure | Seed 1 | Seed 2 |
| --- | --- | --- |
| Arrivals | 12% → 18% → 19% → 20% | 20% → 16% → 18% → 19% |
| Out of time, no closer | 21% → 32% | 6% → 14% |
| Head speed | 24.8 → 27.7 mm/s | 24.3 → 24.8 mm/s |
| Head speed toward the target | 3.8 → 2.9 mm/s | 4.9 → 4.1 mm/s |
| Heading error | 60° → 74° | 52° → 63° |
| Body on the ground | 0.7% → 0.5% | 0.7% |
| Legs touching | 11.8% → 11.1% | 11.4% → 9.7% |
| Joint movement (random commands = 1) | 0.68 → 0.69 | 0.68 |
| Legs' action spread | 0.451 → 0.434 | 0.451 → 0.433 |
| Spine's action spread | 0.439 → 0.435 | 0.439 → 0.437 |
| Critic's explained variance | 0.27 → 0.11 | 0.21 → 0.08 |

**Evaluation** (seed 101, 32 worlds, each world's first 256-step episode):

| Actor | Arrivals | Start distance closed | Head's path | Body on ground | Legs touching |
| --- | --- | --- | --- | --- | --- |
| Mean action, seed 1 | 0% | 8.6% | 8.1 mm | 3.8% | 0.3% |
| Mean action, seed 2 | 0% | 7.9% | 8.8 mm | 4.6% | 4.8% |
| Zero action (standing still) | 0% | −0.2% | 0.2 mm | 99.4% | 0% |
| Random action | 0% / 3% | −1.7% / −2.8% | 323 / 317 mm | 0.9% / 1.1% | 52% |

**Legs' sweep joints** in the evaluation of seed 1 (from its recording):
with the mean action each sweep angle varies by 4.4° (standard deviation)
and changes 0.95° per frame, with no rhythm (autocorrelation peak 0.01),
left and right legs moving together (correlation +0.36), and neighbouring
segments in phase (+0.58 at no lag). Random commands: 27° and 17° per frame.

**Reading:** the training arrivals, about 20%, are a straight walk's luck
(18%), and the aim gets worse over the run (heading error and "no closer"
endings grow). With the exploration noise off, the mean action barely moves:
the walk in training comes from the noise, not from the policy's mean.
