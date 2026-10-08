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

**Runs:** not run yet.

**Results:** to be filled in.
