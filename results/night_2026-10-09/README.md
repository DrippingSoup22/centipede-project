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

## Tests

| # | Change | Seeds | Machine | Status |
| --- | --- | --- | --- | --- |
| 01 | 256 worlds, minibatch 2048 | 1 | 3080 | Running |
| 02 | Colored noise, beta 0.5 | 1 | 3080 | Planned |
| 03 | Pink noise, beta 1 | 1, 2 | Kaggle | Running |
| 04 | Command cost, ratio 4 | 1 | 3080 | Planned |

## Results

The reference is the reward-study baseline on Kaggle (64 worlds, seeds 1 and
2; `runs/2026-10-08_2233_reward_01_baseline_seed1`, `_seed2`). Its mean action
arrived in none of the 32 evaluation episodes; its training arrivals ended at
19-20%.
