# Reward study

Decided with the user on 2026-10-09. The reward is built up from a baseline
that keeps only the components it is certain to need, adding one component
at a time, so that each comparison shows what that component changes.
Everything else stays fixed for the whole study; a fixed setting changes only
for a stated reason, and then the baseline is run again.

## Protocol

**Fixed settings** (`configs/reward_study/`):

- Body: model v3 with spine control (every segment but the rear commands the
  spine joint behind it).
- Task: targets 30–60 mm away within ±30°, arrival when the target lies under
  the head, range circle 2.5 × the start distance, 256-step episodes (5.12 s).
- Agents: the default networks and PPO settings, one independent learner per
  segment, starting from the standing prototype
  (`runs/2026-10-07_2221_baseline_stage2`).
- Training: 64 worlds, rollout window 64 steps, minibatch 512, 4 epochs, 32
  update cycles (2,048 steps per world).
- Seeds 1 and 2 for every condition, trained in one Kaggle session (one T4
  each). Each run is evaluated with its policy's mean action, next to the
  zero (standing still) and random baselines, on seed 101, 32 worlds.

**Baseline reward** (`docs/environment.md`, Rewards, rules R0–R4): arrival 1;
the head's progress, shared by every segment; a step cost, body contact, and
leg contact, with the cost budget split 2 : 3 : 1.

**Measures**, per seed, over the first and the last quarter of the 32 cycles:
arrivals, the head's speed toward the target, body on the ground, legs
touching, the joints' movement per step (in units of random commands), and
the legs' and the spine's action spread. From the recordings: how often the
travel turns toward the target's side, and how often the head comes within
10 mm of the target. From the evaluation: arrivals and the head's path with
the mean action, against the baselines. The gait measures (phase lag between
neighbouring segments, left–right correlation) are to be written.

## Conditions

| # | Condition | Change from the baseline | Status |
| --- | --- | --- | --- |
| 01 | [Baseline](01_baseline/README.md) | – | To run |
| 02 | Command cost | A cost on the size of each command, `‖a‖²`, as in Gymnasium's Ant | Planned |
| 03 | Colored exploration noise | Exploration noise correlated over time, `1/f^β` with β = 0.5 (the PPO default of Hollenstein et al., AAAI 2024), in RL_lib | Planned |
| – | gSDE | State-dependent exploration noise (Raffin et al., CoRL 2021), only if colored noise is not enough | Fallback |
| – | Dense steering | A reward for the head turning toward the target | Proposed |
| – | Dense progress | More weight on the progress relative to the arrival | Proposed |

The reasons for these candidates, with sources, are in
`claude-notes/reward-benchmarks.md` (local).

## Results

To be filled in as conditions finish: the mean of the two seeds, with each
seed's value.
