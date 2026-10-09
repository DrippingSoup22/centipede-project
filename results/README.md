# Results

The data behind the course report. Each study keeps the small files of every
run it uses, organised by condition and seed, with a README that says what
was compared and what was measured. The full run folders, with their
recordings and checkpoints, stay in the local `runs/` under the same folder
names; runs trained on Kaggle also stay in the outputs of that notebook's
versions.

## Layout

```text
results/
├─ README.md             This file: the layout and the index of studies
├─ reward_study/         One reward component at a time on a fixed baseline
│  ├─ README.md          The protocol, the conditions, and the results table
│  └─ 01_baseline/       One folder per condition
│     ├─ README.md       What the condition changes, its runs, its results
│     ├─ seed1/          The small files of each seed's run folder
│     └─ seed2/
└─ earlier/              Runs before the reward study that the report may use
   ├─ README.md          What each run tested and what it showed
   └─ <run folder>/      The small files of that run folder
```

Each run's folder here holds:

| File | Contents |
| --- | --- |
| `configuration.toml` | Every setting of the run |
| `run_info.json` | Its sessions: code version, device, start checkpoint |
| `training_log.jsonl` | One line per update cycle with every diagnostic: the learning curves |
| `report.html` | The training report |
| `evaluations/` | The evaluation's results (`.json`) and report (`.html`), agents and any baselines |
| `launched/` | The configuration files and console output of each launch |

Left out, and kept in `runs/<run folder>/`: the recordings (`recordings/` and
the evaluations' `.npz` files) and the checkpoints.

## Studies

| Study | What it asks | Status |
| --- | --- | --- |
| [`reward_study/`](reward_study/README.md) | Which reward components make the independent segments walk and steer, and whether a gait emerges | Baseline done |
| [`night_2026-10-09/`](night_2026-10-09/README.md) | Screening tests of smooth exploration and smoothness costs, run by the assistant overnight: what makes the policy's mean action reach the target | Done; open questions listed |
| [`earlier/`](earlier/README.md) | Far targets and the movement cost; the follower's share of the progress, on the older task | Complete |
