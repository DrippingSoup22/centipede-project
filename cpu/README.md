# Centipede CPU implementation

The working MuJoCo/PettingZoo implementation lives here, with its own package,
tests, presets, and launchers. Shared models, scientific documentation, and
ignored results remain at the repository root. See the
[project README](../README.md) for the goals and the documentation index, and
the [CPU plan](plan.md) for the development stages.

## Status

The RL_lib compatibility audit and the Stage 1 environment contract are complete.
Stage 2 is also complete: the internal MuJoCo simulation and public PettingZoo
parallel environment are implemented and covered by focused tests. Stage 3 has
validated the observation, reward, contact, and episode contracts with controlled
cases for the head, an interior segment, and the rear. Stage 4 is complete:
eight independent PPO learners, rollout collection, updates, and checkpoint
restoration have passing tests. Stage 5 is complete: its single-environment,
four-window smoke experiment completed 1,024 environment transitions and saved
all four learner checkpoints. Its first frozen evaluation found reduced rear
segment contact, but no checkpoint arrivals and almost no head movement on two
held-out seeds. Stage 6 is complete: a longer, eight-window run and frozen
evaluation on four held-out seeds confirmed early contact-avoidance learning in
the rear segments. Two controlled four-environment studies then showed that
shorter collection windows can produce much stronger posture learning within
the same transition budget, but did not demonstrate whole-body standing or
walking. Stage 7 is complete: synchronous serial and spawned CPU collection now
share one rollout boundary and pass parity, reset, ordering, sample-count,
failure, and cleanup checks with two and four workers. A manual two-worker smoke
run also completed 512 transitions and saved a valid eight-learner checkpoint.
Stage 8 is complete: both CPU benchmark reports were reviewed, useful execution
profiles were retained in the plan, and further CPU tuning was deliberately
deferred. Stage 9 is complete: every run now starts from one TOML plan file
whose `kind` selects training, evaluation, or comparison. GPU work now follows its own [development plan](../gpu/plan.md), with
stage numbers restarting at 1. The CPU plan remains the history and migration
reference.

The CPU environment stack uses a small internal Gymnasium `MujocoEnv` for the
physical simulation and rendering. A public PettingZoo `ParallelEnv` wraps it to
expose simultaneous actions, partial observations, and separate rewards for the
eight segment agents. MaMuJoCo is retained only as a design reference.

## Preserved CPU baseline

The `cpu-stage8` Git tag identifies the reviewed CPU implementation through
Stage 8. It preserves simulation, independent PPO learners, serial and spawned
environment pools, checkpoints, evaluation reports, and the experiment CLI.
It is a working infrastructure reference, not a claim of learned walking.
The tag preserves the original root-level layout; the working tree now keeps
this implementation under `cpu/`. The retained benchmark conclusions are in the
Stage 8 section of the [plan](plan.md).

In the presets, `rollout_window_steps` is the rollout length, `update_cycles`
is the number of collect-and-update cycles, and `max_episode_steps` is the
episode limit. Runs saved under the earlier names remain readable; see the
[plan](plan.md#configuration-names).

## Setup

The baseline was checked on Windows on 2026-09-23 with Python 3.13.15,
MuJoCo 3.12.0, Gymnasium 1.3.0, PettingZoo 1.27.0, NumPy 2.5.2, and
PyTorch 2.14.0. The tests used pytest 9.1.1 and Ruff 0.16.5.
These are the verified versions, not a guarantee that every newer version behaves
identically. The separately installed `RL_lib` was clean at commit
`6476e9a553b36bb7d94970f7959a21639c3fa4c2`; preserve that revision alongside
this baseline when reproducing it.

From this directory, using the Centipede virtual environment and with `RL_lib`
beside the repository:

```powershell
python -m pip install -e ..\..\RL_lib
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check .
python -m pytest -q
```

The sibling RL library is installed in editable mode because it is not a
published package dependency. Reinstall the editable package after relocating an
existing checkout. The Python import remains `centipede`; no learning or
simulation behavior changes with this directory layout. The automated checks do
not launch a full training or evaluation experiment.

## Running experiments

Every run starts from one TOML plan file. Its top-level `kind` says what the
file does, and every setting lives in the file, so the command takes only the
plan path:

```powershell
.\cpu\run.cmd cpu\configs\smoke.toml
```

| Plan | `kind` | What it runs |
| --- | --- | --- |
| [`configs/smoke.toml`](configs/smoke.toml) | `training` | A quick infrastructure check, without evaluation |
| [`configs/train.toml`](configs/train.toml) | `training` | The editable training profile; its `[evaluation]` seeds evaluate the new run when training ends |
| [`configs/evaluation.toml`](configs/evaluation.toml) | `evaluation` | A new evaluation of the run named by `source`, using that run's saved training settings |
| [`configs/compare_execution.toml`](configs/compare_execution.toml) | `comparison` | One training run per combination of execution settings, with a timing report |

A training plan creates a new directory under its `output_root`, saves the
resolved settings and a copy of the plan, and writes each evaluation into a
separate directory inside the run. Edit a plan for the *next* run; a saved run
keeps the settings it was trained with.

A comparison names a `base` training plan and changes some of its
`[training]`, `[learner]`, or `[rollout]` fields. Use a `[grid]` to run every
combination of listed values, or `[[variant]]` tables to name variants whose
settings change together:

```toml
kind = "comparison"
base = "train.toml"
output_root = "../../runs/comparisons"

[[variant]]
name = "window-64"
rollout.rollout_window_steps = 64
training.update_cycles = 128
```

Every variant is checked before the first one starts. Each variant trains in its
own directory; an `[evaluation]` table in the comparison evaluates every variant
as well. After each variant, `comparison_results.json` and
`comparison_report.md` are refreshed with training time, transitions per second,
final-checkpoint evaluation scores when present, and any failure, so an
interrupted comparison keeps its completed results.

`run.cmd` runs `python -m centipede` with
`%USERPROFILE%\.venvs\Centipede\Scripts\python.exe`; set `CENTIPEDE_PYTHON` to
use another interpreter. With the environment active, `python -m centipede
PLAN` works the same way, and reinstalling the editable package also provides
the `centipede-cpu PLAN` command. Plan paths are resolved from the current
directory, and paths inside a plan from that plan's folder. Nothing trains or
evaluates until you run the command.

During evaluation, each checkpoint or baseline displays one live bar per held-out
episode seed. The count advances with simulated steps; an early arrival ends the
bar at the actual step rather than displaying a false 100% completion. Earlier
one-off configurations and the previous flag-based launcher are preserved only
in the ignored local archive.

## Interactive model viewer

The preferred viewer setup is native Windows so MuJoCo uses the Windows OpenGL
driver directly. From PowerShell in the repository root, open the full model:

```powershell
& "$HOME\.venvs\Centipede\Scripts\python.exe" .\cpu\tools\view_model.py
```

Pass another XML file to inspect an isolated component:

```powershell
& "$HOME\.venvs\Centipede\Scripts\python.exe" .\cpu\tools\view_model.py .\models\head.xml
```

The Windows environment currently uses Python 3.13 and MuJoCo 3.12. Activating
it first is optional:

```powershell
& "$HOME\.venvs\Centipede\Scripts\Activate.ps1"
python .\cpu\tools\view_model.py
```

The command opens a static MuJoCo viewer with a camera framed around the model.
Use the mouse to orbit, pan, and zoom; close the window to end the command. Static
mode deliberately does not advance physics: the frozen model uses a 0.1 ms
timestep, and trying to simulate 10,000 steps per second would make this visual
inspection needlessly slow. A later frozen-policy recorder will show learned
motion without rendering during ordinary training.
