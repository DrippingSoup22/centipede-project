# Centipede

A cooperative reinforcement-learning project using Gymnasium and MuJoCo for
simulation, PettingZoo for the parallel multi-agent interface, and the shared
`../RL_lib` project for learning algorithms.

## Goals

1. Learn to walk and navigate toward planar targets.
2. Develop walking that resembles the propagating wave-like movement of a real
   centipede.

Walking ability, navigation performance, and biological gait resemblance will
be evaluated separately.

## Current stage

The physical model is complete and frozen as the **v1 flat-ground baseline**.
The canonical model is [`models/assembly.xml`](models/assembly.xml). The accepted
design and validation results are recorded in [`docs/model.md`](docs/model.md).

V1 has:

- One rounded head and seven trunk units.
- Sixteen two-link legs.
- Active, spring-centered spine yaw and passive spine pitch.
- 69 position coordinates, 68 velocity coordinates, and 55 torque motors.
- 643.4 mg total mass and MuJoCo-derived inertia.
- A validated 0.1 ms physics timestep and separated ground/self friction.
- Collision-free, zero-mass visual membranes at the seven body seams.

The frozen XML SHA-256 is
`92143cf54b030856e436a1de6f4333c327ffb9a5f812a0e4a0b4c4aed107d50e`.
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
deferred. Stage 9 targets GPU physics and learning on Kaggle through incremental
reviews named Stage 1 bis onward. The first review concerns the physical contract
and simulation boundary. Stage 10 then applies the validated execution profile to
whole-body posture. Walking, navigation, control variants, and gait measurement
follow separately.

The agreed environment stack uses a small internal Gymnasium `MujocoEnv` for the
physical simulation and rendering. A public PettingZoo `ParallelEnv` wraps it to
expose simultaneous actions, partial observations, and separate rewards for the
eight segment agents. MaMuJoCo is retained only as a design reference.

## Preserved CPU baseline

The `cpu-stage8` Git tag identifies the reviewed CPU implementation through
Stage 8. It preserves simulation, independent PPO learners, serial and spawned
environment pools, checkpoints, evaluation reports, and the experiment CLI.
It is a working infrastructure reference, not a claim of learned walking.
GPU migration will start separately; no GPU branch or backend is part of this
baseline.

Version control includes source, tests, XML models, reusable TOML presets,
documentation, and launchers. `runs/`, `archive/`, virtual environments, and
generated caches remain local. Back up any checkpoints or reports that must
survive loss of this computer separately: a Git tag does not preserve ignored
files. The retained benchmark conclusions are in the Stage 8 section of the plan.

The current configuration names are preserved for compatibility with saved runs:
`steps_per_environment` is the rollout length, `rollout_windows` is the number of
collect/update cycles, and `max_episode_steps` is the episode limit. Their planned
renaming belongs to the next development phase.

### Verified local setup

The baseline was checked on Windows on 2026-09-23 with Python 3.13.15,
MuJoCo 3.12.0, Gymnasium 1.3.0, PettingZoo 1.27.0, NumPy 2.5.2, and
PyTorch 2.14.0. The tests used pytest 9.1.1 and Ruff 0.16.5.
These are the verified versions, not a guarantee that every newer version behaves
identically. The separately installed `RL_lib` was clean at commit
`6476e9a553b36bb7d94970f7959a21639c3fa4c2`; preserve that revision alongside
this baseline when reproducing it.

With the desired virtual environment active and `RL_lib` beside this repository,
install both projects from the Centipede directory:

```powershell
python -m pip install -e ..\RL_lib
python -m pip install -e ".[dev]"
python -m pip check
python -m ruff check .
python -m pytest -q
```

The automated checks do not launch a full training or evaluation experiment.

## Documentation

The active design is intentionally limited to four design documents, supported
by one code-facing data reference:

| File | Owns |
| --- | --- |
| [`docs/plan.md`](docs/plan.md) | Development stages, component boundaries, and completion gates |
| [`docs/model.md`](docs/model.md) | Frozen body, joints, mass, contacts, simulation settings, and validation |
| [`docs/control.md`](docs/control.md) | Segment agents, observations, action ownership, networks, and training architecture |
| [`docs/environment.md`](docs/environment.md) | Targets, episodes, rewards, curriculum, and evaluation |
| [`docs/data-types.md`](docs/data-types.md) | Named records, fields, aliases, and serialized schemas used by the code |

[`AGENTS.md`](AGENTS.md) contains working directives for Codex.

## Project structure

```text
Centipede/
|-- configs/                 Reusable training and evaluation presets
|-- docs/                    Design documents and the data-type reference
|-- models/                  Frozen MuJoCo XML models
|-- src/centipede/
|   |-- environment/         Simulation, observations, rewards, and public task
|   |-- training/            Independent learners and rollout coordination
|   `-- experiment/          Configuration, runner, evaluation, and reports
|-- tests/environment/       Environment component and contract checks
|-- tests/training/          Learner, rollout, checkpoint, and worker checks
|-- tests/experiment/        Experiment, runner, and report checks
|-- tools/                   Terminal entry points and small project utilities
|-- runs/                    Ignored training and evaluation outputs
|-- run.cmd                  Short Windows experiment launcher
|-- pyproject.toml  Package, dependency, and tool configuration
|-- .gitignore  Local and generated-file exclusions
|-- AGENTS.md   Working instructions for Codex
`-- README.md   Project overview
```

Reusable RL functionality belongs in `../RL_lib`; centipede-specific code stays
here. The root-level `runs/` directory is reserved for ignored training and
evaluation results, consistent with `RL_lib`. The local `archive/` directory is
also ignored and intentionally absent from the published structure.

The source package is grouped by responsibility. `environment/` owns the complete
task boundary, while `training/` connects that task to independent RL_lib
learners. `experiment/` owns validated presets, the training runner, frozen
evaluation, and HTML reports. The terminal-facing launcher lives in
`tools/run_experiment.py` and only routes user requests into that application
logic. Frozen checkpoint evaluation is available; policy recording remains
future work.

## Training and evaluation commands

The reusable presets are [`configs/smoke.toml`](configs/smoke.toml) for a quick
training check, [`configs/train_cpu.toml`](configs/train_cpu.toml) for the editable
CPU training profile, and
[`configs/evaluation.toml`](configs/evaluation.toml)
for held-out evaluation seeds. The CPU timing matrix lives in
[`configs/benchmark_cpu.toml`](configs/benchmark_cpu.toml). Edit a preset for the *next* experiment; the
resolved training settings are saved with every run. From PowerShell in this
directory:

```powershell
.\run.cmd --train configs\smoke.toml
.\run.cmd --train configs\train_cpu.toml --evaluate configs\evaluation.toml
.\run.cmd --evaluate configs\evaluation.toml --source runs\train_cpu\<run-id>
.\run.cmd --benchmark configs\benchmark_cpu.toml
```

Every option has a short form: `-t`, `-e`, `-s`, or `-b`. `-h`/`--help` displays usage.
Training alone creates a new directory under the preset's `output_root`; training
plus evaluation evaluates that new run; evaluation alone reads the saved training
settings and checkpoints from `--source`. Each evaluation has a separate report
directory inside its source run. The launcher uses
`%USERPROFILE%\.venvs\Centipede\Scripts\python.exe` by default. Set
`CENTIPEDE_PYTHON` to another interpreter path if needed. It does not start a
training or evaluation until you invoke it.

Benchmarking also calls the ordinary training runner. It first compares valid
worker layouts, carries each environment count's fastest layout into the rollout
comparison, and tests episode limits only at the largest environment count. The
launcher records wall time and transition throughput after every case in JSON
and Markdown, so an interrupted benchmark keeps its completed measurements.

During evaluation, each checkpoint or baseline displays one live bar per held-out
episode seed. The count advances with simulated steps; an early arrival ends the
bar at the actual step rather than displaying a false 100% completion.

The active workflow uses only these purpose-based presets. Every new run stores
its resolved settings, so later edits to a reusable TOML do not change how that
run is evaluated. Earlier one-off configurations and their legacy evaluation
entry point are preserved only in the ignored local archive.

The sibling RL library is installed separately in editable mode during local
development because it is not a published package dependency:

```powershell
python -m pip install -e ..\RL_lib
```

## Model files

| File | Purpose |
| --- | --- |
| [`models/assembly.xml`](models/assembly.xml) | Frozen v1 MuJoCo model |
| [`models/segment.xml`](models/segment.xml) | Isolated trunk reference model |
| [`models/head.xml`](models/head.xml) | Isolated head reference model |

The isolated head and trunk files document the component geometry. They are not
alternative training models.

## Interactive model viewer

The preferred viewer setup is native Windows so MuJoCo uses the Windows OpenGL
driver directly. From PowerShell in the project directory, open the full model:

```powershell
& "$HOME\.venvs\Centipede\Scripts\python.exe" .\tools\view_model.py
```

Pass another XML file to inspect an isolated component:

```powershell
& "$HOME\.venvs\Centipede\Scripts\python.exe" .\tools\view_model.py .\models\head.xml
```

The Windows environment currently uses Python 3.13 and MuJoCo 3.12. Activating
it first is optional:

```powershell
& "$HOME\.venvs\Centipede\Scripts\Activate.ps1"
python .\tools\view_model.py
```

The command opens a static MuJoCo viewer with a camera framed around the model.
Use the mouse to orbit, pan, and zoom; close the window to end the command. Static
mode deliberately does not advance physics: the frozen model uses a 0.1 ms
timestep, and trying to simulate 10,000 steps per second would make this visual
inspection needlessly slow. A Stage 10 frozen-policy recorder will show learned
motion without rendering during ordinary training.

## Local archive policy

Move an artifact to `archive/` when it is no longer part of active training or
documentation. Preserve it instead of deleting it when it records a meaningful
decision, comparison, source, or result. Generated caches and local environments
do not need archival. The root `archive/` directory is intentionally ignored by
Git and is not part of the published project.

Archived files must not be imported, loaded, or used as defaults by active code.
If an archived idea is revived, copy or reimplement it as a new version and
validate it against the current baseline rather than editing history in place.
Before committing, clean the active tree and move meaningful superseded material
to the local archive. Commits and pushes are performed only after explicit user
approval.
