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
parallel environment are implemented and covered by focused tests. Stage 3 will
validate the observation and reward behavior more deeply; no learned policy
exists yet.

The agreed environment stack uses a small internal Gymnasium `MujocoEnv` for the
physical simulation and rendering. A public PettingZoo `ParallelEnv` wraps it to
expose simultaneous actions, partial observations, and separate rewards for the
eight segment agents. MaMuJoCo is retained only as a design reference.

## Documentation

The active design is intentionally limited to four documents:

| File | Owns |
| --- | --- |
| [`docs/plan.md`](docs/plan.md) | Development stages, component boundaries, and completion gates |
| [`docs/model.md`](docs/model.md) | Frozen body, joints, mass, contacts, simulation settings, and validation |
| [`docs/control.md`](docs/control.md) | Segment agents, observations, action ownership, networks, and training architecture |
| [`docs/environment.md`](docs/environment.md) | Targets, episodes, rewards, curriculum, and evaluation |

[`AGENTS.md`](AGENTS.md) contains working directives for Codex.

## Project structure

```text
Centipede/
|-- configs/    Versioned experiment configurations
|-- docs/       The four active design documents
|-- models/     Frozen MuJoCo XML models
|-- src/        Installable Centipede Python package
|-- tests/      Public behavior and boundary checks
|-- tools/      Small project utilities
|-- runs/       Ignored training and evaluation outputs
|-- pyproject.toml  Package, dependency, and tool configuration
|-- .gitignore  Local and generated-file exclusions
|-- AGENTS.md   Working instructions for Codex
`-- README.md   Project overview
```

Reusable RL functionality belongs in `../RL_lib`; centipede-specific code stays
here. The root-level `runs/` directory is reserved for ignored training and
evaluation results, consistent with `RL_lib`. The local `archive/` directory is
also ignored and intentionally absent from the published structure.

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
inspection needlessly slow. Policy motion will instead be rendered by the future
evaluation recorder after the environment and checkpoint interfaces exist.

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
