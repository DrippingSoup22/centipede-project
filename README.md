# Centipede

A cooperative reinforcement-learning project using MuJoCo for simulation and
the shared `../RL_lib` project for learning algorithms.

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
The RL_lib compatibility audit is complete. Its continuous PPO core already
provides the generic learning operations needed by eight independent segment
learners. Centipede must now define its own environment and experiment runner;
no learned policy exists yet.

## Documentation

The active design is intentionally limited to three documents:

| File | Owns |
| --- | --- |
| [`docs/model.md`](docs/model.md) | Frozen body, joints, mass, contacts, simulation settings, and validation |
| [`docs/control.md`](docs/control.md) | Segment agents, observations, action ownership, networks, and training architecture |
| [`docs/environment.md`](docs/environment.md) | Targets, episodes, rewards, curriculum, and evaluation |

[`AGENTS.md`](AGENTS.md) contains working directives for Codex.

## Project structure

```text
Centipede/
|-- docs/       The three active design documents
|-- models/     Frozen MuJoCo XML models
|-- tools/      Small project utilities
|-- .gitignore  Local and generated-file exclusions
|-- AGENTS.md   Working instructions for Codex
`-- README.md   Project overview
```

Reusable RL functionality belongs in `../RL_lib`; centipede-specific code stays
here. A root-level `runs/` directory will be created when training begins and is
reserved for training and evaluation results, consistent with `RL_lib`.

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
