# Centipede

A cooperative reinforcement-learning project using MuJoCo for simulation and the
shared `../RL_lib` project for learning algorithms. Eight segment agents learn
independently and interact only through the shared body, their local
observations, and a shared arrival reward.

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

The project has two implementations of the same task:

- **CPU (`cpu/`), complete through its Stage 9.** Gymnasium/PettingZoo
  simulation, eight independent PPO learners, multiprocess collection,
  checkpoints, frozen evaluation, and a plan-file command line. It is preserved
  as the correctness reference; the Stage 8 baseline is tagged `cpu-stage8`.
  It showed early local adaptation, not standing or walking. See the
  [CPU README](cpu/README.md).
- **GPU (`gpu/`), Stage 1 in progress.** MuJoCo Warp batched physics, intended
  for training on Kaggle. The simulation loads and maps the model, allocates
  batched state, steps, and resets worlds. Physics validation is blocked on the
  local MX330 by an MJWarp collision-kernel parameter limit. See the
  [GPU README](gpu/README.md).

Whole-body posture, walking, navigation, and gait measurement remain future
goals, planned in the [GPU development plan](gpu/plan.md).

Version control includes source, tests, XML models, reusable TOML presets,
documentation, and launchers. `runs/`, `archive/`, virtual environments, and
generated caches remain local. Back up any checkpoints or reports that must
survive loss of this computer separately: a Git tag does not preserve ignored
files.

## Documentation

The shared scientific contract lives in `docs/`. Each implementation keeps its
own setup notes, development plan, and code reference.

| File | Owns |
| --- | --- |
| [`docs/model.md`](docs/model.md) | Frozen body, joints, mass, contacts, simulation settings, and validation |
| [`docs/control.md`](docs/control.md) | Segment agents, observations, action ownership, networks, and training architecture |
| [`docs/environment.md`](docs/environment.md) | Targets, episodes, rewards, curriculum, and evaluation |
| [`cpu/README.md`](cpu/README.md) | CPU setup, verified versions, experiment commands, and model viewer |
| [`cpu/plan.md`](cpu/plan.md) | CPU development stages, component boundaries, and completion gates |
| [`cpu/data-types.md`](cpu/data-types.md) | Named records, fields, aliases, and serialized schemas used by the CPU code |
| [`gpu/README.md`](gpu/README.md) | GPU setup, test commands, and verification record |
| [`gpu/plan.md`](gpu/plan.md) | GPU architecture, stages, and the forward roadmap |

## Project structure

```text
Centipede/
|-- docs/          Shared scientific contract: model, control, environment
|-- models/        Frozen MuJoCo XML models
|-- cpu/           CPU implementation: package, tests, presets, tools, docs
|-- gpu/           GPU implementation: package, tests, docs
|-- runs/          Ignored training and evaluation outputs
|-- .gitignore     Local and generated-file exclusions
`-- README.md      Project overview
```

Reusable RL functionality belongs in `../RL_lib`; centipede-specific code stays
here. The root-level `runs/` directory is reserved for ignored training and
evaluation results, consistent with `RL_lib`. The local `archive/` directory is
also ignored and intentionally absent from the published structure.

CPU and GPU implementations are installed independently, in separate virtual
environments. Shared scientific contracts and XML models stay at the root;
there is no shared application-code package or universal backend wrapper.

## Model files

| File | Purpose |
| --- | --- |
| [`models/assembly.xml`](models/assembly.xml) | Frozen v1 MuJoCo model |
| [`models/segment.xml`](models/segment.xml) | Isolated trunk reference model |
| [`models/head.xml`](models/head.xml) | Isolated head reference model |

The isolated head and trunk files document the component geometry. They are not
alternative training models. The interactive viewer is described in the
[CPU README](cpu/README.md#interactive-model-viewer).

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
