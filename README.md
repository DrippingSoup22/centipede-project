# Centipede

A reinforcement-learning project in which a simulated centipede learns to walk.
Each of its eight body segments is an independent agent that controls only its
own pair of legs. The segments never share networks or data: they cooperate only
through the shared body, what each can feel of its neighbours, and a shared
reward when the head reaches a target.

The physics runs in [MuJoCo](https://mujoco.org/), on the CPU or, with MuJoCo
Warp, on the GPU. The learning algorithms come from the sibling `../RL_lib`
library, and recorded runs are watched with the sibling `../MujocoReplay`
viewer.

## Goals

1. Learn to walk and to steer the head toward targets on flat ground.
2. See whether the legs develop a wave-like rhythm that travels along the body,
   as in a real centipede.

Walking ability, navigation, and resemblance to a real gait are evaluated
separately.

## Status

- **Model: complete and frozen.** The current body, v4, is in
  [`models/assembly_v4.xml`](models/assembly_v4.xml) and described in
  [`docs/model.md`](docs/model.md).
- **Design: agreed.** The program structure, the environment, and the agents are
  described in the documents below.
- **Code: complete.** The physics simulation runs batches of worlds on the CPU
  and, with MuJoCo Warp, on the GPU. The environment builds on it: targets,
  episodes, observations, rewards, and diagnostics. The agents, one independent
  PPO learner per segment using RL_lib's batched PPO, act, store their data,
  learn, and save. The interaction loop connects them for training and
  evaluation, and the experiment runs everything from one configuration file and
  writes each run's log, checkpoints, report, and recordings for replay in the
  sibling `../MujocoReplay` viewer. A run is described by one configuration file
  and runs on the CPU or on an NVIDIA GPU. A first implementation, built before
  the project was restructured, is preserved in Git history (tag `cpu-stage8`).

## Documentation

| Document | Describes |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | How the program is organised: layers, components, and how they connect |
| [`docs/model.md`](docs/model.md) | The centipede's body: dimensions, joints, motors, mass, contacts, physics settings |
| [`docs/configuration.md`](docs/configuration.md) | The TOML files that start every run, and what a run writes: run folder, log, checkpoints, and report |
| [`docs/environment.md`](docs/environment.md) | The task: physics simulation, actions, observations, targets, episodes, rewards |
| [`docs/agents.md`](docs/agents.md) | The learners: independent segment agents, networks, PPO settings, checkpoints |
| [`docs/diagnostics.md`](docs/diagnostics.md) | What each component measures about a run, for the training log and the report |

## Project folders

```text
Centipede/
├─ docs/             Architecture, model, environment, and agents documents
├─ models/           Frozen MuJoCo XML models
├─ src/centipede/    Program code, one folder per component
├─ configs/          Example TOML files: smoke test, probe, training levels, evaluation
├─ benchmarks/       Speed measurements, such as the physics simulation's
│  └─ results/       Their saved results, one folder per measurement; local only
├─ tests/            Automated tests, one file per component
└─ runs/             Training and evaluation results; local only
```

`runs/` and `benchmarks/results/` are written on the computer that runs them and
are not part of the repository.

## Setup

A fresh Windows computer first needs Python 3.13, Git, and, for the GPU, a
current NVIDIA driver. Install the first two with `winget install
Python.Python.3.13` and `winget install Git.Git` (or from python.org, ticking
"Add python.exe to PATH", and git-scm.com), then open a new PowerShell. Clone
this repository, `RL_lib`, and `MujocoReplay` side by side into one folder,
since the setup installs the other two from `..\RL_lib` and `..\MujocoReplay`.

One Python environment runs everything: the CPU and GPU simulation backends,
the learning code, and the tests. It lives outside the repository. From the
repository root, in PowerShell (`py -3.13` is Windows' Python launcher, which
works even where plain `python` is not found):

```powershell
py -3.13 -m venv "$HOME\.venvs\Centipede"
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pip install "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pip install -e ..\RL_lib
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pip install -e ..\MujocoReplay
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pip install -e ".[gpu,dev]"
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pybind11_stubgen mujoco -o typings --ignore-all-errors
& "$HOME\.venvs\Centipede\Scripts\python.exe" -m pytest
```

MuJoCo's functions are compiled code without type descriptions, so editors
cannot see them. The `pybind11_stubgen` line generates those descriptions into
the ignored `typings/` folder, where the editor's type checker finds them.
In the editor, select `$HOME\.venvs\Centipede\Scripts\python.exe` as the
Python interpreter.

PyTorch is installed first from its CUDA 12.6 build, so that tensors can live on
the GPU; plain `pip install torch` gives a CPU-only build on Windows. This build
still supports older GPUs such as the MX330 (compute capability 6.1). The
`gpu` extra adds NVIDIA Warp (its CUDA 12 build, for the same reason) and MuJoCo
Warp. Verified with Python 3.13.15, MuJoCo 3.12.0, MuJoCo Warp
3.14.0, Warp 1.17.0, PyTorch 2.14.1+cu126, and NumPy 2.5.3: MuJoCo Warp's GPU
arrays become PyTorch GPU tensors without copying.

Two benchmarks measure speed on a GPU:
[`benchmarks/training_profile.py`](benchmarks/training_profile.py) measures
where the time of a training cycle goes, part by part, and projects how long a
configured run will take; run it with a training file before launching a long
run, for example
`python benchmarks/training_profile.py configs/baseline.toml --worlds 64 256 1024`.
[`benchmarks/physics_options.py`](benchmarks/physics_options.py) compares the
physics' speed and soundness under other solver settings and timesteps.

## Running

Every run starts from one TOML configuration file, given to the program's
entry point from the repository root:

```powershell
python -m centipede configs/smoke.toml
```

The file chooses everything about the run, including where it runs: the
physics simulation on the CPU or on an NVIDIA GPU (`backend` in
`[environment.simulation]`) and the learning on the CPU or on the GPU
(`device` in `[agents]`). The files in `configs/` are examples to copy and
change: a smoke test that finishes in seconds on the CPU, a light probe on the
CPU, training runs of increasing length on the GPU, and an evaluation of a
trained run. [`docs/configuration.md`](docs/configuration.md) lists every
setting, and what a run writes into its folder under `runs/`. A GPU older than
Volta (compute capability 7.0) cannot compile MuJoCo Warp's default solver;
the conjugate-gradient solver (`gpu_solver = "cg"`) runs on it, slowly.
