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

- **Model: complete and frozen.** The current body, v3, is in
  [`models/assembly_v3.xml`](models/assembly_v3.xml) and described in
  [`docs/model.md`](docs/model.md).
- **Design: agreed.** The program structure, the environment, and the agents are
  described in the documents below.
- **Code: Stage 7 complete.** The physics simulation runs batches of worlds on
  the CPU and, with MuJoCo Warp, on the GPU, and is covered by tests that pass
  locally and on a Kaggle T4. The environment builds on it: targets, episodes,
  observations, rewards, and diagnostics, tested locally. The agents, one
  independent PPO learner per segment using RL_lib's batched PPO, act, store
  their data, learn, and save, tested locally; they have not been trained yet.
  The interaction loop connects them for training and evaluation, tested
  locally with a short real run. The experiment runs everything from one
  configuration file and writes each run's log, checkpoints, and an HTML report;
  it is tested locally, and a first CPU probe run was trained, continued, and
  evaluated. Training and evaluation also record the poses of chosen worlds
  for replay in the sibling `../MujocoReplay` project, a viewer for recorded
  MuJoCo poses that is developed separately. Nothing has learned to walk yet. On 2026-10-01 the project was restarted with
  a new structure. A first implementation reached early learning on the CPU,
  with the rear segments learning to avoid ground contact, but no standing or
  walking. It is preserved in Git history (tag `cpu-stage8`) and in the local
  archive.

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
├─ configs/          TOML files for the smoke test, probe, training, and evaluation
├─ notebooks/        Kaggle notebooks that run the project on a cloud GPU
├─ benchmarks/       Speed measurements, such as the physics simulation's
│  └─ results/       Their saved results, one folder per measurement; local only
├─ tests/            Automated tests, one file per component
└─ runs/             Training and evaluation results; local only
```

`runs/`, `benchmarks/results/`, and `archive/` stay on this computer and are not
part of the repository.
Back up results that must survive separately.

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
still supports older GPUs such as the local MX330 (compute capability 6.1). The
`gpu` extra adds NVIDIA Warp (its CUDA 12 build, for the same reason) and MuJoCo
Warp. Verified on 2026-10-03 with Python 3.13.15, MuJoCo 3.12.0, MuJoCo Warp
3.14.0, Warp 1.17.0, PyTorch 2.14.1+cu126, and NumPy 2.5.3: MuJoCo Warp's GPU
arrays become PyTorch GPU tensors without copying.

To run the tests on a Kaggle T4, import
[`notebooks/kaggle_gpu_tests.ipynb`](notebooks/kaggle_gpu_tests.ipynb) into
Kaggle, select the GPU T4 accelerator, turn on internet access, and run all
cells. The notebook clones this repository and the public RL_lib and
MujocoReplay repositories, so it tests the latest pushed commits.
[`notebooks/kaggle_physics_options.ipynb`](notebooks/kaggle_physics_options.ipynb)
runs [`benchmarks/physics_options.py`](benchmarks/physics_options.py) the same
way, comparing the physics' speed and soundness under other solver settings
and timesteps.
[`notebooks/kaggle_training_profile.ipynb`](notebooks/kaggle_training_profile.ipynb)
runs [`benchmarks/training_profile.py`](benchmarks/training_profile.py), which
measures where the time of a training cycle goes, part by part, and projects how
long a configured run will take. Locally, run it with a training file before
launching a long run, for example
`python benchmarks/training_profile.py configs/probe.toml`.

## Training on Kaggle

[`notebooks/kaggle_training.ipynb`](notebooks/kaggle_training.ipynb) trains on
Kaggle's GPUs: one run per T4, so two runs at once (for example two seeds) on
the T4 x2 machine. Its first code cell holds every choice: the configuration
file, the seeds of new runs or the run folders to continue, and the session's
time limit. Start it with **Save Version → Save & Run All (Commit)**, which
runs in the background for up to 12 hours and keeps `/kaggle/working/runs` as
the version's output. A run stops cleanly with a checkpoint before the time
limit; to continue it, add the previous version's output as input and list its
run folders in the settings cell. The notebook's first cell explains each step.

## Archive

Material that is no longer active is moved to the local `archive/` folder rather
than deleted, when it records a decision, a comparison, or a result. Archived
files are never imported or used by active code. An archived idea that is revived
is rebuilt as a new version and checked against the current baseline.
