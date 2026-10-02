# Configuration files

Every run starts from one TOML file and one command:

```text
python -m centipede CONFIG.toml
```

The file is organised like the [architecture](architecture.md): each component
owns one section, named after it, and reads and checks only that section.
Sub-sections are used only for groups of settings that are edited together. The
exact fields may grow or change; this structure stays.

Every setting has a default equal to the first value given in the component's
document, so a file only needs the values it is about. The run folder always
saves the complete configuration with every value filled in, so nothing is
hidden afterwards.

## Sections

| Section | Owned by | Holds |
| --- | --- | --- |
| `[run]` | Experiment | What to do (`mode`), the run's name and seed, checkpoints, continuing a run |
| `[environment]` | Environment | Episode length and observation radius |
| `[environment.simulation]` | Physics simulation | Model file, backend, number of worlds, GPU memory |
| `[environment.target]` | Environment | Where targets are placed and when the head has arrived |
| `[environment.rewards]` | Environment | Reward and cost values |
| `[agents]` | Agents | Device, networks, learning rate, normalisation |
| `[agents.ppo]` | Agents | PPO settings |
| `[interaction_loop]` | Interaction loop | Steps per window and number of update cycles |
| `[evaluation]` | Experiment | Seeds, episodes, and optional baselines; evaluation files only |

The settings in each section, with their meanings and first values, are listed
in [environment.md](environment.md#settings), [agents.md](agents.md#settings),
and the [architecture](architecture.md) for the interaction loop.

## Training files

`mode = "train"` starts a new run, or continues an earlier one named by
`continue_from`. Smoke tests, probe tests, and full training all use this mode
and the same code; they differ only in their values, mainly the number of worlds,
episode length, and number of update cycles.

| File | Purpose | Size |
| --- | --- | --- |
| `configs/smoke.toml` | Check that everything runs from start to finish | Tiny |
| `configs/probe.toml` | See how a choice of hyperparameters performs | Light |
| `configs/training.toml` | The actual experiment | Full |

A complete training file, with every section written out:

```toml
[run]
mode = "train"
name = "probe"                        # run folder: runs/<date>_<name>
seed = 1
checkpoint_every_cycles = 16
# continue_from = "runs/2026-10-05_probe"

[environment]
max_episode_steps = 8192
observation_radius = 1

[environment.simulation]
model_path = "models/assembly_v2.xml" # required
backend = "cpu"                       # required: "cpu" or "gpu"
world_count = 4
# contacts_per_world = 128            # GPU only
# constraints_per_world = 512         # GPU only

[environment.target]
distance_range_m = [0.010, 0.020]
bearing_range_deg = [-15, 15]
arrival_radius_m = 0.001

[environment.rewards]
arrival_reward = 1.0
efficiency_cost = 0.003
body_contact_cost = 0.010
leg_contact_cost = 0.005
distance_ratio_epsilon_m = 1e-6

[agents]
device = "cpu"                        # "cpu" or "cuda"
hidden_layers = [64, 64]
initial_action_std = 0.5
learning_rate = 3e-4
normaliser_epsilon = 1e-8
observation_clip = 10.0

[agents.ppo]
discount = 0.999
gae_lambda = 0.95
clip_ratio = 0.2
update_epochs = 4
minibatch_size = 64
max_gradient_norm = 0.5
entropy_coefficient = 0.001

[interaction_loop]
rollout_window_steps = 256
update_cycles = 128
```

## Evaluation files

`mode = "evaluate"` tests a saved run. The task settings (environment, rewards,
episode length) are read from that run's saved configuration, so evaluation
always tests the problem the agents were trained on. Results are written inside
the evaluated run's folder.

```toml
[run]
mode = "evaluate"
source = "runs/2026-10-05_probe"
checkpoint = "latest"

[evaluation]
seeds = [101, 102, 103, 104]
episodes_per_seed = 1
baselines = ["zero"]                  # optional; leave out for none
```

Baselines run on the same seeds for comparison. They are off unless listed:
`"zero"` always sends zero actions, which shows whether the agents do better than
doing nothing; `"random"` sends uniformly random actions, which becomes a useful
comparison once the centipede moves.
