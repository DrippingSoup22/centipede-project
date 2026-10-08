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
| `[run]` | Experiment | What to do (`mode`), the run's name and seed, checkpoints, the report, and which earlier run to start from, continue, or evaluate |
| `[environment]` | Environment | Episode length and observation radius |
| `[environment.simulation]` | Physics simulation | Model file, backend, number of worlds, GPU memory |
| `[environment.target]` | Environment | Where targets are placed and when the head has arrived |
| `[environment.rewards]` | Environment | Reward and cost values |
| `[agents]` | Agents | Device, networks, learning rate, normalisation |
| `[agents.ppo]` | Agents | PPO settings |
| `[interaction_loop]` | Interaction loop | Steps per window and number of update cycles |
| `[evaluation]` | Experiment | Seeds, episodes per seed, and optional baselines; evaluation files only |

The settings in each section, with their meanings and first values, are listed
in [environment.md](environment.md#settings), [agents.md](agents.md#settings),
and the [architecture](architecture.md) for the interaction loop. The `[run]` and
`[evaluation]` settings are listed below.

## Checks across sections

Each component checks its own section. The experiment adds the checks that
need two sections at once:

- The physics and the agents run on the same kind of device: `backend = "cpu"`
  requires `[agents] device = "cpu"`, and `backend = "gpu"` requires
  `device = "cuda"`. The other two combinations are rejected for now. They may
  be allowed later, for example a single world on the CPU with large networks
  on the GPU; the interaction loop would then move tensors between devices.

## Kinds of file

There are three kinds of file, told apart by their `[run]` section:

| Kind | `[run]` holds | The file contains |
| --- | --- | --- |
| Training | `mode = "train"` and the run's own settings | Every section of a new run; only the values that differ from the defaults need writing |
| Continuing | `mode = "train"` and `continue_from` | Only `[run]`, optionally with the session's `time_limit_hours`, and optionally `[interaction_loop]`, such as a larger `update_cycles` |
| Evaluation | `mode = "evaluate"` and `source` | `[run]`, `[evaluation]`, and optionally `[environment]` sections |

Continuing and evaluation files name an existing run and start from its saved
configuration: their own sections replace only the matching values, and every
result is checked by the components as usual. Any other section is rejected
with an error naming it, so nothing in such a file is silently ignored.

## Training files

`mode = "train"` starts a new run. Smoke tests, probes, the baseline, the GPU levels, and full
training all use this mode and the same code; they differ only in their values, mainly
the number of worlds, episode length, and number of update cycles.

| File | Purpose | Size | Report |
| --- | --- | --- | --- |
| `configs/smoke.toml` | Check that everything runs from start to finish | Tiny, seconds on the CPU | No |
| `configs/probe.toml` | See how a choice of settings behaves on the CPU | Light, about half an hour on the CPU | Yes |
| `configs/baseline.toml` | The first, smallest training run, against which later changes are compared | About 11 minutes on the RTX 3080 (64 worlds, 32 cycles of 64 steps, 1,024-step episodes) | Yes |
| `configs/quick.toml` | Check that a GPU run works and measure its speed | Quick, about 3 minutes on the RTX 3080 (1,024 worlds, 3 cycles of 64 steps) | Yes |
| `configs/medium.toml` | See how a choice of settings starts to behave on the GPU | Medium, about 14 minutes on the RTX 3080 (256 worlds, 8 cycles of 256 steps) | Yes |
| `configs/training.toml` | The actual experiment | Long, about 7.5 hours on the RTX 3080 (1,024 worlds, 128 cycles of 256 steps) | Yes |

The times are projected from the step times the training profile measured on
the RTX 3080 desktop, the training machine (Stage 8.4 in `plan.md`): 0.29 s
per step at 64 worlds, 0.40 s at 256, and 0.82 s at 1,024. The first run on a
machine adds about half a minute compiling the kernels.

The `[run]` settings of a training file:

| Setting | Default | Meaning |
| --- | --- | --- |
| `name` | required | Part of the run folder's name; letters, digits, `-` and `_` |
| `seed` | 0 | Starts every random sequence of the run: the agents' and the environment's |
| `checkpoint_every_cycles` | 16 | A checkpoint is saved after every this many cycles, and always after the last |
| `report` | `true` | Whether to write the run's report; the smoke test turns it off |
| `runs_folder` | `"runs"` | Where run folders are created; created if missing |
| `start_from` | none | A run folder (its latest checkpoint) or a checkpoint file: the new run's agents begin from it |
| `record_every_episodes` | 1 | One episode length of windows is recorded for replay every this many episode lengths, as one file: cycles 1 to 4, then 5 to 8, ... for 256-step episodes and 64-step windows. All worlds start together, so each file holds every world's whole episode from its first step, or several episodes for the worlds that arrived early. `0` records nothing |
| `record_levels` | 4 | Worlds are ranked by their summed reward over the window and split into this many levels |
| `record_per_level` | 8 | How many worlds of each level are kept, evenly spaced from the level's best to its worst |
| `record_selection` | `"ranked"` | `"ranked"` keeps the worlds chosen by level; `"first"` keeps the first worlds, as many, so that consecutive recordings show the same worlds |
| `time_limit_hours` | none | Bounds one training session: after each cycle, if one more cycle (at this session's average time per cycle) would end after the limit, the run saves a checkpoint and its report and stops, to be continued in a new session. A continuing file may set a new value |

A complete training file, with every section written out:

```toml
[run]
mode = "train"
name = "probe"                        # run folder: runs/<date>_<time>_<name>
seed = 1
checkpoint_every_cycles = 16
report = true
# runs_folder = "runs"
# start_from = "runs/2026-10-07_1432_easy"
# record_every_episodes = 1          # one file per episode length
# record_levels = 4
# record_per_level = 8
# record_selection = "ranked"

[environment]
max_episode_steps = 8192
observation_radius = 1

[environment.simulation]
model_path = "models/assembly_v3.xml" # required
backend = "cpu"                       # required: "cpu" or "gpu"
world_count = 4
# gpu_solver = "newton"               # GPU only: "newton" or "cg"
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
device = "cpu"                        # "cpu" or "cuda", matching the backend
hidden_layers = [64, 64]
initial_action_std = 0.5
optimizer = "adam"                    # "adam", "adamw", or "sgd"
learning_rate = 3e-4                  # the rate of the first update
learning_rate_schedule = "constant"   # "constant", "linear", or "cosine"
final_learning_rate = 3e-5            # the rate of the last update, if not constant
weight_decay = 0.0
momentum = 0.9                        # SGD only
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

### Starting from another run

A training file with `start_from` creates a **new run** whose agents begin with
another run's learned networks, optimizers, normalisers, and random
generators. Everything else is the new file's own, so the task can change: for
example, targets placed farther away once the centipede reaches near ones. The
new run counts its cycles from zero and keeps its own log and report, which
names the checkpoint it started from. Training in stages is a chain of such
runs.

Two agent settings must match the checkpoint, and a difference is rejected:
`hidden_layers`, because the networks must have the same shape, and
`optimizer`, because each kind of optimizer keeps its own state (checkpoints
from before the optimizer could be chosen count as Adam). The learning rate,
its schedule, weight decay, and momentum may differ: the new run's own values
replace those the optimizers were saved with, and its schedule counts the new
run's cycles from the first. The number of segments and the observation size must match
too; the agents check those when loading.

### Continuing a run

A continuing file resumes an existing run, for example after it stopped at its
time limit, or to train it for more cycles:

```toml
[run]
mode = "train"
continue_from = "runs/2026-10-07_1432_probe"

[interaction_loop]
update_cycles = 256                   # optional: the new total
```

Training resumes in the same folder from the latest checkpoint and counts on
from its cycle. Windows logged after that checkpoint, by a session that stopped
between checkpoints, are removed from the log and trained again. Checkpoints
hold no unfinished episodes (see [agents.md](agents.md#checkpoints)), so the
environment starts new ones, drawn with the run's seed plus the number of
completed cycles so that they differ from the run's first episodes. The saved
configuration is updated, and each session is recorded in `run_info.json`.

## Evaluation files

`mode = "evaluate"` tests a checkpoint of a saved run. The task settings come
from that run's saved configuration, so by default evaluation tests the problem
the agents were trained on. Results are written inside the evaluated run's
folder.

```toml
[run]
mode = "evaluate"
source = "runs/2026-10-07_1432_probe" # a run folder (its latest checkpoint) or a checkpoint file

[evaluation]
seeds = [101, 102, 103, 104]
episodes_per_seed = 8
baselines = ["zero", "random"]        # optional; leave out for none

[environment.simulation]              # optional: replaces the run's own values
backend = "cpu"
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `seeds` | required | Each seed resets the environment once and runs one episode in every world |
| `episodes_per_seed` | 8 | The number of worlds, so the number of episodes per seed |
| `baselines` | none | `"zero"`, `"random"`, or both, run on the same seeds as the agents |
| `record` | `true` | Write a replay recording of every world for each actor and seed |

An evaluation file may also contain `[environment]` sections, whose values
replace the run's: `backend = "cpu"` evaluates a run trained on a GPU on the
CPU, and the agents' device follows the backend; farther targets test how far
the learned walking carries. Every changed value is listed in the results. The
number of worlds cannot be changed this way, since `episodes_per_seed` sets it.
CPU and GPU physics drift apart quickly (Stage 2 of the plan measured this), so
results from the two backends are not identical.

Each world counts only its first episode; worlds that finish early start new
episodes, which are not counted. An evaluation lasts up to `max_episode_steps`
steps per seed and per actor, so long episodes on the CPU take a while.

Baselines are off unless listed: `"zero"` always sends zero actions, which shows
whether the agents do better than doing nothing; `"random"` sends uniformly
random actions, which becomes a useful comparison once the centipede moves. The
agents act with their policy's mean action, without exploration noise, and the
checkpoint is never changed.

## What a run writes

```text
runs/2026-10-07_1432_probe/
├─ configuration.toml   the complete settings; can be run again as they are
├─ run_info.json        each training session: start, first cycle, settings its file set, code version, packages, device
├─ training_log.jsonl   one line of diagnostics per window
├─ report.html          the run's summary (unless report = false)
├─ checkpoints/         cycle_0016.pt, cycle_0032.pt, ...
├─ recordings/          cycles_0001-0004.npz, cycles_0005-0008.npz, ...: replays, one episode length each
└─ evaluations/         <date>_<time>_<checkpoint>.json and .html for each evaluation,
                        and <the same>_<actor>_seed<seed>.npz recordings
```

Folders are created only when missing, and a run is never overwritten: a new run
whose folder name is taken, by a second run in the same minute, gets a numbered
suffix.

Before training, the terminal lists the run's settings. First come the six
that shape training, each with the key that sets it: worlds
(`world_count`), episode length (`max_episode_steps`), rollout window
(`rollout_window_steps`), update cycles (`update_cycles`), minibatch size
(`minibatch_size`), and epochs (`update_epochs`). Then what they add up to: the
samples each segment agent collects per update cycle (worlds × rollout window),
the gradient steps it takes on them (epochs × minibatches), and the steps per
world over the whole run (update cycles × rollout window), also in episode
lengths. Last comes every setting of the run, defaults included, with a `*` on
those the file sets itself.

The terminal then shows one line per pass, under a header, drawn again in place
as a bar of `#` and `.` fills. In training a pass is a window: the bar fills
as its steps are collected and shows `learning` during the update; the line
then holds the collecting and learning times, the time the remaining cycles
will take, and values of every step of every world (the reward per step, the
head's distance to its target, the share of steps with a body on the ground),
with the episodes that ended counted as arrivals and time-outs. The last line
gives the session's time and overall speed. In evaluation a pass is one actor
and seed: the bar fills toward the time limit, with the most time the
evaluation can still take, and jumps to full when every world's first episode
has ended; the line then holds the time it took, the time left, and its first
episodes (the share that arrived, their mean return, and the head's distance
to its target at the end).

The log and `run_info.json` hold what the reports need. Each log line has the
cycle, the world steps collected so far, and every diagnostics category,
including the bin counts of the episode histograms (`episode_distributions`,
see [diagnostics.md](diagnostics.md#episode-summary)). Each session in
`run_info.json` also lists the settings its configuration file set itself
(`settings_written`), so that a report can tell chosen values from defaults.

The **recordings** hold the poses of the recorded steps for replay in the
sibling MujocoReplay project: `recordings/cycles_AAAA-BBBB.npz` holds the
windows collected in cycles `AAAA` to `BBBB`, one episode length, with the
worlds chosen by level; the agents update after every window, so the file
marks each update as an event and each frame with the updates done before
it. The last file of a session can be shorter, where the session ended.
An evaluation's recordings hold every world of each
actor and seed, from the first episode's first step until the last world's
first episode ended. Each file carries the model, the targets as markers, and
the run's setup, so it replays on its own
(see [diagnostics.md](diagnostics.md#recordings)). With the defaults, a
training run records every episode length, 32 worlds each: about 2.3 MB per
256 steps for models v2 and v3, so 9 MB for a 1,024-step episode. While a
file is recorded, every world's poses stay in device memory; a run that would
need more than 1 GiB for it is refused, with the settings to change. To watch a run's recordings, in order, with
MujocoReplay installed:

```powershell
mujoco-replay runs\<run>\recordings\*.npz
```

The **report** is one HTML file that opens in any browser, offline. It is
refreshed at every checkpoint, so an interrupted run still has one. It works the
same for a CPU run with a few worlds and a GPU run with thousands: the worlds
are summarised on the device before anything is logged, so the page grows with
the number of windows, not of worlds.

### How the report is laid out

The report is ordered by priority, and the size of each part follows it.

- **Header.** The run's name; one line with the cycles done, world steps,
  episodes, training time, and sessions; and one chip per group of settings
  (backend and worlds, model, episode limit, targets, observation radius,
  rewards, window and cycles, network, PPO, seed). Chips that include a setting
  the configuration file set itself are highlighted; the others show defaults.
  Every setting is listed in the details.
- **1 · Results.** Numbers first, then the charts that show progress. Episode
  values are taken over whole episode lengths of windows
  (`max_episode_steps / rollout_window_steps` windows: 4 for 256-step episodes
  and 64-step windows), the blocks of cycles 1 to 4, 5 to 8, and so on, each
  within one training session. All worlds start together, so in each block
  every centipede ends at least one episode, by arriving or by running out of
  time; the first block is "first", and the large values are the last block.
  Six tiles: the arrival share (with the number of episodes); the time the
  arrivals took, exact because every episode that runs out of time lasts the
  time limit; the share of episodes that ended closer than they started,
  arrivals included; the share of steps with a body on the ground and with
  legs touching; and the segments' speed toward their goals, which rises when
  the centipede walks forward even before it steers. Each tile shows its value
  in the first block, with an arrow coloured blue when the change is for the
  better and orange when it is for the worse. Below them, the largest chart,
  **where episodes ended**: one bar per block, stacked to 100%, holding the
  share that arrived (at the bottom) and the episodes that ran out of time by
  how far from their target they ended (within 5 mm, 5–10, 10–20, 20–50, and
  50 mm or more), blue for good and red for bad. Next to it the same episodes
  of the first and the last block as counts per distance bin. Then the reward
  per step split into its terms, whose scale depends on the reward settings,
  next to how long the arrivals of the first and the last block took. These
  count episodes rather than average distances, so that a few centipedes
  that wander far away do not hide what the others did.
- **2 · Learning** (smaller). Charts over training: critic accuracy, action
  spread, and policy change (KL), each as the segments' mean with their range,
  and the learning rate when a schedule changes it.
- **3 · Behaviour** (smaller). A table of the body, one row per segment (feet,
  body, and legs touching, height, uprightness, speed, speed toward the goal),
  averaged over the last block, with a switch to the first; both views share
  their colours. Then three numbers: the heading error, the length of the
  head's path per episode, and the share of time the head is upside down.
- **4 · Run** (small). Four numbers: the time per update cycle (collecting
  and learning), the simulation speed, the training time, and the episodes.
- **Details,** behind a "Show details" button: the other learning values
  (entropy, clipped samples, critic loss, policy loss); the head's distance to
  its target as means over the worlds; each behaviour value per segment over
  training; the episodes that ended per window and the four histograms window
  by window (readable only with many worlds); the speed and the time per
  window over training; the physics' contacts, constraint rows, and solver
  iterations; every other logged value; a table of every value at the start
  and at the end; the training sessions; the run's facts; and all settings.

Charts over training run along the world steps collected, so that runs with
different numbers of worlds can be compared; hovering shows the cycle too.
Dotted vertical lines mark where a later session resumed the run. The first
window of each session also holds its start-up, such as compiling the GPU
kernels (about 50 s on a fresh Kaggle machine), so it is drawn apart and left
out of the speed's median.

Every point of an episode value is a mean over the episodes that ended in that
window, and every point of a step value a mean over its worlds and steps. When
few stand behind each point (fewer than 100 episodes, or 10,000 world steps, in
a typical window), the points are drawn faint under a moving average; with many
worlds, as on the GPU, the line is drawn as it is. Worlds that start together
also time out together, so long episodes give few windows with episode values;
the values measured at every step, such as the head's distance and the reward
per step, have a point in every window, so they lead the results.

Values are shown in readable units: millimetres, millimetres per second,
degrees, seconds of simulated time, and percentages. Percentages never go
below 0 on an axis, and uprightness is always drawn on its whole range from −1
to 1, so that small differences are not magnified. Heatmaps of values that have a better
way (a higher return, height, uprightness, or speed toward the goal; less body
contact and fewer legs touching) run from red for the worst values shown to
blue for the best, through grey at zero or at the middle; their colour key says
which end is better. Values that are neither good nor bad, such as speed or
feet on the ground, run from light to dark blue. Clicking a chart's title
opens a short explanation of what it shows and how to read it, and hovering
over any mark shows its exact value. The reward per step and the mean return
are the only values the report computes itself, as sums and means of logged
values.

### Which chart for which question

Each kind of chart is used for one kind of question:

| Question | Chart | Why |
| --- | --- | --- |
| What is the value, and is it better? | A number, with the first episode length's value (training) or each baseline's (evaluation) below it, and an arrow coloured by whether it is better | One value is read exactly as a number; a chart adds nothing to it |
| How did a value change over training? | Line over world steps | Positions along one axis are read most precisely; world steps make runs of different sizes comparable |
| Where did the episodes end? | Bars stacked to 100%: arrived, then the episodes that ran out of time by how far from the target they ended; one per episode length in training, one per actor (and seed) in an evaluation | Each episode counts once, so a few centipedes that wander far away cannot hide what the others did, and arrivals stay apart from the episodes that ran out of time |
| How did the episodes spread over a value? | Counts per bin as horizontal bars, first against last or one bar per actor; whole histograms in the details | A mean hides two groups of outcomes or a few failures |
| What makes up a total? | Stacked parts, positive parts above (or right of) zero and costs below (or left of) it, with the total marked | The reward terms add up to the reward, so stacking shows the total and what drives it at once; areas over training, one bar per actor in an evaluation |
| How does each segment behave? | Table, one row per segment, each column coloured on its own scale, with a switch between first and last, or between actors | Eight segments and eight values read best as a coloured table; shared colours make the switch show the change |
| How did each segment change over training? | Heatmap: one row per segment, one column per window, in the details | Eight lines would tangle; colour shows the pattern along the body and over time together |
| What is the exact value? | Table, in the details only | Tables are for looking values up, not for seeing patterns |

A pair plot is not used: over training every pair of values moves with time, so
its panels would suggest relationships that are not there.

### Changing the report

The layout and the explanations are declared at the start of the page's script
in `src/centipede/experiment/report_page.html` (`METRICS`, `SETTING_CHIPS`,
`RESULT_TILES` and `BEHAVIOUR_STATS`, which both reports share,
`TRAINING_LAYOUT`, `TRAINING_DETAILS`, `EVALUATION_LAYOUT`, and
`EVALUATION_DETAILS`), so changing the report means editing those lists. A value added to a diagnostics category that no layout places
appears among the details automatically. The layout is a baseline for later
work: values may be moved or resized and new ones added, but what it shows
stays.

### The evaluation report

An **evaluation report** follows the same plan as the training report, with
each actor next to the others where training puts the first episode length
next to the last; it does not need baselines. Its header lists the seeds and
episodes, and its chips mark the settings changed for the evaluation. Every
value counts each world's first episode, and every actor starts from the same
poses and targets, which the seeds decide.

- **1 · Results.** The same six tiles as in training, each holding the agents'
  value over every seed and the lowest and highest seed. When baselines are
  listed, each baseline's value follows below a line, with an arrow pointing up
  where the agents' value is higher, blue when that is better and orange when it
  is worse. Below them, **where episodes ended**: one bar per actor stacked to
  100%, with the classes and colours of training, and under it a thin bar per
  seed when there are several, which shows whether the result holds on every
  seed. Next to it, how long the arrivals took, per actor. Then where the
  reward comes from: one bar per actor, the arrival term to the right of zero
  and the three costs to the left, with their sum, the reward per step, marked
  across the bar; and the return per episode as a number.
- **2 · Behaviour** (smaller). The body table, with a switch between the
  actors that keeps each column's colours, then the heading error, the length
  of the head's path per episode, and the share of time the head is upside
  down, as numbers with the baselines below.
- **Details,** behind a "Show details" button: the four episode histograms
  with the actors side by side, each segment's return per actor, every value
  per actor (the mean over seeds, with the lowest to highest seed), the
  evaluation's facts, and the settings.

Evaluations made before the end distance was recorded (before 8 October 2026,
10:00) show everything else and say that where each episode ended was not
recorded.
