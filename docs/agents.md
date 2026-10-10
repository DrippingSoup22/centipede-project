# Agents

The agents are the learners: one **segment agent** for each body segment (eight
in v1), managed as one group. They sit on layer 2 of the
[architecture](architecture.md), next to the environment, and never talk to it
directly; the interaction loop passes data between them. The agents know nothing
about MuJoCo, targets, or how observations and rewards are calculated.

Learning uses **PPO** from the reusable package in `../RL_lib/src/rl_lib`, which
also provides networks, policies, normalisation, and advantage estimation. The
experiment runners in `../RL_lib/experiments` are not used; this project has its
own.

RL_lib's PPO originally handled one observation at a time through NumPy. For
this project it is extended with a batched, tensor-based form that works on any
device, as described in `../RL_lib/docs/batched-ppo.md`. A second round there gave
each PPO its own random generator and checkpoint methods, reports how well the
critic predicted the returns, and removed every point where acting or learning
made the CPU wait for the GPU. An external library
such as Stable-Baselines3 was considered and not chosen: its PPO runs its own
training loop as a single agent, keeps collected data in NumPy on the CPU, clips
actions to their bounds, and trains actor and critic with one combined loss,
all of which conflict with this design.

## Interface

`W` is the number of worlds and `N` the number of segments. Everything passed in
and out is a PyTorch tensor on the agents' device. The group is created from
the environment's `segment_count`, `observation_size`, `world_count`, and
`segment_action_sizes`, the interaction loop's `rollout_window_steps`, which
sizes the storage, and the run's seed. It works for any number of segments.
Each segment agent has as many actions as its segment takes: six for its
legs, or with leg clocks twelve, six per leg (its clock's tempo and five
values of its step shape); with spine control, the spine joint behind it, for
every segment but the rear (only the head with a passive follower spine); and
with a clock per segment, last, its clock's tempo
([environment.md](environment.md#actions-and-timing)). The environment's
`tempo_columns` and `motor_columns` tell the group which of each agent's
actions set clock tempos and which sets each of its motors, which only its
learning diagnostics need: a segment's tempo spread is the mean over its
tempo actions, and with leg clocks its leg motors' spreads are those of its
joints' centres.

| Operation | Takes | Returns or does |
| --- | --- | --- |
| `act(observations, training)` | Observations `(W, N, observation_size)` | The joint action `(W, N, action_size)`, where a segment with fewer actions gets zeros as padding; when `training`, each segment agent also updates its normaliser and remembers what it needs for learning |
| `record(rewards, terminated, truncated, final_observations)` | The environment's results for the step just taken | Each segment agent stores its own part |
| `update()` | Nothing | Each segment agent learns from its own stored data, then clears it |
| `state_dict()`, `load_state_dict(state)` | — | All segment agents' state, for checkpoints; the experiment writes it to a file |

During evaluation, `act` is called with `training` off: each segment agent
returns its policy's mean action, without changing its normaliser or
remembering anything, and nothing is stored or learned. The mean shows what
the policy has learned, without the exploration noise added in training. For
comparison, the front file also provides two baselines
with the same `act`: one that always returns zero actions and one that returns
uniformly random actions.

## Files

The agents live in `src/centipede/agents/`.

| File | Covers |
| --- | --- |
| `agents.py` | Front file: creates one segment agent per segment, gives each its slice of the observations, joins their actions, and passes on recording, learning, saving, and loading; also provides the zero and random baselines |
| `segment_agent.py` | One segment's networks, optimizers, observation normaliser, and PPO update |
| `rollout_storage.py` | One segment agent's collected data |
| `settings.py` | Agent settings |
| `diagnostics.py` | The learning category of [diagnostics.md](diagnostics.md#learning), filled once after each update |

## Independence

Each segment agent has its own actor and critic networks, optimizers,
observation normaliser, stored data, advantages, and losses. Nothing is ever
shared or pooled between segments: no shared networks or critic, no combined
losses or gradients, and no messages between agents. They cooperate only through
the shared body and the shared arrival reward, which is simply copied to each
agent.

From any one agent's point of view, the others are part of its environment, and
they keep changing as they learn. This is a known difficulty of independent
learners and part of what this project studies.

Each segment agent also has its own random generator, inside its PPO, for
sampling actions and shuffling its data. Its seed, which also sets its starting
weights, is `seed * segment_count + segment_index`, so a run's seed repeats
every agent's choices and no two pairs of run seed and segment share a seed.

Each segment agent evaluates its own networks. Evaluating all of them together
in one operation, keeping separate parameters, was considered: one agent's
action for 1,024 worlds takes about 1 ms on the GPU, against seconds of physics
per step, so it is not worth the added complexity.

## Observations

Every segment agent has the same input size, `observation_size`: 83 with the
first observation radius of 1, and more with spine control, clocks of either
kind (with leg clocks, also the step shape in use), or neighbour clocks (see
[environment.md](environment.md#what-each-segment-sees)). Inputs that are always
zero for a segment, such as the head's missing neighbour ahead, never affect its
network.

Each segment agent normalises its own observations with running statistics
before they enter its networks. The statistics are updated only when an
observation is used to choose an action during training, and the stored data
keeps the exact normalised observation that was used. Bootstrapping and
evaluation use the statistics without changing them.

## Collecting and learning

Each segment agent stores, for every step of every world, its observation,
action, action probability, value estimate, reward, and episode-end signals. The
storage is allocated **once**, when the run starts, with room for
`W × rollout_window_steps` steps. It lives on the same device as the agent's
networks and is filled in place, so on the GPU the data never leaves graphics
memory. After each update it is reused, not reallocated.

All agents keep their policies unchanged while a window is collected, then each
updates from its own storage only. Updating at the same moment shares no
information; it only prevents some agents from changing while others are still
collecting.

How a stretch of stored steps ends decides how its returns are estimated:

| The stretch ended because | The agent uses |
| --- | --- |
| The target was reached (terminated) | A final value of zero |
| The time limit was reached (truncated) | Its critic's value of the final observation |
| The window ended mid-episode | Its critic's value of the next observation |

One value covers all three cases. The environment's `final_observations` hold,
for every world, the observation right after the step: the next one if the
episode continues, the last one if it ended. When recording a step, each segment
agent stores its critic's value of that observation, normalised without
updating the statistics. Advantage estimation ignores it on terminated steps, so
those get a final value of zero.

Advantages are computed separately for every uninterrupted stretch in each
world, then all stretches of the same segment are used together. A window ending
never resets the environment; the episode continues in the next window.

## Checkpoints

A checkpoint is one bundle holding, for every segment agent, its PPO state
(networks, optimizers, and random generator), normaliser, and training counters,
plus the settings, the number of segments, the observation size, and each
segment's number of actions. It is saved
only after a completed update. A random generator is restored only on the same
kind of device it was saved on, since CPU and CUDA generators draw different
sequences; elsewhere, for example when a run trained on a GPU is evaluated on
the CPU, each agent keeps the sequence its seed gives. Loading it into an environment with a different number of
segments, observation size, or numbers of actions fails with a clear error,
except when a new run starts from it (below). It holds no stored data,
physical state, targets, or unfinished episodes, so continuing from a checkpoint
starts new episodes. The optimizers' saved state also brings back the
settings they were saved with, so after loading, each agent applies its own
weight decay and momentum, and the experiment sets the learning rate before
every update. A new run that starts from a checkpoint must use the same hidden
activation, since trained weights mean something else under another one, and
the same kind of optimizer (see
[configuration.md](configuration.md#starting-from-another-run)).

**Widening.** A new run may give its agents more to see or to command than the
run it starts from, as spine control does: two more inputs, after all the
others, and a seventh action for every segment but the rear. Its agents are
then widened from the checkpoint. Every saved weight keeps its place; the
weights of the new inputs and of the new action start at zero, so at the first
step each agent acts exactly as the saved one did and commands zero to its new
motor, which is what the spine received before. The new action's spread starts
at `initial_action_std`. The saved optimizer state no longer fits the larger
networks, so the optimizers start fresh, and so does the random generator,
from the run's seed. The normaliser keeps its statistics and gives each new
input a mean of 0 and a spread of 1; since the saved statistics count
millions of observations, the new inputs keep about that scale, the spine's
angle in radians and its speed in radians per second.

A new run may also give its agents wider hidden layers: as many layers as
before, each at least as wide. Every saved weight again keeps its place, and
no saved unit takes anything from a new unit of the layer below. A new hidden
unit keeps the random weights a fresh network gives it from the layer below,
so that it can learn, while the layer above takes nothing from it yet: the
network computes exactly what the saved one did, and the new units grow into
use as training goes on. This is the function-preserving growth of Net2Net
(Chen, Goodfellow & Shlens, ICLR 2016), which starts a larger network from a
trained smaller one instead of from scratch; a new unit with zero weights on
both sides would never learn, since no gradient would reach it. Continuing or
evaluating a run never widens: the sizes must match.

## Settings

These are the keys of the agents sections of the configuration file (see
[configuration.md](configuration.md)). Each first value is also the default.

| Setting | First value | Meaning |
| --- | ---: | --- |
| **`[agents]`** | | |
| `device` | `cpu` | Where networks and stored data live: `cpu` or `cuda` |
| `hidden_layers` | [64, 64] | Hidden layer sizes of both actor and critic; a run that starts from another may make them wider, never fewer or narrower ([Checkpoints](#checkpoints)) |
| `hidden_activation` | `relu` | The hidden layers' activation in actor and critic: `relu` or `tanh`, which Andrychowicz et al. (ICLR 2021) found best for networks of this size, and ReLU worst. A run that starts from another must keep its activation |
| `actor_last_layer_scale` | 1 | Multiplies the starting weights of the actor's last layer, which gives the mean action ([Starting policy](#starting-policy)); used only by fresh agents, since a run that starts from another loads its weights |
| `initial_action_std` | 0.5 | Starting value of every learned action spread, including an action a widened agent gains |
| `action_std_schedule` | `learned` | `learned`: each action's spread is a learned value; `log_linear`: the spreads are not learned, and every cycle sets them all, from `initial_action_std` in the first cycle to `final_action_std` in the last, in a straight line of their logarithms (overriding a saved run's spreads), as the PPO paper annealed its humanoid tasks' log spread (Schulman et al., 2017) |
| `final_action_std` | initial_action_std | The spread of the last cycle, with a `log_linear` schedule |
| `exploration_noise_beta` | 0 | Colors the exploration noise over time, `1/f^β`: 0 is white noise, drawn afresh at every step; 0.5 the PPO default of Hollenstein et al. (AAAI 2024), 1 pink noise (Eberhard et al., ICLR 2023). Each step's noise stays standard normal, so the policy's probabilities are unchanged ([RL_lib's colored-noise.md](../../RL_lib/docs/colored-noise.md)) |
| `exploration_noise_sequence_steps` | 1000 | How many steps of colored noise each world's actions draw at a time: 1,000 as Hollenstein et al. do, or one episode length (`max_episode_steps`) as Eberhard et al.'s code does. The longer the sequences, the more of the noise drifts more slowly than an episode. Unused with white noise |
| `optimizer` | `adam` | Each network's optimizer: `adam`, `adamw` (Adam with decoupled weight decay), or `sgd` |
| `learning_rate` | 3e-4 | Learning rate of the first update |
| `learning_rate_schedule` | `constant` | How the rate changes over the run's update cycles: `constant`, `linear`, or `cosine` (half a cosine, slow at both ends) |
| `final_learning_rate` | learning_rate / 10 | Learning rate of the last update, when the schedule is not constant |
| `weight_decay` | 0 | Pulls the weights toward zero at every step; added to the gradient (L2) for Adam and SGD, decoupled for AdamW |
| `momentum` | 0.9 | SGD's momentum; unused by the Adam optimizers |
| `normaliser_epsilon` | 1e-8 | Keeps observation normalisation well defined |
| `observation_clip` | 10.0 | Normalised observations are clipped to ± this value |
| **`[agents.ppo]`** | | |
| `discount` | 2^(−1/episode length) | Discount factor γ; the configuration fills it in from `max_episode_steps` (rule R0 of the reward, see [environment.md](environment.md#the-rules)), 0.9973 for 256-step episodes. Built on their own, the agents' settings default to 0.999 |
| `gae_lambda` | 0.95 | GAE λ |
| `clip_ratio` | 0.2 | PPO clip ratio |
| `update_epochs` | 4 | Passes over the stored data per update |
| `minibatch_size` | 64 | Samples per gradient step |
| `max_gradient_norm` | 0.5 | Gradient clipping |
| `entropy_coefficient` | 0.001 | Weight of the entropy bonus |
| `temporal_smoothness_coefficient` | 0 | Weight of CAPS's temporal term in the actor's loss (Mysore et al., ICRA 2021): the distance between the mean actions in consecutive steps, so that the policy's mean action changes smoothly; each segment agent keeps the observation after every step for it ([RL_lib's action-smoothness.md](../../RL_lib/docs/action-smoothness.md)). 0 leaves it out |

Fixed by design rather than configured: actor and critic are separate networks,
the policy is a Gaussian squashed by tanh into −1 to 1, each action's spread is
one learned value that does not depend on the observation (RL_lib's
`std_mode = "global"`), and actor and critic each have their own optimizer,
of the same kind and with the same settings. RL_lib's PPO takes the
optimizers ready-made, so choosing them needs nothing from the library.

### Starting policy

A fresh actor's last layer gives the mean action. With PyTorch's starting
weights, that mean already differs from one observation to the next, as if
the agent began with random reflexes. With `actor_last_layer_scale = 0.01`
the layer starts a hundred times smaller, so every mean action starts near
0 whatever the agent sees: at first the actions differ only by their
exploration noise, whose spread starts at `initial_action_std`. Andrychowicz
et al. (ICLR 2021) found that the starting policy matters surprisingly much
for on-policy learning, and recommend this start with a spread of 0.5, which
did best in four of their five tasks.

The experiment sets the learning rate before every update with the agents'
`set_learning_rate`, computed from the schedule and the run's cycle
(`AgentSettings.learning_rate_at`), and the learning category logs it.

The number of steps collected before each update, `rollout_window_steps` (first
value 256 per world), belongs to the interaction loop.

**Minibatch size and speed.** A minibatch takes a few milliseconds almost
independently of its size up to 4,096 samples, so an update's time depends
mostly on how many minibatches it takes. With 1,024 worlds and 256 steps per
window, a `minibatch_size` of 64 means about 131,000 minibatches per update for
the eight agents, roughly as long as collecting the window; 4,096 means about
2,000. Runs with many worlds therefore use large minibatches, as the GPU
examples in `configs/` do.
