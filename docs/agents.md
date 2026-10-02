# Agents

The agents are the learners: one **segment agent** for each body segment (eight
in v1), managed as one group. They sit on layer 2 of the
[architecture](architecture.md), next to the environment, and never talk to it
directly; the interaction loop passes data between them. The agents know nothing
about MuJoCo, targets, or how observations and rewards are calculated.

Learning uses **PPO** from the reusable package in `../RL_lib/src/rl_lib`, which
also provides networks, policies, normalisation, and data types. The experiment
runners in `../RL_lib/experiments` are not used; this project has its own.

## Interface

`W` is the number of worlds and `N` the number of segments. Everything passed in
and out is a PyTorch tensor on the agents' device. The group is created from two
numbers the environment reports, `segment_count` and `observation_size`, so it
works for any number of segments.

| Operation | Takes | Returns or does |
| --- | --- | --- |
| `act(observations)` | Observations `(W, N, observation_size)` | The joint action `(W, N, 6)`; each segment agent remembers what it needs for learning |
| `record(rewards, terminated, truncated, final_observations)` | The environment's results for the step just taken | Each segment agent stores its own part |
| `update()` | Nothing | Each segment agent learns from its own stored data, then clears it |
| `save()`, `load()` | — | All segment agents' state, for checkpoints |

During evaluation, `act` chooses actions without recording or learning. For
comparison, the front file also provides two baselines with the same `act`: one
that always returns zero actions and one that returns uniformly random
actions.

## Files

The agents live in `src/centipede/agents/`.

| File | Covers |
| --- | --- |
| `agents.py` | Front file: creates one segment agent per segment, gives each its slice of the observations, joins their actions, and passes on recording, learning, saving, and loading; also provides the zero and random baselines |
| `segment_agent.py` | One segment's networks, optimizers, observation normaliser, and PPO update |
| `rollout_storage.py` | One segment agent's collected data |
| `settings.py` | Agent settings |

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

## Observations

Every segment agent has the same input size, `observation_size`: 83 with the
first observation radius of 1 (see
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

Advantages are computed separately for every uninterrupted stretch in each
world, then all stretches of the same segment are used together. A window ending
never resets the environment; the episode continues in the next window.

## Checkpoints

A checkpoint is one bundle holding, for every segment agent, its networks,
optimizers, normaliser, random state, and training counters, plus the settings,
the number of segments, and the observation size. It is saved only after a
completed update. Loading it into an environment with a different number of
segments or observation size fails with a clear error. It holds no stored data,
physical state, targets, or unfinished episodes, so continuing from a checkpoint
starts new episodes.

## Settings

These are the keys of the agents sections of the configuration file (see
[configuration.md](configuration.md)). Each first value is also the default.

| Setting | First value | Meaning |
| --- | ---: | --- |
| **`[agents]`** | | |
| `device` | `cpu` | Where networks and stored data live: `cpu` or `cuda` |
| `hidden_layers` | [64, 64] | Hidden layer sizes of both actor and critic (ReLU) |
| `initial_action_std` | 0.5 | Starting value of the six learned action spreads |
| `learning_rate` | 3e-4 | Adam learning rate, constant, no weight decay |
| `normaliser_epsilon` | 1e-8 | Keeps observation normalisation well defined |
| `observation_clip` | 10.0 | Normalised observations are clipped to ± this value |
| **`[agents.ppo]`** | | |
| `discount` | 0.999 | Discount factor γ |
| `gae_lambda` | 0.95 | GAE λ |
| `clip_ratio` | 0.2 | PPO clip ratio |
| `update_epochs` | 4 | Passes over the stored data per update |
| `minibatch_size` | 64 | Samples per gradient step |
| `max_gradient_norm` | 0.5 | Gradient clipping |
| `entropy_coefficient` | 0.001 | Weight of the entropy bonus |

Fixed by design rather than configured: actor and critic are separate networks,
the policy is a Gaussian squashed by tanh into −1 to 1, and actor and critic each
have their own Adam optimizer.

The number of steps collected before each update, `rollout_window_steps` (first
value 256 per world), belongs to the interaction loop.

## Open questions

- **GPU support in RL_lib.** RL_lib does not yet let the caller choose the device
  for networks, sampling, storage, and updates. This should be added to RL_lib as
  a general feature, after agreeing its scope.
- **Batched networks.** Many small networks may not keep a large GPU busy. Since
  all segment agents have the same input size, their networks could be evaluated
  together in one operation while keeping separate parameters, optimizers, and
  data. Do this only if measurements show it is worth it.
