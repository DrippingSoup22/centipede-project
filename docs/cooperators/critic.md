# Critic

**Status: decided** (the user, 2026-10-14) **and built**
([As built](#as-built)).

A critic estimates how good a state is, so that learning can ask after
every action: did things go better or worse than expected? That difference,
the advantage, is what teaches the actor. If the critic predicts badly, the
advantage is mostly noise and the actor learns nothing.

## The setup

Every segment learns with two critics of its own (the user, 2026-10-14):
a **goal critic** for the progress it is paid for, and a **cost critic**
for what its acting costs, energy and pain ([reward.md](reward.md)).

| | Head | Each follower |
| --- | --- | --- |
| Actor | Its own; sees itself, its neighbours within the radius, and the target | Its own; sees itself and its neighbours within the radius |
| Goal critic | Its own; judges the task (progress toward the target, arrivals), so it also sees the target | Its own; judges the path (progress along the trail of the segment ahead) |
| Cost critic | Its own; judges its energy and pain and its neighbours' | The same |
| Each critic's input | What its actor sees, plus those neighbours' actions | The same |
| Shared with other segments | Nothing | Nothing |

There is no critic of the whole body. Critics are used only in training;
an agent acts with its actor alone. Head and followers differ only in what
their goal is, and in the head's goal critic seeing the target.

**From cycle 3 the followers have no goal critic** (the user, 2026-10-15):
they are paid no goal, only the cost, so each follower learns with its cost
critic alone, its advantage normalised on its own; the head keeps both
([reward.md](reward.md#cycle-3-followers-that-follow-the-users-design)).
The environment lists the reward channels each segment learns from
(`segment_reward_channels`), and each agent builds a critic, and a
baseline, for each; the log shows a missing critic's explained variance as
empty.

**From cycle 4 every segment has a trail critic** (the user, 2026-10-16):
the reward has three channels, goal, trail and cost
([reward.md](reward.md#cycle-4-the-trail-and-carrying-oneself-the-users-design)).
The followers learn from trail and cost, 1 : 1, the trail being their goal;
the head from all three, task, trail and cost weighed **1 : 0.25 : 1**
(`[agents] head_advantage_weights`), so that its line never outweighs
reaching targets. Each agent weighs its critics by the weights of the
channels it learns from.

**From cycle 5 every segment has three critics, one per priority** (the
user, 2026-10-16): the reward has four channels, goal, coordination, trail
and health ([reward.md](reward.md#cycle-5-coordination-first-the-users-design)).
The followers learn from coordination, trail and health, the head from
goal, trail and health, both weighed **40 / 35 / 25** (1 : 0.875 : 0.625;
`[agents] advantage_weights` and `head_advantage_weights`, one weight per
channel, each segment using its own channels'). Separate critics rather
than one with ratios: a segment's own actions show clearly in its
coordination cost, half in its shared energy, and hardly at all in its
trail under the untrained noise; one critic of their sum would let the
blurred signals hide the clear one, and the weights then set each
priority whatever its units or the stage of training. A third critic per
agent adds about 2 to 3% to a cycle's time (learning took 1.9 s of 24 s
in run 07).

**From cycle 6 every segment has two critics again, goal and health** (the
user, 2026-10-16): the reward has two channels
([reward.md](reward.md#cycle-6-distance-on-the-trail-the-squeeze-its-price-the-users-design)).
For the head the goal is its target, less being pushed; for a follower the
distance it walks on its trail, less its squeeze on the segment ahead.
Health is energy, pain and hanging for both. Every segment weighs them
**1 : 0.8** (`[agents] advantage_weights = [1.0, 0.8]`), so that health is
always present but never outweighs moving. The squeeze sits in the goal
channel, beside the distance it prices, so that the critic sees one
trade-off, how far against how hard to push, at the goal's full weight.

**Cycle 7 keeps the two critics and their weights**
([reward.md](reward.md#cycle-7-the-heads-line-the-users-design)). A
follower's goal critic now judges the head's line it erases, less its
distance from the line and its front joint's force, both squared; the
head's judges its target alone. Why 1 : 0.8 rather than 1 : 1: goal and
health disagree mainly on moving at all, since energy always prefers a
body standing still, and with each advantage normalised on its own the
weights alone decide who wins there; the goal comes first (principle 1 of
[reward.md](reward.md#principles)).

**How the two are combined.** Each critic's advantage is normalised over
the batch on its own, and the two are added with one weight, the same for
every segment: goal and costs count **1 : 1** to start. The weight alone
sets the balance, whatever the units of the two (progress is a distance,
the costs an energy), and a goal whose raw signal is small early in training
keeps its voice (van Seijen et al., NeurIPS 2017; the oscillators' head did
the same). The goal never counts less than the costs ([reward.md](reward.md),
principle 1); in run 57, a task weight of 0.2 did not pull the body along.

**Size: two hidden layers of 64** to start (the user's proposal), against
run 57's 256 × 256. A critic now judges a local situation, five segments,
and the hexapod's leg networks were this size (Schilling et al., 2020).
It is not mainly a matter of speed: the physics dominates a step, and an
update's time depends more on how many minibatches it takes than on their
width. Whether the critics are big enough shows in their explained
variance, measured at every update; main can already widen a run's
networks from a checkpoint without losing what they learned
([agents.md](../agents.md#checkpoints)).

## Why each segment has its own critic

The first plan was one critic for the whole body (centralized training with
decentralized execution, as MAPPO: Yu et al., NeurIPS 2022). It was dropped
for three reasons:

1. **The experiment stays honest.** The study asks whether local,
   independent agents can cooperate. A judge of the whole body would make
   part of the coordination come from a central teacher. With local
   critics, everything a segment knows, while acting and while learning,
   comes from what it can sense.
2. **A central critic does not make agents more cooperative.** Lyu et al.
   (AAMAS 2021) prove that after the critics converge, a central critic and
   local critics give the actors the same expected gradient, and that the
   central critic's policy gradient has at least as much variance: it
   carries the teammates' random choices straight into each agent's
   learning signal. What it buys is steadier value learning. Their proofs
   cover critics of action histories, not exactly MAPPO's state critic, so
   this is strong guidance rather than a verdict.
3. **Its one big advantage is not needed.** A central critic's main job is
   to make a reward the agents cannot see learnable. The pairing rule asks
   instead for a reward each segment can see ([reward.md](reward.md)).

Local critics have a theory of their own (Qu, Wierman and Li, Operations
Research 2022): when agents affect each other locally and each reward is
local, the value depends less and less on far agents, exponentially with
their distance, so a critic of the nearest agents is nearly exact.

## What a critic sees: its neighbourhood, and what the neighbours did

Each critic sees what its actor sees, plus the actions its neighbours took
at the same step.

**The problem it solves.** When segment 5 acts and the body then does well,
was it segment 5, or did segment 4 happen to push hard at the same moment?
A critic of the state alone cannot tell, so segment 4's luck lands in
segment 5's advantage. With twelve agents, this noise from the others'
exploration is the main enemy of learning.

**How.** A critic that also sees the neighbours' actions predicts "given
what my neighbours just did, what should I expect?" Whatever the neighbours
caused is in that prediction and is subtracted with it, so the advantage
keeps mostly the segment's own part.

**Why it is allowed.** This is an action-dependent baseline (Wu et al.,
ICLR 2018). The baseline for one agent may depend on the other agents'
actions without biasing the gradient, provided each agent chooses its
action independently given the state; the paper names multi-agent systems
as the case where this holds by construction. Ours do: every actor samples
its own action.

It is the opposite of a central critic's effect: that one adds the
teammates' actions to the target, this one subtracts them out.

## Accepted limits

- **The chain is not perfectly local.** A push at the tail moves the head
  within the same step, so far segments add noise a local critic cannot
  explain. That is accepted: the agents must learn to cooperate with what
  they can see. Trouble caused by far segments is fixed in the reward, by
  keeping each term within the radius its critic sees, not by a larger
  critic.
- **The followers' goal is the body's** (cycle 13,
  [reward.md](reward.md#cycle-13-the-shared-reward-the-users-design-2026-10-17)).
  Each follower is paid the followers' mean goal, beyond its radius. Its
  main part, the line, is the same for every follower, since the chain
  moves together, and a follower sees it through its own tail; the far
  segments' own costs are the noise, averaged over eleven. Measured on
  paired worlds, the goal's learnability stayed what it was (0.78 against
  0.82 at run 22's cycle 64). Health stays within the radius.
- **The head is different.** It alone sees the target, so its goal and its
  goal critic differ from the followers'. Each agent has its own reward, so
  this keeps independence.

## How each update learns

**Decided** (the user, 2026-10-15), from the GPU locomotion studies closest
to this one and the oscillators' own updates ([research.md](research.md#training-practice-for-ppo-with-many-agents)).
An episode of 20 s is 1,000 steps: the agents act every 20 ms, as in
Rudin et al.'s 20 s episodes at 50 Hz.

| Setting | Value | Why |
| --- | --- | --- |
| Worlds | 4,096 | Rudin et al.'s and Isaac Gym's Humanoid's number (the user's proposal, 2026-10-15). On the RTX 5090 it simulates 15% more steps per second than 2,048, measured; twice the starts and targets in each update |
| Rollout window | 32 steps (0.64 s) | 131,072 samples per segment per update, Isaac Gym's Humanoid batch (4,096 × 32), inside Rudin et al.'s best range (about 100k to 200k with 2,048 to 4,096 robots) and above their floor of 25 steps. A cycle takes 22 s on the RTX 5090 (the smoke of 2026-10-15) |
| Minibatch | 32,768 | Four per epoch, the size of both papers' minibatches |
| Epochs | 5 | Theirs; within MAPPO's 5 to 10 for hard tasks |
| Adam steps per update | 20 | Theirs. Adam's steps have about the same size whatever the minibatch, so the policy moves with their number: the oscillators' 32 at a fixed 3e-4 moved it by a KL of 0.02 to 0.2 an update, far above the papers' 0.01 |
| Learning rate | 3e-4 at the start, then adapted | Each actor's own, set after every update by Rudin et al.'s rule: divided by 1.5 after an update whose KL was above twice 0.01, multiplied by 1.5 after one below half, between 1e-5 and 1e-2. A fixed rate drifts: as training narrows the action spread σ, the same step of the mean is a larger KL (d²/2σ² for two Gaussians of the same spread). The critics keep 3e-4 |

What the window leaves to the critic: on average about 47% of what the
advantage would measure is replaced by the value at the window's end (28%
with 64 steps, 14% with 128; about half in Rudin et al.'s 24-step
windows). The pink
noise's sequences are as long as an episode, whatever the window.

## As built

**Each critic has two networks** (the user's agreement, 2026-10-14). A
critic that sees the neighbours' actions of the same step cannot value a
state nobody has acted in yet: the state an episode is cut in, or the one a
window ends on, where learning must look ahead. So each critic is:

| Network | Sees | Learns | Used for |
| --- | --- | --- | --- |
| Value | What its actor sees | The return targets | Looking ahead: bootstrapping and the return targets (GAE) |
| Baseline | That, and the four neighbours' actions of the same step (36 values) | The same return targets | The actor's advantage: return target minus baseline |

The baseline regresses the return on the state and the neighbours' actions,
so it learns the return's expectation over the segment's own action: Wu et
al.'s action-dependent baseline. Every segment has four such networks, goal
and health (cost before cycle 3), value and baseline, each 2 × 64, and an
actor of 2 × 64.

- **Value normalisation** (MAPPO's first advice, Yu et al., NeurIPS 2022):
  every critic's value and baseline learn its return targets standardised
  by their running mean and spread, so millimetres and millijoules never set
  the scale the networks learn.
- **The combination**: each critic's advantage normalised over the batch on
  its own, then added by `advantage_weights` (1 : 0.8 from cycle 6).
- **Logged**: every critic's and every baseline's explained variance after
  each update. A baseline explaining much more than its value shows the
  neighbours' actions at work.

The PPO is RL_lib's, on its `cooperators` branch; the settings are
`[agents] critic_baselines`, `value_normalization` and `advantage_weights`.
The adaptive learning rate is Centipede's (`[agents] learning_rate_schedule
= "adaptive"`, `desired_kl`), applied once per update rather than after
every minibatch, so that the GPU never waits in the middle of one; each
segment's learning rate is logged.

## Not chosen

| Alternative | Why not |
| --- | --- |
| One critic for the whole body | The three reasons above. Kept only as a possible later comparison, one run with that single change, if local learning stalls |
| A critic that sees farther than its actor | Not needed now; a dial for later if far segments turn out to matter |
| Counterfactual baseline (COMA, Foerster et al., AAAI 2018) | Parked: a harder critic, built for discrete actions; only for a measured credit problem |
| Sequential updates (HAPPO, Kuba et al., ICLR 2022) | Parked: only for measured instability between agents |

More on each in [research.md](research.md).
