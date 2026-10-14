# Critic

**Status: decided** (the user, 2026-10-14).

A critic estimates how good a state is, so that learning can ask after
every action: did things go better or worse than expected? That difference,
the advantage, is what teaches the actor. If the critic predicts badly, the
advantage is mostly noise and the actor learns nothing.

## The setup

| | Head | Each follower |
| --- | --- | --- |
| Actor | Its own; sees itself, its neighbours within the radius, and the target | Its own; sees itself and its neighbours within the radius |
| Walk critic | Its own; sees what its actor sees, plus those neighbours' actions | Its own; sees what its actor sees, plus those neighbours' actions |
| Task critic | Its own; judges the target task | None |
| Shared with other segments | Nothing | Nothing |

There is no critic of the whole body. Critics are used only in training;
an agent acts with its actor alone.

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

## The head's second critic

Only the head sees the target, so it alone is paid for the task, and it
learns that part with a critic of its own, which can see the task's cause.
The head's two advantages are normalised over the batch each on its own and
then added with a weight, so the weight alone sets the balance between the
task and the rest, whatever their scales (van Seijen et al., NeurIPS 2017;
the oscillators' head did the same).

The weight is chosen with the reward. In run 57, a weight of 0.2 did not
pull the body along, mainly because that run's walk reward made standing
still the best choice.

## Accepted limits

- **The chain is not perfectly local.** A push at the tail moves the head
  within the same step, so far segments add noise a local critic cannot
  explain. That is accepted: the agents must learn to cooperate with what
  they can see. Trouble caused by far segments is fixed in the reward, by
  keeping each term within the radius its critic sees, not by a larger
  critic.
- **The head is different.** It alone sees the target, so its reward and
  its critics differ from the followers'. Each agent has its own reward, so
  this keeps independence.

## Not chosen

| Alternative | Why not |
| --- | --- |
| One critic for the whole body | The three reasons above. Kept only as a possible later comparison, one run with that single change, if local learning stalls |
| A critic that sees farther than its actor | Not needed now; a dial for later if far segments turn out to matter |
| Counterfactual baseline (COMA, Foerster et al., AAAI 2018) | Parked: a harder critic, built for discrete actions; only for a measured credit problem |
| Sequential updates (HAPPO, Kuba et al., ICLR 2022) | Parked: only for measured instability between agents |

More on each in [research.md](research.md).
