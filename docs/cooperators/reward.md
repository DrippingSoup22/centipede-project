# Reward

**Status (2026-10-14): the structure is agreed in ideas; the balances are
being set as ratios; no term has a formula or a number yet.** Agreed with
the user: the three layers (task, path, health), energy as effort, α
reaching every neighbour a segment sees. Proposed and awaiting the user:
the shape of α, a goal critic and a health critic for every segment, and
the questions on speed and health below.

## Principles

1. **The goal first.** Moving toward the goal is always worth more than
   what it costs; otherwise standing still becomes the best choice, as in
   run 57.
2. **The pairing rule.** Every term is caused by things that the critic
   predicting it can see ([critic.md](critic.md)).
3. **Outcomes, never form.** A term judges what happened, never how the
   legs moved. Every rule about how to move was gamed in the oscillators.
4. **Cooperation through shared outcomes.** Never through sameness, which
   pays synchrony instead of a wave, or rivalry, which makes neighbours
   compete.

## Structure

**Every segment is paid for progress toward what it can see, and for its
wellbeing.**

| Layer | Who | What it pays |
| --- | --- | --- |
| Task | The head | Progress toward the target, and arriving |
| Path | Each follower | Progress along the trail the segment ahead leaves |
| Health | Every segment | Damage and effort: its own, and its neighbours' through α |

```text
head      → sees the target   → progress toward the target   (task)
segment 1 → sees the head     → progress along the head's trail
segment 2 → sees segment 1    → progress along segment 1's trail
   ...                           (path: the task as a follower sees it)
```

### Task: the head steers

The head is the only segment that sees the target, so it alone is paid for
it: progress, step by step, as its distance to the target shrinks, and
arrival, when the target comes under the head. The head decides where the
body goes.

Progress and arrival are one goal at two time scales, so they stay together
in one critic. Arrivals are rare; progress is the dense guide that leads to
them. Summed over an approach, progress depends only on where the head
started and where it ended, so wiggling earns nothing and progress cannot
be farmed (main's progress works this way,
[environment.md](../environment.md#rewards)). A critic of arrivals alone
would see zero almost every step and a rare spike, and with its advantage
normalised on its own, those spikes would dominate.

### Path: the task as a follower sees it

A follower cannot see the target, so by the pairing rule it cannot be paid
for it. What it can see is where the body goes: the head lays the path, and
for each follower **the segment ahead is its target**. A follower is paid
for its progress along the trail the segment ahead leaves.

- **It is visible.** The segment ahead is always within the radius.
- **It is shared.** In a chain, this progress is how far the body moves
  along its own path, so every follower earns about the same: a common goal
  that every segment can see. This is how the hexapod (mean velocity) and
  the snake (forward progress) paid their limbs.
- **It passes the head's turns along.** Motion along the trail is paid;
  pushing off it is not. A turn of the head passes from segment to segment,
  as in a real centipede, whose body follows the path of its head.

**The head gets no path term.** Its progress toward the target already is
its movement, counted only when it heads for the target. A movement term
blind to direction would also pay it for walking fast the wrong way, a
conflict inside the head that only a weight could settle.

A variant for later: progress along the follower's own heading. It is
simpler and differs only in curves, where the body bends.

### Health: damage and effort

- **Damage outcomes**, judged as what happened: a body dragging on the
  ground, legs colliding.
- **Effort**: the energy the segment's motors spend. This is the general
  rule that can replace every gait rule. Horses use each gait in the range
  of speeds where it costs them the least energy (Hoyt and Taylor, Nature
  1981). Quadruped robots paid for moving at a commanded speed and charged
  for their motors' power found walk, trot and gallop by themselves, each
  at the speed where it is cheapest (Fu et al., CoRL 2021). Energy changed
  nothing in the oscillators' run 43, but that run was dominated by clocks
  and step rewards.

**The cooperation dial α.** A segment's health also counts its
neighbours' damage and effort, as far as it sees them (the user,
2026-10-14). It is essential, not optional: if each segment paid only its
own effort, laziness would pay, since a segment that hangs from its
neighbours saves its own energy while they carry it. That is what the
oscillators' followers learned (no foot down in 42 to 54% of steps). With
α, burdening the neighbours costs the segment too. It is visible as the
pairing rule asks: a critic sees its neighbours' state and actions, and
with torque motors an action is the effort. The idea comes from MA2C (Chu
et al., 2019), where each agent's reward includes its neighbours',
discounted by distance.

## Balances (ratios, not numbers yet)

| Balance | Status | Proposal |
| --- | --- | --- |
| Goal : health, for every segment | Proposed | 1 : 1, through two critics per segment (below) |
| α's shape | Proposed | Halving per segment of distance, as a share of the whole |
| Damage : effort | Open | |
| What effort measures | Open | Work done, force held, or both |
| Progress : arrival, for the head | Open | |

### α: halving with distance

A segment's health is a weighted share of its own and its neighbours'
damage and effort. The proposal: each segment of distance halves the
weight, and the weights are scaled to make a whole. For a middle segment
at radius 2:

| Whose | Weight |
| --- | ---: |
| Itself | 40% |
| Each nearest neighbour | 20% |
| Each second neighbour | 10% |

Why decaying rather than equal:

1. **Weight follows influence.** A segment's actions change its nearest
   neighbours' load the most: the joints next to it carry it first. A
   reward term that a segment cannot change is pure noise in its learning
   signal, so each term should count as much as the segment can affect
   it.
2. **Influence fades with distance** in locally coupled systems,
   exponentially (Qu, Wierman and Li, 2022); halving per step mirrors that.
3. **Each reward stays mostly its own.** Neighbouring windows overlap; with
   equal weights, two neighbours would share four fifths of their health
   signal, and each segment would learn from the others' noise as much as
   from its own.

Why a share of a whole: the head and the rear see fewer neighbours than a
middle segment. Scaled to a whole, every segment's health has the same
size, so the balance between goal and health is the same along the body.

α applies to health only. The path is already shared by the chain;
blending it would add nothing.

### Goal : health, through two critics for every segment

Proposed, a change to [critic.md](critic.md): every segment learns with
two critics, a **goal critic** (the head's task, a follower's path) and a
**health critic**, each advantage normalised on its own and the two added
with one weight, the same for every segment.

- **The balance becomes a pure ratio.** Progress is measured in distance,
  effort in energy. Summed in one critic, their balance would need a
  conversion between the two units; with normalised advantages, the weight
  alone sets it, whatever the units.
- **Every segment has the same structure.** Only what "goal" means differs
  between the head and the followers.
- **The goal keeps its voice.** A normalised advantage has the same size
  however small its raw signal, so a goal that is small early in training
  is never drowned by the costs (run 57's failure).

The costs: twice as many critics (small networks next to the physics), and
a layer whose signal is almost always zero, such as rare damage, has its
noise enlarged by the normalisation.

**1 : 1 to start** (the user's proposal). Both matter, and the goal first
(principle 1) sets the floor: the goal never counts less than health. If
the body learns to keep still, the goal is raised; run 57's 0.2 was too
weak.

## Speed: what is still to decide

- **Free speed.** Progress is paid, more is better, and effort costs. The
  body settles at the speed where more progress stops being worth its
  effort, so the goal : health ratio is the speed dial. One speed, and the
  cheapest gait for it. Nothing new to see.
- **Commanded speed** (Fu et al.). The body is told a speed to match,
  varied in training, and different gaits appear at different speeds. By
  the pairing rule, every segment paid for matching the command must see
  it: a signal for the whole body, like a brain's descending command, an
  exception to the local view. Fu et al. also found that training on
  random speeds collapsed to the slower gaits.

Proposed: free speed now; the commanded speed a later experiment, with its
own question about a whole-body signal.

## Health: what is still to decide

- **What effort measures.** Work done (the motors' power, as Fu et al.)
  charges motion; force held (the squared torque, as Gymnasium's Ant)
  charges holding a load still. With work alone, a segment that holds a
  hanging neighbour still pays nothing, so α could not stop free-riding at
  rest; a muscle also spends energy holding a force. Both together are
  closest to a muscle.
- **Which damage outcomes.** Each must be an outcome, visible to the
  critic, and truly unwanted: a body on the ground, legs colliding, perhaps
  a body turned over. In the oscillators these stayed small because bodies
  stayed up: they guard against rare failures rather than drive learning.
- **Damage : effort.** How many steps of ordinary effort one step of
  damage is worth.

## Left out

- Step-quality terms (foot hold, carry, grip, the step value): they
  prescribe a gait, and runs 8 to 57 disproved them.
- Rhythm and tempo terms.
- Rewards for sameness or for beating the others.
- A separate cost of time: paying for progress already makes standing still
  worse.
