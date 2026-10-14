# Reward

**Status (2026-10-14): the structure is agreed in ideas; the balances are
being set as ratios; no term has a formula or a number yet.**

Agreed with the user: the goal layers (task for the head, path for the
followers), effort as an energy cost, α fading with distance over every
neighbour a segment sees, a goal critic and a health critic for every
segment at 1 : 1, and free speed. Proposed and awaiting the user: effort
and damage as two layers judged by one cost critic, and the list of
damage.

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

**Every segment is paid for progress toward what it can see, and pays for
what its moving costs.**

| Layer | Who | What it says | Judged by |
| --- | --- | --- | --- |
| Task | The head | Progress toward the target, and arriving | Goal critic |
| Path | Each follower | Progress along the trail the segment ahead leaves | Goal critic |
| Effort (proposed as its own layer) | Every segment | The energy its motors spend, and its neighbours' through α | Cost critic |
| Damage (proposed as its own layer) | Every segment | Harm to the body, its own and its neighbours' through α | Cost critic |

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

### Speed is free

Progress is paid, more is better, and effort costs, so the body settles at
the speed where more progress stops being worth its effort: the balance
between goal and costs is the speed dial, and the gait that emerges is the
cheapest for that speed. The speed comes from the environment, not from a
command (the user, 2026-10-14; a commanded speed, as in Fu et al., is not
planned). How the task's distances and the arrival's value make speed pay
belongs to [task.md](task.md).

### Effort: the price of acting (proposed as its own layer)

The energy the segment's motors spend. It is not health: nothing is
harmed, it is what doing things costs, and it shapes how the body walks.
It is the general rule that can replace every gait rule. Horses use each
gait in the range of speeds where it costs them the least energy (Hoyt and
Taylor, Nature 1981). Quadruped robots paid for moving at a commanded speed
and charged for their motors' power found walk, trot and gallop by
themselves, each at the speed where it is cheapest (Fu et al., CoRL 2021).
Energy changed nothing in the oscillators' run 43, but that run was
dominated by clocks and step rewards.

**What it measures: work done and force held** (proposed).

| Measure | Charges | Alone, it would |
| --- | --- | --- |
| Work done: the motors' power (Fu et al.) | Moving | Make holding a load still free: a segment holding up a hanging neighbour pays nothing, so α could not stop free-riding at rest |
| Force held: the squared torque (Gymnasium's Ant) | Holding and moving alike | Not tell useful motion from holding still |

Together they are closest to a muscle, which spends energy moving and also
tires just holding a weight. With torque motors the torque is the action,
so the cost critic sees it, its neighbours' included.

### Damage: harm to the body (proposed as its own layer)

What counts as damage must be (1) an outcome, not a form of movement;
(2) visible to the cost critic; (3) not already paid by another term; and
(4) something that is really happening or that another term pushes toward.
Proposed:

| Candidate | Proposal | Why |
| --- | --- | --- |
| Body on the ground | **Kept** | Effort makes it necessary: holding the body up costs force, so with effort alone, lying on the belly would become the cheapest way to move. It is effort's counterweight. Real centipedes walk with the body raised |
| Legs colliding | **Kept** | Two neighbours' legs hit each other only when they move out of step in space, so avoiding it asks each to take the other into account: the damage most tied to cooperation, yet an outcome, not a gait. Both segments involved pay |
| Body turned over | Left out | Never seen in any run: the many legs keep the body stable. A term that never fires teaches nothing, and if it fired, the goal would already suffer. It returns if a recording shows it |
| Joints pushed against their stops | Left out | Effort already charges it: torque held without motion is force held |
| Feet slipping | Left out | A judgement of the step's quality, gamed in the oscillators; slipping also wastes work, which effort charges |
| Hard impacts, tapping feet | Left out for now | Seen in the oscillators with other motors and clocks; added only if it appears here |

The rule: a damage term enters only for harm we see, or that another term
pushes toward.

### The cooperation dial α (agreed)

A segment's effort and damage also count its neighbours', as far as it
sees them (the user, 2026-10-14). It is essential, not optional: if each
segment paid only its own effort, laziness would pay, since a segment that
hangs from its neighbours saves its own energy while they carry it. That
is what the oscillators' followers learned (no foot down in 42 to 54% of
steps). With α, burdening the neighbours costs the segment too. It is
visible as the pairing rule asks: a critic sees its neighbours' state and
actions. The idea comes from MA2C (Chu et al., 2019), where each agent's
reward includes its neighbours', discounted by distance.

**Fading with distance: halving per segment, as a share of the whole.**
For a middle segment at radius 2:

| Whose | Share |
| --- | ---: |
| Itself | 40% |
| Each nearest neighbour | 20% |
| Each second neighbour | 10% |

Why fading rather than equal:

1. **Weight follows influence.** A segment's actions change its nearest
   neighbours' load the most: the joints next to it carry it first. A term
   that a segment cannot change is pure noise in its learning signal, so
   each term counts as much as the segment can affect it.
2. **Influence fades with distance** in locally coupled systems,
   exponentially (Qu, Wierman and Li, 2022); halving per step mirrors that.
3. **Each reward stays mostly its own.** Neighbouring windows overlap; with
   equal weights, two neighbours would share four fifths of their signal,
   and each segment would learn from the others' noise as much as from its
   own actions.

Why a share of a whole: the head and the rear see fewer neighbours than a
middle segment. Scaled to a whole, every segment's costs have the same
size, so the balance between goal and costs is the same along the body.

α applies to effort and damage only. The path is already shared by the
chain; blending it would add nothing.

## Critics and balances

| Agent | Goal critic | Cost critic |
| --- | --- | --- |
| Head | Task | Effort and damage |
| Each follower | Path | Effort and damage |

Goal and costs count **1 : 1** (agreed), through two advantages normalised
apart ([critic.md](critic.md)). Effort and damage share one critic
(proposed) rather than one each: damage is rare, and a critic of a signal
that is almost always zero would have its noise enlarged by the
normalisation. Inside the cost critic, damage is priced in effort: how
many steps of ordinary effort one step of damage is worth.

| Balance | Status |
| --- | --- |
| Goal : costs | 1 : 1, agreed |
| α | Halving per segment, as a share of the whole, agreed |
| Work : force, inside effort | Open |
| Damage : effort | Open |
| Body on the ground : legs colliding | Open |
| Progress : arrival, for the head | Open, with [task.md](task.md) |

## Left out

- Step-quality terms (foot hold, carry, grip, the step value): they
  prescribe a gait, and runs 8 to 57 disproved them.
- Rhythm and tempo terms.
- Rewards for sameness or for beating the others.
- A separate cost of time: paying for progress already makes standing still
  worse.
