# Reward

**Status (2026-10-14): the structure is agreed in ideas; no term has a
formula or a weight yet.** The task and health layers are agreed with the
user, energy included. The followers' goal, the path layer, is proposed and
awaits the user's agreement.

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

Each segment is paid for progress toward what it can see, and for its
wellbeing:

| Layer | Who | What it pays | Judged by |
| --- | --- | --- | --- |
| Task | The head | Progress toward the target, and arriving | The head's task critic |
| Path (proposed) | Each follower | Progress along the trail the segment ahead leaves | Its critic |
| Health | Every segment | Its own damage and effort, plus α times its neighbours' | Its critic (the head's walk critic) |

### Task: the head steers

The head is the only segment that sees the target, so it alone is paid for
it: progress, step by step, as its distance to the target shrinks, and
arrival, when the target comes under the head. The head decides where the
body goes.

Progress and arrival are one goal at two time scales, so they stay together
in one critic. Arrivals are rare; progress is the dense guide that leads to
them. Summed over an approach, progress depends only on where the head
started and where it ended, so wiggling earns nothing and progress cannot be
farmed (main's progress already works this way, [environment.md](../environment.md#rewards)).
A critic of arrivals alone would learn from rare events only.

### Path: the followers' version of the task (proposed)

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
conflict inside the head that only the weight could settle.

A variant for later: progress along the follower's own heading. It is
simpler and differs only in curves, where the body bends.

### Health: outcomes and effort

- **Damage outcomes**, judged as what happened: a body dragging on the
  ground, legs colliding.
- **Effort**: the energy the segment's motors spend. This is the general
  rule that can replace every gait rule. Horses use each gait in the range
  of speeds where it costs them the least energy (Hoyt and Taylor, Nature
  1981). Quadruped robots paid only for moving at a speed and charged for
  energy found walk, trot and gallop by themselves, each at the speed where
  it is cheapest (Fu et al., CoRL 2021). Energy changed nothing in the
  oscillators' run 43, but that run was dominated by clocks and step
  rewards.

**The cooperation dial α.** A segment's health term also includes its
neighbours' damage and effort, times α for each segment of distance:
α = 0 cares only about itself, α = 1 cares about every neighbour it sees as
much as about itself. It reaches no farther than the critic sees. It is
essential, not optional: if each segment paid only its own effort, laziness
would pay, since a segment that hangs from its neighbours saves its own
energy while they carry it. That is what the oscillators' followers learned
(no foot down in 42 to 54% of steps). With α, burdening the neighbours
costs the segment too. It is visible as the pairing rule asks: a critic
sees its neighbours' state and actions, and with torque motors an action is
the effort. The idea comes from MA2C (Chu et al., 2019), where each agent's
reward includes its neighbours' discounted by distance.

## Which critic predicts what

| Agent | Critic | Predicts |
| --- | --- | --- |
| Head | Task critic | The task layer |
| Head | Walk critic | The health layer |
| Each follower | Its critic | The path and health layers |

The head's two advantages are added with a weight ([critic.md](critic.md)),
chosen once the terms exist.

## Left out

- Step-quality terms (foot hold, carry, grip, the step value): they
  prescribe a gait, and runs 8 to 57 disproved them.
- Rhythm and tempo terms.
- Rewards for sameness or for beating the others.
- A separate cost of time: paying for progress already makes standing still
  worse.

## Open

1. **The path:** the trail of the segment ahead (proposed), or the
   follower's own heading.
2. **Speed:** more is better, or a commanded speed the body must match?
   Fu et al. used commanded speeds, and different gaits appeared at
   different speeds.
3. **Health:** which damage outcomes, and how effort is measured.
4. **α:** whether it blends only health and effort or the path too, and how
   far it reaches within the radius.
5. **The head's weight** between its task and its health.
