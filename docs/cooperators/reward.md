# Reward

**Status (2026-10-14): the structure is agreed in ideas; four ratios
remain; no term has a formula or a number yet.**

Agreed with the user: the goal layers (task for the head, path for the
followers), the costs as energy and pain, α sharing energy only, fading
with distance over every neighbour a segment sees, a goal critic and a cost critic for every segment
at 1 : 1, and free speed. Proposed and awaiting the user: energy
as a small animal's muscles spend it (pushing work, cheaper braking work,
and a dominant cost of force).

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
| Energy | Every segment | Every force its motors exert and every movement they make; its neighbours' through α | Cost critic |
| Pain | Every segment | The force with which its legs hit other legs, the same for both segments of a contact | Cost critic |

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

Progress is paid, more is better, and energy costs, so the body settles at
the speed where more progress stops being worth its energy: the balance
between goal and costs is the speed dial, and the gait that emerges is the
cheapest for that speed. The speed comes from the environment, not from a
command (the user, 2026-10-14; a commanded speed, as in Fu et al., is not
planned). How the task's distances and the arrival's value make speed pay
belongs to [task.md](task.md).

### Energy: the price of every force (the user's design)

**The idea: be as lazy as possible, intelligently.** Energy is everything
the centipede's motors do: every force they exert and every movement they
make. It is not harm; it is what acting costs, and it shapes how the body
walks. Almost every cost reduces to it:

| What happens | How energy charges it |
| --- | --- |
| Standing | Holding the body up against gravity takes force, all the time |
| Resting on the belly | Saves that force; standing up again costs work. The agents weigh the two, as a real Scolopendra rests on its belly when still |
| Dragging the belly while moving | The body's friction on the floor (0.8, as the feet's, [model.md](../model.md#contacts-and-friction)) must be pushed through: work |
| A clumsy step, a slip, legs fighting | Work and force spent for nothing |
| A joint pushed against its stop | Force held without motion |
| Hanging, being carried | Costs the lazy segment little, but its neighbours pay the extra force, and α brings their cost back to it |

So no posture rule is needed: whether to stand, rest or drag is a choice
the physics prices. Energy is the general rule that can replace every gait
rule. Horses use each gait in the range of speeds where it costs them the
least energy (Hoyt and Taylor, Nature 1981). Quadruped robots paid for
moving at a commanded speed and charged for their motors' power found walk,
trot and gallop by themselves, each at the speed where it is cheapest (Fu
et al., CoRL 2021). Energy changed nothing in the oscillators' run 43, but
that run was dominated by clocks and step rewards.

**Two costs of different natures** (the user, 2026-10-14). Moving and
holding both spend energy, but they are counted differently:

| | Work | Force |
| --- | --- | --- |
| What it is | Force times the distance it moves: torque times the angle a joint turns | The cost of producing a force at all, moving or not |
| Counted per | Distance moved: it stops when the motion stops | Time held: a steady rate while the force lasts |
| Lifting a barbell | m·g·h, once, for the lift | Nothing extra while lifting quickly |
| Holding it still | Zero: nothing moves | The whole cost, every second it is held |
| In an electric motor | The mechanical power the motor delivers | Heat in the windings, which grows with the square of the torque |
| In a muscle | Pushing work at about 25% efficiency; braking work several times cheaper | The cost of generating force, which dominates in small animals |

**What the physics and the biology say.**

- **Motors.** In the MIT Cheetah, heat in the actuators was 76% of all the
  energy spent; the work done was the smaller part (Seok et al., IEEE/ASME
  Transactions on Mechatronics 2015).
- **Muscles, work.** Muscle turns about a quarter of the energy it spends
  into positive work (the reference efficiency of Cavagna and Kaneko, J.
  Physiol. 1977); negative work, braking, is several times cheaper per
  joule (Abbott, Bigland and Ritchie, J. Physiol. 1952).
- **Muscles, force.** The energy of running is set mainly by producing
  force: its rate grows with body weight divided by the time a foot is on
  the ground (Kram and Taylor, Nature 1990).
- **Size.** The mechanical work an animal does per kilogram and per metre
  hardly depends on its size, but the energy it spends per kilogram grows
  as animals get smaller (Taylor, Heglund and Maloiy, J. Exp. Biol. 1982;
  Heglund et al., 1982). For an animal of a gram, almost all of the energy
  is the cost of force, made by fast muscle fibres in many quick steps
  (Kram and Taylor, 1990).

**Why the two natures give a preferred speed.** Work is paid per metre and
force per second. Walking faster makes each metre take less time, so the
force cost per metre falls; it also makes the legs swing faster, so the
work per metre rises. The energy per metre is therefore U-shaped in speed,
with a cheapest speed in between: the U-curves Hoyt and Taylor measured in
horses. This is what makes free speed work, and the ratio between work and
force decides where the cheapest speed lies.

**The proposal: energy as a small animal's muscles spend it.**

| Part | How it is counted | Ratio, from the sources |
| --- | --- | --- |
| Pushing work | Positive work of each motor, at muscle efficiency | 1 (the unit) |
| Braking work | Negative work: charged, never refunded, but cheaper | About 1/4 of pushing work per joule (several times cheaper) |
| Force | Each motor's torque, squared, for every moment it is held | Set so that force is by far the larger part in a walk, as in animals of a gram; measured on our model, not guessed |

- **Squared force.** Producing a larger force costs more per unit of force:
  muscles recruit their costlier fast fibres last, and a motor's heat grows
  with the square of its torque. The square also makes sharing a load
  between legs cheaper than one leg carrying it alone; a cost linear in
  force would not care how the weight is shared.
- **Fu et al. counted braking as a gain,** which would pay legs for braking
  each other. Here it is a cost.
- **What the simulation holds.** A motor's torque is held for each 20 ms
  action, so the force cost of an action is exactly its torque squared
  times its duration, and its work is the torque times the angle the joint
  turned; braking is told apart from pushing by the sign of torque times
  joint speed at each physics step. With torque motors the torque is the
  action, so the cost critic sees it.
- **A check of realism, not a rule.** With force dominant, whether a raised
  walk or a belly drag is cheaper depends on how the legs carry the weight.
  Real centipedes walk raised; if our energy made dragging the cheaper way
  to move, the energy model, not the centipede, would be wrong, and it is
  measured before any training.

### Pain: legs hitting legs (the user's design)

Energy cannot price everything. When legs hit each other, two things
happen: force is wasted, which energy charges, and harm is done, which it
does not:

- **A leg could push off a neighbour's leg instead of the ground.** If that
  is cheaper, energy would even reward it. Pain forbids using legs as
  footholds.
- **A hard, brief kick costs little energy** but is harm all the same.

Pain is graded by force: the force with which a segment's legs press on
other legs, so legs that brush without pushing feel almost nothing, as in
the real animal. **Pain is mutual and stays between the two:** by
Newton's third law, both legs of a contact feel the same force, so both
segments pay it, the same amount, and no other segment does (the user,
2026-10-14). This is the user's leg
pushing of 2026-10-11 (oscillators, cycle 1), inert there (under 1% of its
most) while clocks kept the legs apart; here nothing does.

It is also the cost most tied to cooperation. Two same-side neighbours'
legs touch when the rear one's sweep converges on the front one's, so how
far each can sweep depends on how the neighbours' steps are offset in time
(the oscillators' leg geometry, 2026-10-11): pain makes each segment take
its neighbours' steps into account, without saying which offset to use.

### Other harms left out

| Candidate | Why |
| --- | --- |
| Body on the ground | Energy prices it: resting saves force, dragging costs work. Forbidding it would forbid a real centipede's rest (the user) |
| Body turned over | Never seen in any run: the many legs keep the body stable. A term that never fires teaches nothing; it returns if a recording shows it |
| Feet slipping | Energy charges the wasted work; a judgement of the step's quality was gamed in the oscillators |
| Hard impacts, tapping feet | Seen in the oscillators with other motors and clocks; added only if it appears here |

The rule: every cost is energy, except harm that costs no energy, and that
harm enters only when we see it.

### The cooperation dial α (agreed)

A segment's energy also counts its neighbours', as far as it sees them
(the user, 2026-10-14). Only energy is shared: pain stays between the two
segments of a contact, and each follower's path is its own. It is essential, not optional: if each
segment paid only its own energy, laziness would pay, since a segment that
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

α applies to energy only.

## Critics and balances

| Agent | Goal critic | Cost critic |
| --- | --- | --- |
| Head | Task | Energy and pain |
| Each follower | Path | Energy and pain |

Goal and costs count **1 : 1** (agreed), through two advantages normalised
apart ([critic.md](critic.md)). Energy and pain share one critic rather
than one each: pain is rare, and a critic of a signal that is almost
always zero would have its noise enlarged by the normalisation. Inside the
cost critic, pain is priced in energy: how much energy a hit of a given
force is worth.

| Balance | Status |
| --- | --- |
| Goal : costs | 1 : 1, agreed |
| α | Halving per segment, as a share of the whole, agreed |
| Pushing : braking work | About 4 : 1, proposed from muscle data |
| Work : force, inside energy | Proposed: force dominant, as in animals of a gram; measured on our model |
| Pain : energy | Open |
| Progress : arrival, for the head | Open, with [task.md](task.md) |

## Left out

- Step-quality terms (foot hold, carry, grip, the step value): they
  prescribe a gait, and runs 8 to 57 disproved them.
- Rhythm and tempo terms.
- Rewards for sameness or for beating the others.
- A separate cost of time: paying for progress already makes standing still
  worse.
