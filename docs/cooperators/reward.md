# Reward

**Status (2026-10-16): agreed with the user and built; the force price
measured on the model ([As built](#as-built)). The chain was added after
runs 01 and 02 ([The chain](#the-chain-push-and-pull-between-neighbours-the-users-design)).
After run 04, cycle 3 takes the followers' goal away: they follow, paying
the chain at ten times the force price
([Cycle 3](#cycle-3-followers-that-follow-the-users-design)). After runs 05
and 06, cycle 4 returns to run 01's design without the chain and gives the
followers a new goal, the trail, with a cost for not carrying themselves
([Cycle 4](#cycle-4-the-trail-and-carrying-oneself-the-users-design)).
After runs 07 and 08, cycle 5 gives every segment three priorities in
three critics: for the followers coordination, the trail as a reward, and
health; for the head its target, its neck's trail and health
([Cycle 5](#cycle-5-coordination-first-the-users-design)). After runs 09
and 10, cycle 6 returns to two critics, goal and health: each follower is
paid the distance it walks on its trail, less its squeeze on the segment
ahead; the head its target, less being pushed from behind; health is
energy, pain and hanging
([Cycle 6](#cycle-6-distance-on-the-trail-the-squeeze-its-price-the-users-design)).
After runs 11 and 12, cycle 7 gives the followers the head's line: each is
paid the line its tail erases, and charged, squared, for how far its tail
is left from the line and for the force at its front joint; the head has
its target alone ([Cycle 7](#cycle-7-the-heads-line-the-users-design)).
After runs 13 and 14, cycle 8 keeps that reward and averages the forces
over 1 s instead of 0.2 s, so that only a lasting push or pull costs
([Cycle 8](#cycle-8-only-a-lasting-force-costs-the-users-idea)), while the
agents explore with state-dependent noise
([actions.md](actions.md#how-actions-explore)). After runs 19 to 22, cycle
13 shares the reward: each follower is paid the followers' mean goal, pain
and hanging are shared with the neighbours as energy is, and pain is ramped
in over the first 128 cycles
([Cycle 13](#cycle-13-the-shared-reward-the-users-design-2026-10-17)). After
run 23, cycle 14 splits the reward by owner: a team critic for the
followers' mean line and distance from it, the head's target and the body's
mean energy; an own critic for each segment's force share, pain and hanging;
pain from half its weight
([Cycle 14](#cycle-14-the-teams-and-each-ones-own-the-users-split-2026-10-17)).
After run 24, cycle 15 pays the line near it instead of charging the
distance, draws it where the head keeps pointing, and charges every
follower the chain's squeeze; energy and the squeeze are ramped in with
pain ([Cycle 15](#cycle-15-the-line-where-the-head-points-and-the-squeeze-the-users-design-2026-10-17)).
After run 25, cycle 16 moves energy into each segment's own channel,
shared with its neighbours (40% itself, 25% each nearest, 5% each next),
and prices pain on the legs' press averaged over 1 s
([Cycle 16](#cycle-16-energy-shared-with-the-neighbours-pain-averaged-the-users-design-2026-10-18)).**

Agreed with the user: the goal layers (task for the head, path for the
followers); the costs as energy and pain; energy as a small animal's
muscles spend it (pushing work, cheaper braking work, and a dominant cost
of force, counted as squared activation); pain priced like force, 1 : 1;
α sharing energy only, fading with distance over every neighbour a segment
sees; a goal critic and a cost critic for every segment at 1 : 1; free
speed; the chain, each segment's own cost for pushing, lagging and
yanking its neighbours, priced like force.

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
| Chain | Every segment | The force on the joints to its neighbours, along the body: pushing or lagging behind the one ahead, yanking the one behind | Cost critic |
| Trail (cycle 4, replacing the path) | Every segment | How far its body, and the body of the segment it steers, point off their trails | Trail critic |
| Carrying oneself (cycle 4) | Every segment | The net force its joints put on it: pulled, pushing, or held up; the head only held up | Cost critic |
| Coordination (cycle 5) | Each follower | The net force its joints put on it, averaged: lagging, driving the body (pushing the one ahead counted twice), held up, lying on the floor | Coordination critic |
| Trail as a reward (cycle 5) | Every segment | Time spent on the trail: full within 5°, fading beyond | Trail critic |
| Own weight (cycle 5) | The head | Held up by its neck beyond half its weight, or lying on the floor | Health critic |
| Distance on the trail (cycle 6) | Each follower | How far its front end walks along its body, times how well it keeps to its trail | Goal critic |
| Squeeze (cycle 6) | Every segment | A follower pushing the segment ahead; the head pushed from behind; averaged | Goal critic |
| Hanging (cycle 6) | Every segment | Held up by its neighbours (the head beyond half its weight), or lying on the floor; averaged | Health critic |
| The head's line (cycle 7) | Each follower | The line the head's tip draws, paid as its tail erases it | Goal critic |
| Off the line (cycle 7) | Each follower | Its tail's distance from the line it has not erased, squared | Goal critic |
| Front joint (cycle 7) | Each follower | Its front joint's averaged force, pushed or pulled, squared | Goal critic |

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

**One logic for every segment** (agreed, 2026-10-14). Every segment is
paid the distance it gains toward its own target in millimetres; the head's
target is the target, a follower's the trail of the segment ahead
([Path](#path-the-task-as-a-follower-sees-it)). The head alone can reach
its target, so only the head has an arrival. Main counted the head's
progress in halvings of its distance instead; in millimetres, the head and
the followers measure their goals the same way, and every millimetre is
worth the same near a target and far from it, as every millimetre of
walking costs the same energy. A farther target therefore pays more
progress in all, in proportion to the walk it asks for: the user's idea of
an arrival scaled by distance, carried by the progress itself.

**The arrival is worth one average approach** (agreed): reaching a
target is worth as much as the progress of walking the middle distance of
the current level's range, so an average approach earns half its worth
walking and half arriving, at every level. It makes finishing better than stopping
just short, which progress alone would barely distinguish, while staying
small enough that rare arrivals do not swamp the goal critic's signal.
It need not pay back the trip's energy: goal and costs are judged by two
critics and weighed 1 : 1 in every update ([critic.md](critic.md)), not
summed into one value, so a long walk never makes a target "not worth it";
with progress in millimetres, the trade between gaining ground and saving
energy is the same at every distance.

Progress and arrival are one goal at two time scales, so they stay together
in one critic. Arrivals are rare; progress is the dense guide that leads to
them. Summed over an approach, progress depends only on where the head
started and where it ended, so wiggling earns nothing and progress cannot
be farmed. A critic of arrivals alone would see zero almost every step and
a rare spike, and with its advantage normalised on its own, those spikes
would dominate.

**No price per millimetre in energy** (the user's view, 2026-10-14:
progress is one thing, cost another). Each critic's advantage is
normalised on its own before the two are weighed 1 : 1, so the scale of
each side disappears: progress counted as 1 or as 1,000 per millimetre
teaches the same. Millimetres and joules are never converted into each
other; what sets the balance is the weight, and in every update the
centipede improves gaining ground and saving energy with equal emphasis,
each measured against its own usual spread. There is no break-even point:
a centipede that walks well keeps both improving, faster progress for less
energy per millimetre. Two consequences: inside the goal, only the ratio
of progress to arrival matters (above); and the exchange rate between
millimetres and energy is not fixed but follows how much each signal
varies, so if the walk ends too lazy or too wasteful, the 1 : 1 weight is
the dial, never below 1 for the goal. Normalising the critics' value
targets too (MAPPO's first advice) makes even the critics blind to scale.

### Path: the task as a follower sees it

A follower cannot see the target, so by the pairing rule it cannot be paid
for it. What it can see is where the body goes: the head lays the path, and
for each follower **the segment ahead is its target**. A follower is paid
for its progress along the trail the segment ahead leaves: the path that
segment has walked, measured in millimetres gained along it.

**A follower never arrives.** The joint keeps the segment ahead one unit
ahead, so its target can never be reached; its goal is continuous, to keep
advancing along the trail. It gets no share of the head's arrival either:
it cannot see the target, so an arrival would be an effect without a cause
(the pairing rule), as for the oscillators' followers. Its part in every
arrival is already paid, millimetre by millimetre, as it carried the body
along the head's path. Nor is it about momentum: a body of a gram has
almost none ([forces.md](forces.md)); the follower's work is advancing,
step after step.

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
| Force | Each motor's activation, its torque as a share of its maximum, squared, for every moment it is held | Set so that force is by far the larger part in a walk, as in animals of a gram; measured on our model, not guessed |

- **Squared activation.** Producing a larger force costs more per unit of
  force: muscles recruit their costlier fast fibres last, and a motor's
  heat grows with the square of its torque. In simulations of human gait,
  the sum of squared muscle activations predicted a realistic walk, while
  energy-like costs gave stiff, straight-legged stances (Ackermann and van
  den Bogert, J. Biomech. 2010). The maximum torques then act as muscle
  sizes ([forces.md](forces.md)). The square also makes sharing a load
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

**Priced like force** (agreed). Pain is counted in the same currency as
the force cost: the contact force as a share of the strongest push a leg's
sweep makes at its foot (about 0.9 mN, about one unit's weight; the
user's measure of 2026-10-11), squared, for every moment it lasts. Pain and
energy count **1 : 1** in it: being pressed with the force of one's own
strongest push hurts as much as pushing that hard. It is the least that
does the job: any pain makes a leg used as a foothold costlier than the
ground; squared, a brush costs almost nothing and a hard kick a lot; and
harm is never cheaper than the effort it equals. It is raised only if
recordings show legs still colliding.

It is also the cost most tied to cooperation. Two same-side neighbours'
legs touch when the rear one's sweep converges on the front one's, so how
far each can sweep depends on how the neighbours' steps are offset in time
(the oscillators' leg geometry, 2026-10-11): pain makes each segment take
its neighbours' steps into account, without saying which offset to use.

### The chain: push and pull between neighbours (the user's design)

**Why** (2026-10-15, runs 01 and 02). A follower is paid for its progress
along the trail, but the joint keeps it one unit behind the segment ahead,
so that progress is the whole body's speed, and every follower can raise
it by pushing. Both runs learned exactly that, and nothing told the
followers that the head was being pushed against its will: at cycle 64
the head of run 01 carried 1.25 unit weights on its belly and touched the
ground in 52% of steps, a sled pushed by followers walking on their feet,
and it never learned to turn (heading error 139°). The missing signal is
the force between neighbours. A difference of speed cannot show it, since
the joint keeps two neighbours at a fixed distance and so at almost the
same speed; the force through the joint shows who does the work. Zero
means each segment carries and moves itself.

**The force.** For each joint, the force of the segment ahead on the
segment behind, along the segment behind's forward axis, in unit weights:
positive when the segment ahead pulls it (it lags), negative when it pushes
the segment ahead. MuJoCo computes it (`cfrc_int`, the force each body's
parent exerts on the subtree it heads); the body tree runs from the head
back, so a segment's subtree is itself and everything behind it, and the
force on its front joint holds the push or lag of every segment behind.
Only the part along the body is priced (the user): the vertical part,
hanging on the neighbours, is left to energy, which already made runs 01
and 02 carry their weight without any rule (their followers' feet down in
80 to 95% of steps); the sideways part carries the head's turns along the
body, and pricing it would tax steering.

**Who pays: each segment, for its own two joints** (the user's chain, after
the oscillators' tempo chain: the gap ahead linear, the gap behind
squared).

| Joint | What happens | Whose doing | The segment pays |
| --- | --- | --- | --- |
| Front | It pushes the segment ahead | Its own: it is faster | The size, linear |
| Front | It is pulled, lagging | Its own: it is slower | The size, linear |
| Rear | It pulls the segment behind | Its own: it should wait | The square |
| Rear | It is pushed from behind | The segment behind's | Nothing: that segment pays it at its front |

Keeping up with the segment ahead is a follower's duty, so the front joint
costs linearly, already for small gaps. Waiting for the segment behind is a
courtesy, so the rear joint's pull costs its square: small pulls almost
nothing, big yanks a lot; the two prices meet at one unit weight. A segment
pushed from behind pays nothing for it: if it did, its cheapest answer
would be to run from the pusher, as the head of run 01 would learn to run
straight on instead of turning. The head has no front joint, the rear no
rear one.

**Its own, not shared.** Every joint is already charged to the two segments
it joins, and the physics already passes the chain along: the force on a
segment's front joint holds the push or lag of every segment behind it, so
a segment that lets a push through pays for it, and holding it back passes
the charge on until it reaches the segment that pushes. Sharing the chain
by α, or weighting the joints further away (the user's proposal of 2 : 1
for the joints one and two segments away, considered on 2026-10-15), would
charge a segment for joints it cannot change: the joint two ahead holds
the lag of the segment ahead, which a segment can only cancel by pushing
it, and the joint two behind is already inside its own rear joint. Costs
an agent cannot affect blur what it learns from its own actions (Wolpert
and Tumer, Advances in Complex Systems 2001). The weighted chain stays a
variant to screen if the plain one reacts too late. Seeing the joints
around it is what lets a segment anticipate: each segment observes them
([observations.md](observations.md)).

**The price** (agreed): `chain_price` = 1, a unit weight on a joint for a
second costing as much as one motor fully on, 1.98 mJ/s
([As built](#as-built)).

### Cycle 3: followers that follow (the user's design)

**Why** (2026-10-15, run 04, the chain's pilot on 3 seeds). The chain lifted
the head off its belly and it began to steer, but two seeds of three ended
circling their targets: over 3 m walked per 20 s episode for 35 to 47 mm
closed. Measured on the cycle-256 checkpoints (4 worlds per seed;
`analysis/orbit_check.py` in `../Training`):

| Measure | Run 04 |
| --- | --- |
| The head pushed by segment 1 | 85 to 97% of steps, 0.5 to 0.7 unit weights |
| Strongest push, joints 1 to 3 | 0.8 to 1.0 unit weights |
| Sideways force at the joints | 0.3 to 0.8 unit weights, as much as along |
| Spine yaw joints | 10 to 12° of their 15° on average, at the limit 30 to 87% of the time, the middle zigzagging, the neck held to one side |
| The head's path | 165 mm/s on circles of about 35 mm radius, its target 57 to 76 mm away, heading error about 90° |

A follower never arrives, so progress along the trail is a reward for speed
without a ceiling, and energy cannot cap it: with force dominant, a faster
walk is cheaper per millimetre ([As built](#as-built)). Eleven followers
paid for speed outvoted the head, the only segment that knows when to slow
down; they pushed it, and a chain pushed from behind buckles, like a rope
pushed or a trailer jackknifing. Bent and pushed, the head can only circle.

**The design** (the user's, agreed 2026-10-15):

- **The followers have no goal.** Only the head is paid, for its task; a
  follower learns from its cost alone, with one critic, the cost critic.
  The followers are called so because they follow: the head sets the pace
  and the way, and the body's duty is to go with it.
- **The chain is the followers' duty, priced high.** Each follower pays the
  force on its front joint along the body, push or lag, linearly, at ten
  times the force price (`chain_price` = 10): a unit weight on the joint for
  a second costs 19.8 mJ, as ten motors fully on. Its minimum, zero force,
  is a follower carrying and moving itself at the pace its leader sets. Only
  differences teach, and each critic's signal is normalised on its own, so
  while the forces swing they dominate what a follower learns; once it
  follows well they stay near zero, and what still varies, energy, leads.
- **The rear pull: an open question, tested both ways.** The head's square
  of its pull on segment 1 is its safeguard against dragging the body faster
  than the body adapts (run 05, `chain_rear_pull = "head"`); but before the
  body walks, every move of the head pulls, and that cost could teach the
  head, the only segment with a reason to move, to stand still (run 06,
  `"none"`). No follower pays its rear pull: each joint is charged to the
  segment behind it, whose duty is to follow.

**The price, checked** (`analysis/price_check.py` in `../Training`, CPU,
2026-10-15; per follower and second, the reward's own code at price 10):

| Case | Speed | Energy | Chain | Total |
| --- | ---: | ---: | ---: | ---: |
| Scripted raised walk, every segment carrying itself | 39 mm/s | 0.8 mJ | 3.3 mJ | 4.0 mJ |
| The head walks, the followers command nothing | 0 | 0.0 | 8.5 mJ | 8.5 mJ |
| Run 04 at cycle 256, pushing and circling | 158 mm/s | 5.6 mJ | 13.4 mJ | 19.0 mJ |

Following beats lying idle, and lying beats pushing. Two lessons. The head
alone cannot move a passive body (its feet push at most about friction
times its weight, 0.8 unit weights, which a resting segment's belly
matches), so if the followers learn to rest before they learn to follow,
nobody moves and no single follower's action changes its joint's force: a
deadlock, run 57's stillness again. The fallback, if a run shows it, is
Hwangbo et al.'s curriculum (Science Robotics 2019, "Policy training
details"): every cost but the objective multiplied by a factor rising from
0.3 to 1, here energy and pain, the chain at full price from the start. And
even the scripted walk keeps a small steady joint force (about 0.15 unit
weights), whose chain costs four times its energy; if energy never comes to
lead, the price is the dial.

**The sideways force: measured, not priced.** A follower that leaves the
head's path, cutting a corner or going straight, pushes or is dragged
sideways at its front joint, so the sideways force seemed the signal of a
bad turn. Scripted turns say otherwise (`analysis/turn_forces.py` in
`../Training`: the scripted walk with the spine's yaw motors at a constant
command, since
feedback on the angle at 50 Hz makes the joints chatter; mean square of the
sideways force per joint and step, in unit weights squared):

| Walk | Turn radius | Spine bend | Sideways² |
| --- | ---: | ---: | ---: |
| Straight | — | 2° | 0.10 |
| Even bend, gentle | 109 mm | 3° | 0.11 |
| Even bend, tight | 38 mm | 13° | 0.26 |
| Even bend, tightest | 25 mm | 15° | 0.43 |
| Rear bent more (cuts inside) | 44 mm | 10° | 0.18 |
| Rear bent less (swings outside) | 308 mm | 12° | 0.29 |
| S-shape | 62 mm | 12° | 0.15 |

An even bend keeps every segment on one circle, the head's path; an uneven
one leaves it. The sideways force grows with how hard the body turns, not
with whether it follows the path: a correct tight turn costs more than a
wrong one. Walking straight, left and right legs alternating, rocks each
joint sideways by 0.26 unit weights a step even when every segment carries
itself (0.01 with both legs together), a sway an average over a stride
removes but a price per step would charge. And the bend at a follower's
front joint is its leader's command, so a cost on that joint's sideways
force would charge the follower for its leader's choice. So the chain
prices the force along the body only, and whether the followers trace the
head's path is measured in the runs, before any rule rewards it.

### Cycle 4: the trail and carrying oneself (the user's design)

**Why** (2026-10-15, runs 05 and 06, the cycle-3 design with and without
the head's rear pull, seed 1, 128 cycles). The followers stopped pushing:
the head's joint carried −0.01 and +0.02 unit weights, no belly dragged,
pain was about zero. But nothing steered: 0.11 and 0.04 targets an
episode, none in time, against run 04 seed 1's 0.49 at the same cycle.
Three findings (`analysis/orbit_check.py` and `spine_chatter.py` in
`../Training`):

- **The spine chatters, in every run measured** (run 04 too). The mean
  spine command changes sign on 23 to 50% of steps, and each joint swings
  across its whole ±15°: the motor's 15 µN·m beats the spring's 5.2 at the
  limit, so any command above about 0.35 reaches the limit within one
  decision. The middle joints carry oscillating forces of 0.5 to 0.6 unit
  weights.
- **A follower cannot learn to lower a force it does not command.** Each
  segment commands the joint behind it, but paid the chain of the one in
  front. Segment 1, behind the light neck, cut its chain from 7.6 to
  2.5 mJ/s; the middle followers, shaken by joints their leaders command,
  only from 11.1 to 10.1, while their force rose from 3.4 to 5.8, so their
  whole cost grew.
- **Unpushed, the head hangs.** Its feet carried 0.05 to 0.15 of its
  weight (1.55 in run 04, pressed down by the push), segment 1 holding it
  up through the joint. Without grip it cannot pull, and it shook in place
  on a path of 1 mm radius.

The user's reading: along the body nothing stretches, so the body moves as
one object, every segment at the same speed at every instant. A follower
cannot have a speed of its own; it can only push or brake the whole body.
Paying it for its own progress (runs 01 to 04) lets it push the head;
charging it only a force that its leader causes (runs 05 and 06) teaches it
nothing. A follower's goal is to follow the segment directly ahead of it:
to go where it went. Every rule that prescribed a behaviour got the
opposite, while energy, which only prices, gave a raised and lazy walk; so
the new terms price outcomes and leave the rest to energy.

**The design** (the user's, agreed 2026-10-16), built on run 01's design,
without the chain:

- **Direction belongs to the place, speed to the moment.** Where the body
  turns is fixed to the ground: a follower should turn where its leader
  turned, so its reference comes from that place. Its speed is the body's at
  that instant; a speed copied from where its leader was would be a speed
  from the past, pushing the head when it slows to turn and braking it when
  it speeds up. So only the head is paid for moving (its task), and the
  followers' goal is where their bodies lie.
- **The trail.** A segment's trail is the path its front end traced: the
  head's tip, or the spine joint it shares with the segment ahead. Its rear
  end is one segment length behind (4.3 mm, the head 7.15 mm), so it lies
  on its trail when its body points at where its front end was one segment
  length ago. Its **offset** is the angle between the two. It is measured
  by distance, never time: zero at any speed and standing still; each trail
  is kept only one segment length back, and a front end swaying in place
  adds nothing to it. Before a segment has walked a length, its trail runs
  straight back along it.
- **One number, two payers.** The joint between two segments only bends,
  so a segment's offset is also the error of the joint in front of it: the
  bend that would put the segment on its trail, less the bend it has. The
  segment ahead steers that joint with its spine motor and pays the offset
  fully; the segment itself, whose legs can only fight the joint, pays a
  part, `trail_own_weight` (0.25 to start; the user also considers 0.5), so
  that its legs help mend errors. Whoever causes an error pays it, which
  cycle 3 lacked; a chattering spine knocks its own follower off the trail.
  The head lays the trail: where it points is free, set by its task. It pays
  its own offset fully, since only it decides where it walks: turning on
  the spot about its neck costs almost nothing, walking crabwise does. It
  also pays segment 1's, through its neck. The last segment pays its own.
- **Squared.** An offset counts as its square in degrees, computed as
  twice (1 − cos) of the angle in squared degrees: the square while the
  angle is small, levelling off past a right angle. The natural sway of a
  walk, about 5° in the middle of the body, then costs little beside real
  errors, and a body walking backward (180°) costs only a few times a bad
  turn.
- **Its own critic, the trail critic.** The followers learn from trail and
  cost, 1 : 1, their only goal being the trail; the head from task, trail
  and cost, 1 : 0.25 : 1, so that its line never outweighs reaching
  targets ([critic.md](critic.md)).
- **Carrying oneself.** Energy alone does not stop a follower going limp
  (below). So each follower pays the net force its two spine joints put on
  it, its legs included: along the body, when it is pulled or pushes, and
  up, when its neighbours hold it. The head pays only the second, so that
  it is free to pull the body when the followers lag and still made to
  stand on its own feet. Unlike the chain's front joint force, which carried
  the push of every segment behind, the net force on a segment is its own
  legs' doing. Linear in unit weights, priced at a quarter of the force
  price (`carry_price` 0.25).
- **Seen.** Each observation block holds its segment's offset and net
  joint force ([observations.md](observations.md)), so each critic sees the
  offsets it pays, its own and those of the segment behind, within its
  radius.
- **The spine stays active** (the user's choice, from biology: Anderson et
  al., J. Exp. Biol. 1995, found the body muscles active in walking
  *Scolopendra*; the spine helps the legs, never replaces them). A passive
  spine, as in Aoi et al.'s centipede-like robot (Sci. Rep. 2016), whose
  passive body joints follow its first module's steering, is kept for a
  later test.

**The trail, measured on scripted walks** (`analysis/trail_check.py` in
`../Training`, CPU, 2026-10-16; the scripted wave walk with the spine's yaw
motors at constant or alternating commands; mean offset sizes in degrees
over 5 s):

| Scripted case | The head's own | Segment 1 (the neck) | Trunk |
| --- | ---: | ---: | ---: |
| Walking straight | 4.7 | 3.9 | 6.5 |
| Walking straight, the neck bent (crabwise) | 14.4 | 6.1 | 6.3 |
| Spine flipping ±0.5 every 3 steps (like the runs' chatter) | 23.6 | 10.8 | 8.9 |
| Spine flipping ±1 every 3 steps | 48.1 | 22.1 | 15.9 |
| Body bent in an S | 25.7 | 15.4 | 18.6 |
| Even bend, legs walking along each segment | 6.7 to 10.7 | 5.8 to 10.4 | 8.7 to 15.6 |
| Standing still | 0.1 | 0.2 | 0.1 |
| Standing, the head turning its neck | 3.9 | 2.4 | 0.7 |

The walk sways about 5° in the middle of the body and 12 to 18° at the
free tail. A crabwise head and a chattering spine show at the front; no
scripted turn follows its trail, since the scripted legs walk along their
own segment, and in a turn the rear segments skid inward: the learned legs
must steer along the trail instead. The chattering spine made the scripted
walk 2.5 to 4 times faster (42 to 103 and 168 mm/s), as an undulating
body: probably why runs 01 to 04, paid for speed, learned it.

**A lazy follower, measured** (`analysis/carry_check.py` in `../Training`,
CPU, 2026-10-16; the scripted walk with segment 5's legs walking, limp or
holding still; per second, through the reward's own code; forces in unit
weights):

| Segment 5's legs | Energy it pays (shared) | Carrying itself at 0.25 | Total | Pulled forward | Held up |
| --- | ---: | ---: | ---: | ---: | ---: |
| Walking | 0.71 mJ | 0.18 mJ | 0.89 mJ | 0.03 | −0.05 |
| Limp | 0.49 mJ | 0.60 mJ | 1.09 mJ | 0.13 | 1.04 |
| Holding still, dragged | 0.91 mJ | 0.35 mJ | 1.26 mJ | 0.55 | 0.07 |

With energy alone, going limp is a third cheaper for the follower: it hangs
from its neighbours, who hold its whole weight, and energy sharing returns
too little of their effort to it. Carrying oneself at a quarter of the
force price puts walking first, limp second, dragged last, while energy
stays most of a walking follower's cost (the support its neighbours give
it swings with every step, so even walking pays a fifth of its total).

**What remains a risk.** The followers have no positive reward: they move
because the head pulls them and because lagging and hanging cost. If the
head does not lead, the body stands still: the deadlock of runs 05 and 06,
whose shaking head kept a speed of 28 to 49 mm/s on a path of 1 mm radius.
So a deadlock is read from the path's radius and the closing speed, not
the head's speed, and its dial is the head's goal weight. A ramp of the
costs (Hwangbo et al.'s, named in cycle 3) would change nothing here:
each critic's advantage is normalised on its own, so a reward's scale
disappears; only the weights between critics move the balance.

### Cycle 5: coordination first (the user's design)

**Why** (2026-10-16, runs 07 and 08, seed 1, stopped by the user at about
cycles 100 and 70). The spine stopped chattering (its sign flipped on 1 to
5% of steps) and the followers' offsets fell from 60° to 16°, but the
body learned to stand still. At cycle 64 the mean action reached no
target: run 07's head lay on its belly (1.92 of its weight on it, 0.02 on
its feet) while segments 1 to 3 braced their feet, and run 08's head
walked 10 mm in 20 s. In training the targets fell from 1.6 to 0.7 a
minute and the head's speed from 16 to 11 mm/s. Three causes:

- **Stillness was free.** A still body lies exactly on its trail and spends
  no energy; nothing paid a follower to move, and carrying oneself bit only
  when the head pulled, which it did not.
- **A small preference becomes a full one.** Each critic's advantage is
  normalised on its own, so a channel's size does not matter, only its
  weight. Once the large offsets were gone, what remained in the trail
  channel was mostly "a walking body sways, a still one does not", at the
  trail's full weight: the walk faded as the trail improved. So no term may
  prefer standing still over a normal walk.
- **The head lay down for free**: carrying oneself charged it only for
  being held up by its neck.

**Paying distance was rejected.** Path-following agents are paid per step
for speed along the path times how close they keep to it: race cars,
speed × (cos(angle) − distance from the centre) (Jaritz et al., ICRA
2018, after Perot et al., CVPR Workshops 2017); boats, a product of speed
along the path and closeness to it, plus a cost per step against idling
(Meyer et al., IEEE Access 2020); a snake robot, closeness to the path and
progress along it (Liu, Guo and Fang, Sensors 2022). Tracking rewards
without speed, as DeepMimic's (Peng et al., 2018) or a commanded velocity's
(Rudin et al., CoRL 2022), follow a reference that moves with the clock;
our trail is laid only where the body went, so a still body tracks it
perfectly. But a follower paid per millimetre is paid to make the whole
body faster, by pushing or by undulating: the scripted chattering spine
walked 2.5 times faster than the clean walk with the trunk only 2° further
off its trail, so it would have earned twice the pay. The user's choice:
the trail is paid by time, and moving comes from the forces.

**The design** (the user's, agreed 2026-10-16):

- **Three priorities, three critics, weighed 40 / 35 / 25** (1 : 0.875 :
  0.625): for each follower coordination, the trail, health; for the head
  its target, its neck's trail, health. Weights decide only where two
  goals disagree; where they agree, all improve together. One critic with
  ratios between the parts was weighed and rejected (measured below): the
  three signals are of very different quality for the segment's own
  actions, and in one critic the blurred ones hide the clear one; and a
  ratio would not hand over from coordination to energy as training goes
  on, since both fall about fourfold from the noisy start to a clean walk.
- **Coordination, the followers' first priority**, in unit weights: the
  net force its two spine joints put on a follower along the body (it lags,
  or it drives the body), the part of its own surplus that pushes the
  segment ahead counted once more (`push_factor` 2: pushing the one ahead
  is worse than pulling the one behind, the user's rule), what its
  neighbours hold it up with, and the floor's push on its body. A follower
  that carries itself and walks with the body pays nothing, whatever passes
  through its joints. The head pays none of it along the body: it is free
  to pull or brake, so it alone sets the pace, and every follower out of
  step with the body feels it in its own net force, which it observes.
- **The trail, the second priority, as a reward paid by time**: each step
  its duration times exp(−(e − 5°)²/20²), e being the offset of the
  segment steered and a quarter of its own as one angle (the rear's own
  alone), anything within the 5° dead zone paid in full. The head's is the
  same, for its neck and, at a quarter, for walking crabwise. Reward or
  cost makes no difference to learning: the episodes never end early, so a
  constant shift is absorbed by the critic.
- **Health, the third**: energy and pain as before, and for the head its
  own weight: what its neck holds it up with beyond half its weight
  (`head_free_support`), and the floor's push on its body, priced as force
  (`carry_price` 1).
- **Averaged measures.** The forces and the trail are averaged over 0.2 s
  (an exponential average, `averaging_s`) before they are priced, and each
  segment observes the averaged forces it pays. The actions are never
  filtered: only the reward's reading of the forces is. A segment has two
  legs; while one swings it leans on its neighbours, and a moment later
  holds them up. That is walking, not laziness: per step it would charge a
  perfect walker 0.30 and shape the gait at the first priority.
- **Energy shared by sides** (the user's, `energy_sharing = "sides"`):
  every segment counts itself 40% and each side 30%, as a middle segment
  does. A missing neighbour's share goes to the nearest one on its side,
  a missing side's to the other side: the head counts itself 40%, segment
  1 40% and segment 2 20%; segment 1 counts the head 30%; the rear counts
  itself 40%, segment 10 40% and segment 9 20%.
- **Targets within 90° either side from level 0** (the user:
  `start_bearing_deg` 90, from 45), so that turning is practised from the
  first cycle; the circle still opens fully as the level rises.

**Measured before building** (`analysis/coordination_check.py` in
`../Training`, CPU, v4; the untrained policy's noise, spread 0.5 and pink,
alone or added to the scripted walk, and clean scripted cases). A follower's
coordination cost, segment 5, in unit weights:

| Segment 5 | Per step | Averaged over 0.2 s |
| --- | ---: | ---: |
| Walks with the body | 0.30 | 0.06 |
| Limp | 1.17 | 1.14 |
| Holding its legs, dragged | 0.69 | 0.64 |
| Stepping 1.6 times too fast | 0.49 | 0.25 |
| Everyone standing | 0.02 | 0.02 |

The head is held up by its neck by 0.38 of its weight when it walks, even
averaged, and by none when it stands, since its two legs lean on the neck
in turn: counting all of it would make standing still cheaper than
walking, hence the free half. Hanging as in runs 05 and 06 (its feet
carrying 5 to 15%) costs it about 0.4 unit weights, limp on its belly
about 0.6.

Trail errors (the steered offset and a quarter of the own as one angle),
in degrees, median and 90th percentile over the trunk:

| Motion | Per step | Averaged |
| --- | ---: | ---: |
| Clean walk | 6, 16 | 2, 5 |
| Walk + the policy's noise on the legs | 33, 90 | 19, 60 |
| Walk + noise on the spine | 11, 23 | |
| Walk + noise on both | 37, 113 | 24, 61 |
| Standing | 0.5, 1.4 | |

Under the untrained noise the body really wobbles off its trail, mostly
from the legs' noise; averaging removes part of it and most of a clean
walk's sway, which sets the dead zone at 5°. Credit, segment 5's noise
switched off under the walk with noise (the change in each segment's
mean):

| Segment | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coordination (unit weights) | −0.05 | −0.17 | 0.28 | 0.10 | 0.01 |
| Energy (mJ/s) | 0.27 | 0.55 | 1.09 | 0.55 | 0.27 |
| Trail (degrees) | −2.7 | 0.4 | 2.0 | 1.9 | 0.6 |

**Does energy alone make a segment carry itself?** (`analysis/
sharing_check.py`; the scripted walk with one segment limp, energy per
second as the segment pays it.) No: the hinges carry a limp segment
without their motors' work, so its neighbours' energy barely rises while
its own falls to zero.

| Limp segment | Walking, old sharing | Limp, old | Walking, by sides | Limp, by sides |
| --- | ---: | ---: | ---: | ---: |
| Head | 0.83 | 0.47 | 0.82 | 0.65 |
| 1 | 0.80 | 0.59 | 0.81 | 0.64 |
| 5 | 0.69 | 0.45 | 0.69 | 0.45 |
| 11 | 0.75 | 0.55 | 0.74 | 0.77 |

Sharing by sides removes the rear's gain and halves the head's, but a
middle follower going limp still saves a third, so being held up and lying
on the floor stay in coordination and in the head's health.

**Why 40 / 35 / 25 holds** (estimated from the measures above). Energy
agrees with coordination and the trail almost everywhere: pushing,
braking and jitter all cost energy too. It disagrees only on going limp,
which costs about 1.2 spreads of coordination and saves about 0.34 spreads
of energy: at 40 against 25 coordination wins about six to one, so energy
can stay present without ever deciding against cooperating.

**Risks.** Moving still comes only from the head: if it does not lead, the
followers feel no force and the body stands; the head now pays for lying
down and its health weighs 25 against its target's 40. Under the
untrained noise the trail reward is mostly the noise's wobble, so it is
learned after coordination, as the priorities intend.

### Cycle 6: distance on the trail, the squeeze its price (the user's design)

**Why** (2026-10-16, runs 09 and 10, seeds 1 and 2, 128 cycles each). In
training the followers learned in the intended order: coordination fell
from 0.54 to 0.29 unit weights and the trail reward rose from 13% to 60%
of its most. But the body learned to stand. The heading error rose to 105
to 118° and the head drew away from its targets on average; at cycle 128
the mean action reached 0 to 0.05 targets an episode and walked 34 to 91
mm in 20 s, the head shaking at 650 to 840°/s with the spine joints pinned
at their limits and its feet carrying 1.2 to 1.3 of its weight. Every term
of cycle 5 was lowest standing still, and only the head could start the
motion: a chicken and egg, since a head that walks drags followers that
hold still, and only then would they feel a force to follow. Two more
findings:

- **The noise made standing better than walking.** At the starting spread
  of 0.5 the body wobbles off its trail and squeezes its joints whatever
  it does, and a still body wobbles less (measured below).
- **The spine pinned.** A spine command above about 0.35 holds the joint
  at its 15° limit, so two thirds of its range did the same thing
  ([actions.md](actions.md#the-spines-scale-cycle-6)).

**The design** (the user's, agreed 2026-10-16; the numbers are the
assistant's proposals, measured below, which the user accepted):

- **Two critics, goal and health, 1 : 0.8** for every segment
  (`[agents] advantage_weights = [1.0, 0.8]`;
  [critic.md](critic.md)), which the user preferred as cleaner.
- **A follower's goal: distance on the trail, less its squeeze.** Each step
  the millimetres its front end walks along its own body, times the
  trail's quality, cycle 5's shape: 1 within 5°, exp(−(beyond / 20°)²),
  for the offset of the segment it steers and a quarter of its own. Less,
  at 300 mm/s per unit weight (`squeeze_price_mm_per_s`, 6 mm a step), the
  averaged force with which its front joint pushes the segment ahead. Paid
  for distance, a follower wants to move, and a still body earns nothing;
  priced for pushing, it gains only by walking as fast as the segment ahead
  goes. This closes the loophole that made cycle 5 reject paying distance:
  a follower pushing the body faster pays more for the push than it gains.
  The user's option 1 of two; option 2, a follower also paying for the
  pull from behind, was left out: a follower pulled is one the segment
  behind lags from, and the price would fall on the wrong one.
- **The head's goal: progress, arrival, less being pushed** (the user's
  option 3): the same price on the averaged net force of its joint that
  pushes it forward. It can lose that cost only by walking at the pace the
  body pushes it to, so the followers' wish to move reaches it as a force.
  Its neck stays unpaid, to be watched: no trail term for the head, which
  is free to point where its target is.
- **Every term is local** (the user's fundamental rule): a segment is paid
  or charged only for what its own block or its neighbours' show. Its
  squeeze is its own front joint force; the head's push is its own net
  force; the offsets are its own and the segment behind's; the distance is
  its own motion. No follower's reward depends on the head's state.
- **Health: energy, pain and hanging**, priced as force. Energy is shared
  by sides as in cycle 5 (itself 40%, each side 30%); pain stays with the
  pressed. Hanging is what its neighbours hold it up with (the head's
  beyond half its weight, `head_free_support`) and the floor's push on its
  body, averaged, at one force price per unit weight (`hanging_price` 1).
  Being dragged along the body is no longer charged: the rigid chain moves
  every segment at the same speed, so a dragged follower is still paid its
  distance, and only hanging and the neighbours' shared energy price
  going limp.
- **The forces and the trail averaged over 0.2 s**, as in cycle 5; the
  actions are never filtered.
- **The spine's command scaled by 0.35** (`spine_action_scale`), so that
  the policy's whole range spans the joint's.
- **Exploration starts at a spread of 0.25** (`initial_action_std`), pink
  (β = 1), against 0.5 before.
- **Targets within 90° either side from level 0**, kept from cycle 5 and
  now the default (`start_bearing_deg` 90). The network stays 64 × 64.
- The earlier designs' settings are gone from the code: the chain,
  carrying oneself, coordination, the trail as its own channel, the
  followers' progress toward the segment ahead, and the old sharing.

**Measured before building** (`analysis/push_price_check.py` in
`../Training`, CPU, v4: the scripted wave walk, one world, per follower
and per second, forces averaged over 0.2 s; noise is the untrained
policy's, pink, added to the walk's commands).

- **Followers pushing a slow head** walk 36 mm/s instead of its 15. Only
  segments 1 to 3 feel the squeeze at their front joints, 0.13, 0.08 and
  0.02 unit weights; the rear feel none, since the push they add is
  passed forward through the chain.
- **The noise sets the squeeze's floor.** Under noise of spread 0.5 the
  followers' front joints are squeezed 0.15 to 0.20 on average (90th
  percentile 0.4 to 0.56) whatever they do.
- **Chattering does not pay**: a spine flipping every three steps earns 31
  and 22 mm/s of paid distance against the clean walk's 35.
- **A limp follower keeps its pay** (36.1 mm/s) and saves energy: only
  hanging charges it.

The followers' goal per second at 300 mm/s per unit weight, paid distance
less the squeeze:

| Case | Walking | Standing |
| --- | ---: | ---: |
| Noise 0.5, the cycle-5 start | −32.5 | −26.1 |
| Noise 0.25, cycle 6's start | −0.9 | −23.5 |

At the old spread standing would beat walking; at 0.25 walking wins by 23
mm/s, about two thirds of the clean walk's pay.

**Energy sharing, by sides or equal?** The user asked whether every
follower should share its health equally with all the neighbours it sees.
`analysis/sharing_check.py` in `../Training` (the scripted walk with one
segment limp; energy per second as that segment pays it):

| Limp segment | Walking | Limp, by sides | Limp, equal |
| --- | ---: | ---: | ---: |
| Middle (5) | 0.69 | 0.45 | 0.60 |
| Head | 0.82 | 0.65 | 0.72 |
| Rear | 0.74 | 0.77 | 0.87 |

Equal sharing makes going limp save less, but each segment's own energy
becomes a fifth of its health, the rest its neighbours' noise, so its own
actions show less in what it learns from. Hanging now charges going limp
directly, so sides were kept: itself 40%, as cycle 5.

**Risks.**

- The rear followers feel no squeeze, so they learn not to push only
  through the chain: their push reaches segments 1 to 3, who must brake
  it, and the shared energy of braking. If the rear push, the front three
  pay.
- A limp follower keeps its distance; hanging must stay dearer than the
  energy it saves (by sides a middle follower saves 0.24 mJ/s; held up by
  about one unit weight, as cycle 5 measured, it pays about 2 mJ/s).
- The head's neck is unpaid: a head walking crabwise is not charged.

### Cycle 7: the head's line (the user's design)

**Why** (2026-10-16, runs 11 and 12, stopped by the user at cycles 41 and
15). The followers learned to brake. The squeeze charged only a front
joint pushed together, so a follower pulling back on the segment ahead
paid nothing; with every follower braking, the squeeze fell sixfold and the
body drifted backward, which cost nothing either, since the distance was
multiplied by the trail's quality, about 0 for a body walking against its
trail. Measured afterwards (`analysis/brake_price_check.py` in
`../Training`, the scripted walk under noise 0.25): one follower braking
alone gained 12 mm/s over walking. Pricing the front joint both ways closed
that, but the user stepped back: every price on pushing or pulling had
found a new loophole, and a trail that each segment's own front end draws
is redrawn by walking backward.

**The design** (the user's, agreed 2026-10-16; the scales measured below):

- **The head's line.** The head's tip draws a line on the floor. Every
  follower has its own copy, which its tail erases as it passes, wherever
  the tail is: a copy runs from the furthest point its tail has reached to
  the tip. The line is the head's, so no follower can redraw it, and each
  follower sees it where it lies, inside its observation sphere (the user,
  2026-10-16: the radius is a sphere around the agent, and everything
  inside it is visible, never the target).
- **Paid as erased.** A follower is paid the millimetres of line its tail
  erases. Ground covered once is never paid twice, so the sway of a stride
  and the jitter of the noise earn nothing.
- **Charged as left behind.** Its tail's distance from the nearest point of
  the line it has not erased: beside the line, the distance sideways;
  behind its end, the distance back to it. Walking backward leaves the
  marker by itself, with no rule about direction.
- **The front joint kept free of force.** The force at a follower's front
  joint is its own surplus push or drag plus that of everything behind it.
  Keeping it at zero says all of the user's rules at once: cooperate with
  what comes from ahead, resist a push from behind, pull along one that
  drags, and pass nothing forward. The one that misbehaves always shows it
  at its own front joint; the one that resists keeps its joint clean and
  pays only the energy. Pushed or pulled cost the same.
- **One balance, squared.** goal = line erased − 40 mm/s × [(d / 3 mm)² +
  (F / 0.3)²]: being 3 mm off the line, or 0.3 unit weights at the joint,
  costs a whole walk's pay. Small errors are almost free and large ones
  dear (the user's preference); the distance is counted up to 10 mm, so
  that a few lost worlds do not swamp an update's advantages.
- **The head** is paid its target and arrivals alone, with no force price
  (the user's decision); its health is everyone's.
- **Unchanged:** health (energy shared by sides, pain, hanging), the two
  critics at 1 : 0.8, the spine at 0.35, the starting noise 0.25 pink. The
  trail of cycles 4 to 6 leaves the code; each block's trail offset gives
  way to its tail's place on the head's line
  ([observations.md](observations.md)).

**Measured before building** (`analysis/marker_check.py` in `../Training`:
a prototype of the line on the scripted wave walk, v4, CPU, one world,
noise 0.25 unless clean; the goal per second, mm/s, under the chosen
balance and, for comparison, a linear one of 3 mm/s per mm and 100 mm/s per
unit weight):

| Case | Followers, squared | Segment 5 alone, squared | Followers, linear |
| --- | ---: | ---: | ---: |
| Walking, clean | 28.1 | 12.3 | 27.5 |
| Walking | −4.9 | −18.6 | 11.1 |
| Standing | −111.6 | −101.4 | −19.5 |
| Everyone backward | about −460 with the cap | | −78 |
| Followers push a slow head | −8.5 | | 10.4 |
| A lazy head holding, pushed by the followers | −11.6 | | 4.1 |
| A lazy head limp, pushed | −17.4 | | 7.7 |
| The head walks, the followers hold | −92.4 | | −17.9 |
| Segment 5 brakes, the others walk | | −67.8 | |
| Segment 5 limp | | −43.5 | |
| Segment 5 steps too fast | | −38.3 | |

Walking wins in every case, for the group and for one follower alone,
whose every deviation loses at least 20 mm/s (7 with the linear balance,
under which pushing a slow head nearly matched walking). Under the noise
every value is negative; only the differences teach. Two more findings: the
front joint force peaks in the middle of the body (about 0.2 unit weights
at segment 5), not at the head, so the head's end is not overloaded; and a
noisy walk leaves its tails 2 mm off the line, a standing body drifts 5 mm,
a body walking backward 20 mm and more. Centipede's `HeadLine` reproduces
the prototype within 0.0001 mm over 300 steps, walking and backward.

**Risks.**

- **A lazy head** (the user's concern): the followers score −4.9 with a
  head that walks and −11.6 to −17.4 pushing a lazy one, but standing is
  worse still (−111.6), so a lazy head would be carried. Watch the head's
  net force (pushed forward), its feet and its energy. The user's idea for
  later: the tip draws only while the head walks by its own effort.
- **Turns cost the followers** (−18.1 in an even turn under noise): they
  cannot avoid them, so it should teach nothing wrong.
- **The scripted walk is not balanced**: its middle joints carry 0.2 unit
  weights of pull. A learned walk must even out the efforts, the
  cooperation sought.
- **A head walking backward** pushes every follower off its line.

### Cycle 8: only a lasting force costs (the user's idea)

**Why** (2026-10-16, runs 13 and 14, 128 cycles each). Cycle 7 found no
loophole: the followers stayed on the line and freed their front joints,
and the head was never carried. In training the targets rose tenfold, from
0.05 to 0.5 an episode between cycles 66 and 128, but without noise the
body stood: the exploration noise was the gait
([actions.md](actions.md#how-actions-explore)). And 95% of the front
joint's cost was that noise's jitter, not a steady push: at segment 5 the
averaged force's mean was −0.02 to +0.06 unit weights, its root mean square
0.26. The cost charged every random shove of exploring, so the agents
learned that moving costs.

**The change** (the user's idea): charge a force by how hard and how long
it lasts, not by its instants. The forces are averaged over 1 s
(`averaging_s` 1) instead of 0.2 s; the prices are unchanged. A walk's own
push and pull, which alternate within a stride, then cancel, and a segment
that keeps pushing or dragging still pays in full. The average serves every
averaged force, so the hanging cost too charges a lasting hang, and each
segment observes the same averages it pays.

**Measured** (`analysis/exploration_check.py` in `../Training`, the
followers' front joint cost per follower in mm/s, group mean):

| | Averaged 0.2 s | 1 s | 2 s |
| --- | ---: | ---: | ---: |
| Run 13, cycle 128, its pink noise | 17.7 | 5.3 | 2.9 |
| Run 14, cycle 128, its pink noise | 22.2 | 8.2 | 5.4 |
| The same with red noise | 32.4 / 71.4 | 14.7 / 51.6 | 10.3 / 39.7 |
| The scripted walk, clean | 9.9 | 9.1 | 8.7 |

Over 1 s the noise's cost falls by about 70%, about to what the standing
policy paid over 0.2 s (5.8 and 9.6); red noise costs two to three times
pink at every window, so it stays out. The scripted walk's own cost hardly
changes: 95 to 99% of it is a steady push of its unbalanced gait, which a
learned walk must even out.

### Cycle 10: each segment pays its own ground force (the user agreed)

**Why** (2026-10-16, after runs 13 to 18). The front joint's force has the
wrong owner. MuJoCo's joint forces make it exact: the force at a segment's
front joint is the force its subtree needs, so T_k = N_k + N_k+1 + … +
N_11, where N is each segment's own net joint force, its front joint's
force less its rear one's. Each follower paid for every error behind it,
the middle ones most (the tug of war), and a follower walking in front of
one that drags paid that drag: walking beside a brake was punished, which
feeds the all-standing trap. Measured on the scripted walk
(`analysis/tow_check.py` in `../Training`; a follower's goal per second,
stepping / holding, the steppers spread along the body):

| Steppers, tow | Front joint, f | Own net, N | Own ground, G = N + tow |
| --- | --- | --- | --- |
| 8 of 12, none | −43.9 / +4.1 | −10.0 / −32.3 | as N |
| 3 of 12, none | −3.9 / −13.1 | −36.4 / −5.6 | as N |
| 3 of 12, 0.3 | −211 / −43 | −115 / −14 | −20.4 / −93.1 |
| 3 of 12, 0.5 | −866 / −275 | −191 / −3.7 | −7.1 / −154.8 |
| 8 of 12, 0.3 | −347 / −39 | −20.9 / −38.2 | +26.5 / −182 |
| 8 of 12, 0.5 | −759 / −235 | −20.1 / −16.4 | +45.5 / −304 |

Under the front joint's price holding beside walkers paid more than
stepping, with or without a tow: the braced stance of runs 13 to 18's mean
action. One segment holding while all others walk drags its neighbours;
under the front joint's price the cost lands on the walker behind it (1.66
unit weights at segment 6 against 0.88 at 5), under the own ground force on
itself (0.54 unit weights: −76 mm/s against its neighbours' +24).

**Why the tow must be added back.** A segment's own net force along the
body is its momentum's change less the floor's force and its tow. Under a
tow t, a held segment dragged by the moving body has N = 0.8 − t, and a
segment stepping in time with the body, neither braking nor pushing, has
N = −t: at t = 0.4 both pay the same, and above it holding is cheaper.
Standing still costs nothing under N at any tow, since its feet hold the
tow back exactly. Adding the tow back prices what the segment's own feet
and body do: holding and sliding cost 0.8 at any tow, a step in time with
the body is free, standing still costs t, and a pusher pays its push.
Without a tow the two prices are the same.

**The change.** With `force_price = "own_ground"` a follower's force cost
uses its averaged own net force along the body plus its tow, at the same
scale (0.3 unit weights) and pay (40 mm/s); the front joint is free. The
head keeps no force price: when every follower carries itself, every joint
force is zero and the head neither pulls nor is pushed. The rest is cycle
8's. The tow itself is [tow.md](tow.md).

**Measured as a price.** Under the own ground force stepping beats holding
at every spread placement from a tow of 0.3; without a tow holding still
wins while four or fewer segments step, and stepping wins from six: the
trap at a standstill, and a walk that sustains itself once found. A
scripted walk of every segment under a tow of 0.3 to 0.7 brakes (G +0.21
to +0.46) and still scores +58 to +65 mm/s; a learned walk would push
less. In a body holding still under a tow the soft contacts share the tow
unevenly along the chain (its own ground force 0.24 to 0.76 at t = 0.5),
which reads in the logs but changes nothing in the design.

### Cycle 11: each segment brakes its own tow (the user agreed)

**Why** (2026-10-16, run 19's first 47 cycles). Someone must brake the
tow. In a towed body walking at a steady speed, nothing but the feet holds
the body back, so the tows' total, 12t over the twelve segments, must
leave through feet that brake. The own ground force G = N + t charges a
follower whose feet brake its own tow (N = 0, G = t) and nothing for one
whose feet do nothing (G = 0): that one passes its tow to its neighbours,
N = −t. Cycle 10's free state, a segment stepping in time with the body,
neither braking nor pushing, is that same state. The head pays no force
price, so the followers' cheapest arrangement passes the tows forward until
the head brakes them: with every follower passing, the neck would push the
head with 11t.

Run 19 showed it on its first checks (`review.py check` in `../Training`,
seed 2's training worlds at cycles 44 to 47, whose tows average 0.2):

| Measure | Run 19 |
| --- | --- |
| Segment 1's averaged front joint force, its neck on the head | −0.29 unit weights: pushing the head |
| The followers' averaged net joint forces | about 0 to +0.08 at segments 1 to 7; −0.13 to −0.17 at 8 to 11, near −t |
| The head's feet | braking, −0.27 unit weights |
| The head | heading error 134 to 150°, on course 1%, closing about −20 mm/s |

The rear passed its tow forward already; the middle braked its own and paid
G ≈ t, the largest own ground costs (30 to 53 mm/s at segments 3 to 7), so
its gradient pointed to the rear's state. A head pushed from behind cannot
steer, as run 04's walked straight past its targets.

**The change.** With `force_price = "own_joint"` a follower pays its
averaged own net joint force along the body alone, N, at the same scale
(0.3 unit weights) and pay (40 mm/s); the front joint is free. Passing the
tow on costs t², braking it with one's own feet is free, so every tow is
braked where it acts and none reaches the head. The goal's sixth part, the
follower's own force price, is named `own_force` under either price.

**What it gives up.** Cycle 10 added the tow back because under N a dragged
holder pays (0.8 − t)² and a passive stepper t²: they tie at t = 0.4, the
tow's top. Below it stepping is the cheaper, and the training worlds draw
their tows from 0 to the top, which falls as the walk improves. Standing
still is free at any tow, since the feet hold the tow back exactly, so the
tow no longer prices a standstill; but the noise moves a towed body anyway
(17 to 37 mm/s at tows of 0.3 to 0.5, [tow.md](tow.md#risks)), and a
moving body drags its holders, who then pay. On run 16's checkpoints a
segment's own exploring still paid under N at a tow of 0.4
(`analysis/own_step_check.py`, segments 1, 5 and 10, mm over 3 s): +10 ± 4,
+58 ± 19 and −5 ± 8 at cycle 32; +55 ± 11, +56 ± 24 and +10 ± 12 at cycle
128, against +45, +141 and +18, and +137, +105 and +29 under G.

Without a tow the two prices are the same, so the evaluations, which never
tow, price alike; only what the tow teaches changes. Under N each segment
learns to brake its own tow, which turns into carrying itself as the tow
fades; under G the followers learn a walk that leans on the head.

### Cycle 13: the shared reward (the user's design, 2026-10-17)

**Why** (after runs 19 to 22). The reward has no loophole left, so the
only way to a good reward is a wave in time, and the wave does not emerge:
the mean action stands in every run. The user named two causes, a reward
that is individual in a system of many agents, and pain, which keeps each
segment's legs in their own space before they can learn where to move.
Both were measured.

**A step's gain is shared, its bill is not.** Paired worlds as in M2
(`analysis/reward_shape_check.py` in `../Training`: common random numbers,
40 pairs, segments 1, 5 and 10 in turn acting with their mean for 3 s,
pooled), on run 22 seed 1's checkpoints, without a tow. The effect of one
follower's exploration over the 3 s:

| | Cycle 64 | Cycle 128 |
| --- | --- | --- |
| The line its own tail erased | +5.2 ± 1.5 mm | +5.6 ± 1.3 mm |
| Its own goal costs, off the line and its force price | −7.5 ± 3.4 mm | −3.7 ± 2.4 mm |
| The line the whole body erased | +51 ± 15 mm | +64 ± 14 mm |

The chain shares a step's gain by itself: every follower's tail moves with
the body, so each erases about the same line, and the segment that stepped
sees a twelfth of what its step did. Its bill, its own force price and its
distance from the line, it pays alone. A step that helps the body is, for
the segment that makes it, a loss or close to none, and so is every
other segment's: standing still is each one's best answer. This is the
public goods dilemma of cooperative learning: a pure team reward breeds lazy
agents, a purely local one selfish agents (Mao, Gong and Xiao, 2020).

**Reward shapes compared**, with Agogino and Tumer's two measures of a
multi-agent reward (JAAMAS 2008): *factoredness*, how often a segment's
reward moves the way the body's does (here the share of pairs in which a
segment's exploration changes its reward and the body's, the sum of every
segment's own, with the same sign); and *learnability*, how clearly a
segment's own effect stands out of everything else that moves its reward
(here the mean size of the effect over the reward's spread across the
pairs). On run 22's cycle-64 policy, and in brackets on fresh networks, as
a run starts:

| Shape | Goal: factoredness, learnability | Health: factoredness, learnability |
| --- | --- | --- |
| As built | 78%, 0.82 (72%, 0.66) | 86%, 1.82 (62%, 0.88) |
| The whole reward shared like energy (itself 40%, each side 30%) | 82%, 0.77 (77%, 0.66) | 89%, 3.40 (65%, 1.09) |
| Half own, half the body's mean | 92%, 0.80 (88%, 0.73) | 88%, 1.96 (74%, 0.94) |
| The body's mean | 100%, 0.78 (100%, 0.75) | 100%, 1.68 (100%, 0.94) |

Factoredness carries an error of about ±3 to 4 points. Sharing with the
neighbours aligns the goal only a little: their weights add up to one and
every follower erases about the same line, so it captures no more of a
step's gain; it only lowers the own share of the bill from all of it to
40%. The gain of a step is the whole chain's, so its bill must be the
whole chain's too. Learnability of the goal does not suffer from the
body's mean: its main part is the line, and every follower feels the
body's line through its own tail, which its critic sees. Health is
different: energy, pain and hanging are local, set by a segment and its
neighbours, and shared like energy they are about twice as learnable, the
neighbours' noisy terms averaged out while the own share stays 40%.

**Pain teaches sweeping less before timing.** In run 22's training worlds,
pain was half of every segment's health at the start and was learned away
first:

| Cycle | 1 | 8 | 16 | 32 | 64 | 128 |
| --- | --- | --- | --- | --- | --- | --- |
| Pain's share of health | 52% | 44% | 24% | 15% | 5% | 0.7% |

The scripted wave walk at 2.5 Hz (`analysis/pain_check.py`, one world, 5 s)
pays pain per segment as follows:

| Gait | Pain | Steps with legs touching | Speed |
| --- | --- | --- | --- |
| Wave over 5 segments, sweep 20° | 0.09 mJ/s | 3.6% | 36 mm/s |
| The same, sweep 30° | 0.51 mJ/s | 18.5% | 51 mm/s |
| Wave over 3 or 12 segments, sweep 20° | 0.26 to 0.33 mJ/s | 10 to 12% | 33 to 38 mm/s |
| Every leg of a side in phase | 0 | 0% | 1 mm/s |
| Random actions, spread 0.25 | 0.83 mJ/s | 31% | 0.5 mm/s |

A well-timed wave costs almost no pain, and standing costs none; wide
sweeps and random legs cost much. Pain selects the wave's timing, as
designed ([Pain](#pain-legs-hitting-legs-the-users-design)), but a sweep is
painless only when the neighbours' timing is right, and at the start it is
random: the legs learn to sweep less long before they could learn when. On
fresh networks, pain is also health's noisiest part: shared like energy,
health's learnability is 1.09 with pain and 2.52 without.

**The design** (the user's, agreed 2026-10-17):

- **The goal, the body's.** Every follower is paid the followers' mean
  goal, part by part: the line, its distance and its force price
  (`goal_team_share` 1). A step's gain and its bill are then shared alike,
  and a segment feels a twelfth of each. The head keeps its own goal: it
  alone sees the target, and its steering would weigh a twelfth in a goal
  of the body's mean; the followers take nothing of its target progress,
  which their critics do not see (the pairing rule).
- **Health, the neighbourhood's.** Pain and hanging are shared with the
  neighbours as energy is, itself 40% and each side 30% at radius 2
  (`health_sharing = "all"`). Pain stays mutual inside the share.
- **Pain, ramped in.** Pain's weight rises evenly from 0 in the first
  update cycle to full at cycle 129 (`pain_ramp_cycles` 128), a schedule of
  cycles as the first try; a ramp moved by measures of the gait is for
  later. Hwangbo et al. (Science Robotics 2019) raised every cost from 0.3
  of its weight for the same reason, a high cost on motion making standing
  a good local minimum. Evaluations count pain in full.
- **No tow, gSDE as before.** The test is the reward alone (the user):
  pink noise and the tow return once cooperation is seen.

Cooperative learning pays this way: one outcome of the team for all, the
hexapod's legs the body's velocity (Schilling et al., 2020), Multi-Agent
MuJoCo's segments the robot's reward, StarCraft's units the damage, kills
and wins of the team; OpenAI Five mixed each hero's reward with the team's
mean, the team's share raised from 0.3 to 1 in training (Berner et al.,
2019).

**What it gives up.** A follower that does nothing is paid the same goal as
one that walks: the lazy agent of a shared reward (Sunehag et al., 2018).
Its own costs still bind it in health (40%), and its goal costs a
twelfth. Hanging, shared, costs the hanger 40% of itself: the tail's
escape of run 22, hanging on segment 10, may grow, and the per-segment
reads (feet carry, stance) watch it. With pain at zero at first, legs may
push off other legs, which pain exists to forbid; the leg contacts are
logged. And the parts logged per segment are as paid, after sharing: the
followers' goal parts are one value, and each segment's own measures are
read from the step facts (its line, off-line distance, joint forces, feet
push and carry).

### Cycle 14: the team's and each one's own (the user's split, 2026-10-17)

**Why** (run 23). The shared reward made the noisy body walk without a
tow, and seed 2's mean action walked: the followers erased 7.2 mm/s of
line, every leg stepping 1 to 4.4 times a second. But it shared too much.
A follower's own force cost was paid as the followers' mean, so a segment
that braked paid an eleventh of its brake: by cycle 256 segments 1 and 3
braked (their feet −0.27 unit weights) while 7 and 8 pushed (+0.12 and
+0.22), squeezing the chain between them (front joints −0.37 to −0.58).
Hanging, shared like energy, cost the hanger 40%: seed 1's tail hung, its
feet down 2 to 5% of the time. In the recordings some legs did not move and
were dragged, and the legs behind overlapped them. The user's principle:
every segment has a duty to the team and duties of its own. What the body
achieves and spends is shared; what a segment does to its neighbours, and
what it suffers, is its own.

**Who pays what** (the user's split):

| Who pays | Term | Why |
| --- | --- | --- |
| The team, the same for every follower | The line erased and the distance from it, the followers' mean | The chain moves together: every tail erases about the same line, and where a tail lies depends on the whole body's shape. In run 22 the distance was nine tenths of a stepping follower's own bill (−7.2 of −7.5 mm over 3 s at cycle 64) |
| The team, the same for every segment | Energy, the body's mean | One animal's budget |
| The head | Its target and arrivals | It alone sees the target |
| Each segment, its own | Its force share: its joints' averaged net force along the body, pushing or dragged | The exact split of the chain's force (below) |
| Each segment, its own | Hanging: held up by its neighbours (the head beyond half its weight), or lying on the floor | Being carried is the carried one's doing (the user) |
| Each segment, its own | Pain, both segments of a contact paying, from half its weight at the first update cycle to full at cycle 129 | Under run 23's ramp from zero, legs overlapped (the user) |

**The force share.** The force at segment k's front joint is the sum of
the net joint forces of every segment from k back: T_k = N_k + N_(k+1) +
… + N_last (cycle 10). The chain's push or pull at any joint is built of
the shares behind it, and for a sum every fair split of the credit gives
each its own term, so N_k is segment k's share of the force the body
carries at every joint ahead of it. Each segment sees both: its share in
its own block, and the chain's force at the five front joints of its
window, its own front joint summing everything behind it
([observations.md](observations.md)). Pushing and being dragged cost the
same, squared, at cycle 7's price (40 mm/s at 0.3 unit weights): if braking
against a push were free, the fight of run 23's seed 2, the front braking
against a pushing rear, would be free for one side of it. A segment that
lets a push through pays nothing; the one that pushes pays.

**Two critics, by owner.** Every segment learns from a team critic and an
own critic, weighed 1 : 0.8 as goal and health were, each advantage
normalised on its own ([critic.md](critic.md)). Inside a channel the terms
speak in proportion to their sizes; across channels, a channel's weight and
how clearly a segment's own actions show in it decide.

**One exchange rate, 10 mm per mJ.** Inside a channel millimetres and
millijoules are added, a millijoule counting as 10 mm of line
(`mm_per_mj`). In the team channel the reference walk's energy, 0.78 mJ/s
for a segment (9.3 for the body, [As built](#as-built)), costs about a
fifth of the line it erases, 8 against 36 mm/s; standing raised costs
5 mm/s and resting nothing, so walking stays first. In the own channel,
hanging and pain are priced as force, in millijoules: a unit weight held
up, or a leg pressed with a full push, costs 20 mm/s, as being dragged by
0.2 unit weights does. Measured on run 22's policy, the force share's
spread across worlds over 3 s is 1 to 5 mm at segments 1, 5 and 10,
hanging's 2.5 to 6 mm at this rate; on fresh networks the force share's 6
to 40 mm, pain's, at half, 7 to 51 mm. At 1 mm per mJ pain and hanging
would be lost beside the force share.

**Energy was the loudest voice against exploring.** Measured on cycle 13's
paired worlds (the sums `analysis/reward_shape_check.py` in `../Training`
saved: fresh networks, run 22 seed 1 at cycles 64 and 128; segments 1, 5
and 10 pooled). A channel's pull is how much a segment's exploring for 3 s
changes it, against acting with its mean, in units of the channel's spread
across worlds; with every channel normalised on its own, it is about the
push the channel gives exploring, before its weight:

| Channel | Fresh | Cycle 64 | Cycle 128 |
| --- | ---: | ---: | ---: |
| Run 22's goal: its line, less its distance and force costs | +0.08 | −0.05 | +0.07 |
| Run 22's health: energy shared with the neighbours, its pain and hanging | −0.57 | −1.77 | −1.69 |
| Run 23's goal: the followers' mean | −0.07 | +0.04 | +0.27 |
| Run 23's health: all shared with the neighbours | −0.86 | −3.40 | −3.40 |
| Its own energy, alone | −13.2 | −9.3 | −6.1 |
| Team: the followers' mean line and distance, the body's energy at 10 mm/mJ | −0.10 | −0.03 | +0.18 |
| Own: the force share, pain at the new ramp's weight, hanging | −0.08 | −0.08 | −0.21 |

Exploring costs a segment energy in every pair, since noise always adds
force: alone in a channel, energy is the clearest signal a segment
receives. At cycle 128 run 23's health pulled against exploring thirteen
times harder than its goal pulled for it. Cycle 13 had doubled health's
learnability by sharing it with the neighbours, which made it clearer in
that same direction. Weighed 1 : 0.8, the new channels give exploring a
push of about zero (−0.16, −0.09, +0.01), against −0.38, −1.47 and −1.28
for run 22 and −0.76, −2.68 and −2.45 for run 23. The line alone would pull
for exploring (+0.38 to +0.53), but the distance from it, shared, adds its
noise to the team channel.

**Checked against the closest projects** ([research.md](research.md#team-and-own)):

| Choice | Done the same way |
| --- | --- |
| The task's gain shared | Multi-Agent MuJoCo gives every agent of one robot the robot's single reward; the hexapod's legs the body's velocity; MASQ's four legs one reward |
| Energy shared, in the same reward as progress, at a fixed rate | Multi-Agent MuJoCo: ManyAgent Swimmer pays Δx/Δt − 0.0001 ‖u‖², every agent's commands in the one team reward; MASQ's torque costs are summed in its shared reward |
| Each agent's duties its own, as costs | The bodyguards of the escort problem each keep their own distance, 0 or −1, beside the team's reward (Sheikh and Bölöni, 2020); each quadruped carrying a box keeps its own place on it |
| One critic per owner, weighed | Sheikh and Bölöni: a team critic and a local critic for every agent, their gradients added; one critic of the mixed reward swung between the two |
| Coordination through the force each feels | Ants carrying food align their pull with the force they feel through the load (Gelblum et al., 2015); robots carrying an object read its motion at their own grip (Wang and Schwager, 2016); trains with a motor in every car (Jackiewicz, 2021) |

Where it differs: those team critics see every agent (MADDPG's, MASQ's);
ours sees its neighbourhood, so the team's far terms are noise to it, the
accepted limit of cycle 13 ([critic.md](critic.md)). And no project prices
each agent's share of a force; the share comes from the physics.

**What it gives up.**

- A segment that wastes energy feels a twelfth of it, so energy will shape
  the gait late. Laziness is caught by the own duties instead: a segment
  that does not keep up is dragged and pays its share; one that hangs pays
  its hanging.
- The force is averaged over 1 s, so it teaches carrying oneself, not when
  to step: the wave's timing is still left to pain, support and speed.
- An individual reward that conflicts with the team's can hurt it (Wang et
  al., ICML 2022). At the team's best, a wave walking on the line with
  every segment carrying itself, every own duty is about zero; they
  conflict with the team only on the way there, as costs of exploring,
  measured above.

### Cycle 15: the line where the head points, and the squeeze (the user's design, 2026-10-17)

**Why** (run 24). Seed 2's mean action walked a body length a second with
the work shared and no segment hanging, but the head never steered
(closing on its targets at −3.7 mm/s, its heading 98° off) and was pushed
by the body. Measured on scripted walks, fresh networks and run 24's
checkpoints (the scripts in `../Training/analysis`):

- **The force** (`chain_force_check.py`). A clean scripted wave walk pulls
  every joint a little, +0.02 to +0.21 unit weights averaged over 1 s, and
  never squeezes one. Run 24 squeezed its front joints, −0.13 to −0.17, 95%
  of the time, while every segment's own force stayed near zero: the own
  price rated run 24's walk (0.3 mm/s) above the clean walk (1.2). Most of
  that squeeze came from the crooked front, sideways forces turned along the
  body by the bent joints (about 70%) and the weight carried through tilted
  ones (about 40%), not from pushes along the body; no own force sees it.
- **Turning** (`follow_check.py`, `turn_force_check.py`). The line paid the
  followers for resisting the head. The head alone cannot turn the body: with
  its neck bent 10° and the followers holding straight it turned 4°, so the
  line stayed straight and the resisting followers followed it best, 34 to
  36 mm/s against 28 for following. In run 24 the neck's bend never
  travelled back (each joint's bend against the neck's earlier one:
  correlation about 0 from joint 2 on). Forces do not tell a resisting body:
  it pushes and drags nothing, it goes the wrong way. A clean turn costs no
  force once averaged over 1 s: its step peaks, up to 1.6 unit weights,
  alternate and cancel. A body that follows also needs its legs to steer,
  the inner stride shorter: without it even perfectly following joints skid,
  the head slipping 7° sideways.
- **The ledger** (`ledger_check.py`, run 24's environment driven by a
  scripted well-behaved centipede and by each fault). Under cycle 14 the
  followers earned more for resisting the head (+21.5 mm/s, team + 0.8 own)
  than for walking well (+7.4), and the distance cost made exploring a
  bigger loss than standing (−41 to −138 mm/s per follower, at spreads 0.25
  and 0.4).

**The design** (the user's, 2026-10-17):

- **The line paid near it.** A follower's line is paid as the line its
  tail erases times its closeness, max(0, 1 − (d / 5 mm)²), d its distance
  from the line it has not erased; the distance cost goes. One term instead
  of two, never negative: exploring and standing earn about 0 instead of a
  large cost, a clean walk 0.5 to 2 mm from the line keeps 85 to 99% of its
  pay, and a tail beyond 5 mm earns nothing.
- **The line where the head keeps pointing.** The head's sideslip, the
  angle between where it points and where its centre goes, both averaged
  over a step cycle (0.4 s), moves the line: the tip draws it to the side
  by a drift that grows by the head's speed times the sine of the sideslip
  each step and leaks away over 2 s. A rigid head following a curve has no
  sideslip, so a body that follows leaves the line at the tip; a body that
  ignores a bend the head holds leaves the head slipping, and the line
  moves away from it the longer it does, while a wag or a quick swing to a
  new target leaks away. No tolerance: a 5° one let run 24's sloppy steering
  (4.6° of sideslip on average) tie a good walk.
- **The squeeze, the team's.** Every follower pays alike the squeeze of the
  chain: at every joint, what the legs push, the force along the body with
  the tows of the segments behind it added back, a pull free, squared
  against the force share's scale (40 mm/s at 0.3 unit weights) and
  averaged over the joints. Whoever pushes, everyone pays: a push is an
  error that reaches the whole chain (the user). The healthy slight pull is
  free, and a tow's own push is not charged, so braking it gains nothing.
- **Ramps.** Pain, energy and the squeeze rise from half their weight at
  the first update cycle to full at cycle 129 (`ramp_team_costs`).
- **Nothing on the legs.** A lifted leg gains its segment nothing and costs
  it a little (its share and hanging); a rule on how legs step would
  prescribe the gait (the user).

The biology of a turn: [research.md](research.md#turning).

**Measured** (`ledger_check.py`, 7 s, each follower per second, team +
0.8 own, at full weights; the culprit's own total where one segment errs):

| Behaviour | Cycle 15 | The line as in cycle 14, paid near it |
| --- | ---: | ---: |
| A well-behaved centipede | +13.3 | +19.0 |
| Run 24's walk, fast and not steering | +4.7 | +23.5 |
| The followers resisting the head | −2.1 | +25.4 |
| Standing | −11.1 | −11.1 |
| Limp | −24.8 | −24.8 |
| Legs almost in step (the wave over 12) | −42.5 | −41.5 |
| The head holding, the body pushing it | −97.4 | −97.3 |
| The braking segment, its own total | −100 | |
| The dragged segment, its own total (+17 walking well) | +2 | |
| The segment with a leg held up, its own total (+14 walking well) | +14 | |

A well-behaved centipede wins against every fault and against standing;
the lifted leg is a near tie that gains nothing. Under a servo tow of 0.2
the well-behaved walk earns +19.9 (the tow is off at its speed), standing
braking it −26.6 and lying limp riding it −33.3; braking a tow of 0.4
−74.9: braking is the dearest answer to a tow. Fresh networks at spread
0.4 under a tow of 0.2 erase 10.2 mm/s of line a follower against 4.1 at
spread 0.25 and 0.3 with neither ([tow.md](tow.md)).

### Cycle 16: energy shared with the neighbours, pain averaged (the user's design, 2026-10-18)

**Why** (run 25). Both seeds' mean actions walked to their targets for the
first time, 2.7 to 4.5 targets an episode at the hardest level, but limped:
some legs held up or used as props while others stepped at 5 to 7 Hz, and
the rear did little. Measured (the scripts in `../Training/analysis`):

- **Energy's credit** (`ledger_energy_check.py`). As the body's mean in
  the team channel, energy gave each segment a twelfth of its own, mixed
  with eleven others' noise, and broke the pairing rule: a critic sees five
  segments, the mean depends on twelve. Energy alone cannot be the own
  cost: a dragged or limp tail, a held-up leg and a limp rear half gain
  under energy and pain alone (+0.2 to +9.6 mm/s for the culprit), because
  a segment carried by its neighbours saves its holding force, 85% of its
  energy, while dragging it costs them little friction at walking speed.
  Hanging closes those holes; the force share keeps braking and being
  dragged dear (the tail braking −51 mm/s, dragged −61).
- **The critics' scale** (`critic_check.py`). Each critic standardised its
  targets by every target since cycle 1. Late in run 25 the followers' team
  targets spread over 0.29 to 0.50 of the stored scale and the head's own
  over 0.08 to 0.20, their mean 1.4 stored spreads off: the scale lagged the
  returns, which drift about 1% an update (team) and 0.3% (own).
- **Pain's bursts** (`critic_check.py`). The rarest 1% of steps held 92 to
  100% of pain's variance, a rare collision squared; under the design below
  pain would make 74 to 100% of the own channel's spread in segments 3 to 7,
  a single collision setting a whole batch's own advantages.
- **One normalisation of the summed advantages** (`critic_check.py`), once
  proposed for the head's nearly empty own channel, would leave the own
  channel 2 to 16% of the advantage variance and cut its pull on the policy
  five- to tenfold against run 25's, where per-critic normalisation weighed
  an own millimetre as 1.5 to 9 team millimetres. Not taken: with energy in
  it, no own channel is empty (the head's is 99 to 100% energy).

**The design** (the user's, 2026-10-18):

- **Energy, each segment's own, shared by neighbourhood**
  (`energy_side_shares = [0.25, 0.05]`): 40% of its own, 25% of each nearest
  neighbour's and 5% of each next one's, a missing side's shares going to
  the other side (the head 40% itself, 50% segment 1, 10% segment 2; the
  tail likewise). A segment pays for what it spends and for what it makes
  its neighbours spend, within what its critic sees; the team channel keeps
  no energy. Still ramped in with pain.
- **The own channel keeps the force share, pain and hanging**: energy
  alone leaves the rear's holes open.
- **Pain on the averaged press** (`pain_averaged`): the force with which a
  segment's legs press on other legs, averaged over 1 s like every other
  priced force, squared. A lasting press costs as before; a brush of one
  step costs 0.02² of what it did.
- **The critics**: each one's scale an average over about the last 10
  updates ([critic.md](critic.md#as-built)); the advantages still
  normalised per critic, 1 : 0.8.

**Measured** (`ledger_cycle16_check.py`: run 26's environment as built, 16
worlds, 7 s, a scripted centipede steering through its turns with a wave
over 5 segments, head to tail; each fault's culprits' total, team + 0.8
own, minus the same segments' in that walk, mm/s; in brackets at an own
weight of 4.6, the largest boost per-critic normalisation gave run 25):

| Behaviour | Culprits | Total minus the good walk |
| --- | --- | ---: |
| The followers holding straight through the turns | followers | −18.3 (−7.5) |
| The head holding still | head | −38.8 |
| Legs almost in step (a wave over 12) | followers | −66.2 |
| **The wave running backward, tail to head** | followers | **+4.9 (+12.5)** |
| Legs in random phase | followers | −22.1 |
| Everyone standing | followers | −25.8 (−18.4) |
| Everyone limp | followers | −42.4 |
| Segment 5 braking | segment 5 | −136 |
| Segment 5 dragged | segment 5 | −10.3 |
| Segment 6's left leg held up | segment 6 | −2.3 (−24.6) |
| The tail braking | tail | −54.9 |
| The tail dragged | tail | −61.7 |
| The tail's left leg held up | tail | −20.6 |
| The rear half braking | segments 6 to 11 | −39.6 |
| The rear half limp | segments 6 to 11 | −66.6 |

The well-behaved walk earns each follower +27 mm/s from the team and pays
10 to 18 of its own; at the ramps' start, pain, energy and the squeeze at
half, the order is the same. Every fault pays less, the held-up leg of
segment 6 least so. The one exception is no fault: a wave running the other
way, tail to head, walks this body faster (line +6 to +7 mm/s a follower
over three sets of starts, +4.9, +7.2 and +8.3 in all, with slightly lower
own costs). Scolopendromorph centipedes walk with waves running head to
tail, scutigeromorphs, lithobiomorphs and millipedes tail to head; why an
animal chooses one is not well understood (Kuroda et al., 2014, as cited by
Ambe et al., 2021).

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
segments of a contact, and each follower's path is its own (until cycle
13, which shares pain and hanging the same way and the followers' goal
over all of them). It is essential, not optional: if each
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
From cycle 5 the shares are also the same at the ends: every segment
counts itself 40% and each side 30%, a missing side's share going to the
other side ([Cycle 5](#cycle-5-coordination-first-the-users-design)).

α applies to energy only.

## Critics and balances

| Agent | Goal critic | Trail critic (cycles 4 and 5) | Cost critic (health from cycle 5) |
| --- | --- | --- | --- |
| Head | Task; in cycle 6 less being pushed | Its own offset and segment 1's | Energy, pain, chain, and from cycle 4 being held up; from cycle 5 its own weight beyond half; from cycle 6 hanging |
| Each follower | Path; none from cycle 3; from cycle 5 coordination; in cycle 6 the distance on its trail less its squeeze; from cycle 7 the head's line erased, less its distance from it and its front joint's force | The offset of the segment behind, and its own at `trail_own_weight` | Energy, pain, chain, and from cycle 4 carrying itself (cycle 5: in coordination); from cycle 6 energy, pain and hanging |

Goal and costs count **1 : 1** (agreed), through two advantages normalised
apart ([critic.md](critic.md)); from cycle 4 the followers' trail and cost
count 1 : 1, and the head's task, trail and cost 1 : 0.25 : 1; from cycle 5
every segment's three critics count 40 / 35 / 25 (the followers'
coordination, trail and health; the head's task, trail and health); from
cycle 6 every segment's goal and health count 1 : 0.8. Energy and pain share one critic rather
than one each: pain is rare, and a critic of a signal that is almost
always zero would have its noise enlarged by the normalisation. Inside the
cost critic, pain is priced in energy: how much energy a hit of a given
force is worth.

| Balance | Status |
| --- | --- |
| Goal : costs | 1 : 1, agreed |
| α | Halving per segment, as a share of the whole, agreed |
| Pushing : braking work | About 4 : 1, from muscle data, agreed |
| Work : force, inside energy | Force dominant, as in animals of a gram, agreed: 90% of a raised walk's energy at the level-1 pace, which sets the force price at 1.98 mJ per second of one motor fully on ([As built](#as-built)) |
| Pain : energy | 1 : 1 in activation, agreed |
| Chain : energy | A unit weight on a joint as one motor fully on, agreed: in run 01's push about the followers' energy ([As built](#as-built)) |
| Progress : arrival, for the head | Progress in millimetres; arrival worth one average approach of the current level, agreed |
| Steering : own body, in the trail | 1 : 0.25 for the followers, the head 1 : 1, agreed (cycle 4) |
| Carrying oneself : energy | A unit weight of net force as a quarter of one motor fully on, agreed (cycle 4): a fifth of a walking follower's cost, and going limp dearer than walking |
| Coordination : trail : health | 40 / 35 / 25, agreed (cycle 5), for the head task : trail : health |
| Pushing ahead : other net force | 2 : 1, agreed (cycle 5) |
| The head's own weight : energy | A unit weight beyond half its weight, or on the floor, as one motor fully on, agreed (cycle 5): limp on its belly costs the head 1.2 mJ/s beside its 0.8 mJ/s walking |
| α, cycle 5 | By sides: itself 40%, each side 30%, agreed |
| Goal : health (cycle 6) | 1 : 0.8, agreed |
| Squeeze : distance (cycle 6) | 300 mm/s of goal per unit weight squeezed, agreed: measured to make pushing a slow head lose for segments 1 and 2 |
| Hanging : energy (cycle 6) | A unit weight as one motor fully on, agreed |
| α, cycle 6 | By sides, kept over equal sharing |
| Line : off the line : front joint (cycle 7) | Squared against the reference pay, 40 mm/s, at 3 mm or 0.3 unit weights; the distance counted up to 10 mm, agreed |
| The head's force price (cycle 7) | None, the user's decision |
| Team : own (cycle 14) | 1 : 0.8, as goal and health |
| Millimetres per millijoule inside a channel (cycle 14) | 10: the reference walk's energy a fifth of its line; hanging and pain beside the force share |
| Pain's ramp (cycle 14) | Half at the first update cycle, full from cycle 129, the user's |
| The line's closeness (cycle 15) | Paid in full on the line, nothing beyond 5 mm, squared between |
| The line's drift (cycle 15) | The head's sideslip, averaged over 0.4 s, no tolerance, leaking over 2 s |
| Squeeze (cycle 15) | The force share's price, 40 mm/s at 0.3 unit weights squared, the joints' mean, every follower alike |
| Energy's and the squeeze's ramp (cycle 15) | Pain's: half at the first update cycle, full from cycle 129 |

## As built

Every quantity is per segment and per 20 ms step; the simulation measures
forces and work after each of the step's physics steps (40 on v4). The
reward has two channels, goal and health
(`environment/cooperative_reward.py`), and every segment learns from both
(the environment's `segment_reward_channels`), weighed by `[agents]
advantage_weights`. Its parts are logged apart: progress, arrival, line,
off_line and front_joint in the goal, pushing, braking, force, pain and
hanging in health.

**Averaged measures.** Each step the averages move a share r = 0.02 s /
`averaging_s` of the way to the step's values (`AveragedForces`), starting
at the current values when a world restarts. With them, ḡ is the net force of a
segment's two spine joints on it, its legs included, along its forward
axis and up; f̄ the force of the segment ahead on it along the body,
positive when it pulls, negative when the segment pushes it, zero for the
head; b̄ the floor's push on its body; all in unit weights, means over the
step's physics steps before averaging.

**The head's line** (`environment/cooperative_line.py`, `HeadLine`). The
tip adds a point to the line each time it has moved 0.5 mm from the last;
at a restart the line is the body's own centreline, from the rear's tail
to the tip. A segment's tail is its rear end: the joint behind it, and for
the rear its front end one segment length back along its body. Each step
every tail is placed on the line by the nearest point, looked for within
64 pieces either way of its last place, which gives its arc length along
the line; its erase point is the furthest arc length it has reached. The
head's own copy is measured too, unpaid.

**The goal channel**, in millimetres. The head:

> goal = 1000 · (d_before − d_after) + arrival

with d the flat distance from its tip to the target; the arrival, when the
target comes under the head, is the middle distance of the level's
targets, (near + far) / 2, in millimetres (`arrival_approaches` = 1
average approach). A follower:

> goal = 1000 · Δs − R · 0.02 s · [ (min(δ, c) / δ₀)² + (f̄ / F₀)² ]

with Δs the line its tail erased on the step (how far its erase point
moved), in metres; δ its tail's distance from the line from its erase
point on, in millimetres, counted up to c = `line_distance_cap_mm` (10);
f̄ its averaged front joint force, in unit weights; R =
`reference_pay_mm_per_s` (40), δ₀ = `line_distance_scale_mm` (3) and F₀ =
`joint_force_scale` (0.3).

**The health channel**, in millijoules. Each segment's own energy, over
its leg motors and the spine motor behind it:

> e = 1000 · (W⁺ + ¼ W⁻) + p · Σ a²

with W⁺ and W⁻ the step's pushing and braking work in joules (torque times
joint speed times the timestep, by its sign), a each motor's command as
the motor receives it (a spine motor's scaled by `spine_action_scale`),
and p = 1.98 mJ/s × 0.02 s = 0.0396 mJ, the force price of one motor fully
on for a step. A segment pays its neighbourhood's energy, shared, its own
pain, and its own hanging:

> health = −Σⱼ wᵢⱼ eⱼ − p · mean(F²) / F_push² − p · h · (max(ḡ_up − u, 0) + b̄)

where w are the sharing weights by sides (`energy_sharing_weights`: itself
1 and each side 0.5 + 0.25 spread over that side's neighbours by
distance, a missing neighbour's weight to the nearest one on its side, a
missing side's to the other side, each row scaled to add up to one), F
the force with which its legs press other legs, F_push the strongest push
of a leg's sweep at its foot, 0.94 mN on v4, h = `hanging_price`, and u
the support it may take for free, `head_free_support` (0.5) for the head
and 0 for the followers.

**Sharing** (cycle 13). With `goal_team_share` s, each follower's goal
parts are (1 − s) its own and s the followers' mean, part by part; the head
keeps its own. With `health_sharing = "all"`, pain and hanging are shared
with the weights w as energy is: −Σⱼ wᵢⱼ (painⱼ + hangingⱼ). Pain is
weighed by the ramp of the update cycle, min(1, (cycle − 1) /
`pain_ramp_cycles`), which the experiment sets before each cycle
(`set_pain_weight`); evaluations weigh it 1. The logged parts are as paid,
after sharing.

**Team and own** (cycle 14). With `reward_channels = "team_own"` the
channels are the team's and each segment's own. The head's progress and
arrival, every follower's line and off-line cost, the followers' mean, and
energy, the body's mean, Σⱼ eⱼ / N for every segment, go to the team
channel; the front-joint and own force costs, each segment's own, and its
own pain and hanging to the own channel. Each channel adds its parts with a
millijoule counted as `mm_per_mj` millimetres; the parts are logged in their
own units. Pain's weight starts at `pain_ramp_start`: s + (1 − s) ·
min(1, (cycle − 1) / `pain_ramp_cycles`). The sharing settings of cycle 13
belong to the two channels by kind (`"goal_health"`, the default).

**The line near it, drifting, and the squeeze** (cycle 15). With
`line_closeness_mm` c, a follower's line part is its erased line times
max(0, 1 − (d / c)²) and the off-line part is zero. With
`line_drift_leak_s` τ, `HeadLine` keeps the head's forward axis and its
centre's step, each averaged at 0.05 a step (0.4 s); the sideslip s is the
angle from the averaged step to the averaged axis, and the drift D grows
by the averaged step's length times sin s and shrinks by 0.02 / τ of itself
each step; the tip's new points lie D to the left of the averaged step. A
restart sets both averages and the drift afresh. The squeeze part, team
channel, every follower's: −0.8 mm a step × `squeeze_weight` × the mean
over the joints k ≥ 1 of (min(0, Tₖ + Σⱼ≥ₖ tⱼ) / `joint_force_scale`)², Tₖ
the averaged front joint force of segment k and tⱼ the tows, in unit
weights. With `ramp_team_costs` energy and the squeeze are weighed by
pain's ramp. The diagnostics log the head's sideslip and the drift.

**Energy shared by neighbourhood, and pain averaged** (cycle 16). With
`energy_side_shares` s₁, s₂, … (team and own only), segment i's energy
parts are Σⱼ wᵢⱼ Eⱼ, wᵢᵢ = 1 − 2 Σ s, the neighbours at distance d on each
side s_d, a missing neighbour's share to the nearest on its side and a
missing side's to the other side (`side_share_weights`), in the own
channel. `AveragedForces` keeps a fifth average, the force with which a
segment's legs press on other legs (`leg_leg_force`), and with
`pain_averaged` pain is that average squared, in newtons, times the pain
price.

**The force price, measured** (`analysis/energy_check.py` in
`../CentipedeTraining`; on v4 as revised on 2026-10-15, with leg armature
and a 0.5 ms timestep; before it the price was 2.19 mJ/s). Scripted gaits on
v4: each sweep driven along a
sine by a feedforward command, the lifts pressing in stance and rising in
swing, a wave over five segments. The reference is a raised walk at the
level-1 pace, 2.5 strides a second; the price makes force 90% of its energy.
The 90%: a 1 g animal's speed-dependent metabolic cost (Taylor, Heglund and
Maloiy, J. Exp. Biol. 1982) and its mechanical power (Heglund et al., J.
Exp. Biol. 1982), extrapolated to a gram at 35 mm/s, leave the work at 25%
muscle efficiency about 12% of the cost; force takes the rest, in line with
Kram and Taylor (1990), who found the cost of running set by the force the
legs make. An estimate, not a measurement of centipedes.

| Gait | Speed | Height | Feet carry | Energy | Per mm walked |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rest on the belly | 0 | 1.7 mm | 3% | 0 | |
| Stand | 0 | 7.8 mm | 100% | 5.9 mJ/s | |
| Raised walk, 2 Hz | 27 mm/s | 7.8 mm | 100% | 8.4 mJ/s | 315 µJ |
| Raised walk, 2.5 Hz (reference) | 36 mm/s | 7.6 mm | 100% | 9.3 mJ/s | 257 µJ |
| Raised walk, 6 Hz | 93 mm/s | 7.6 mm | 100% | 14.7 mJ/s | 157 µJ |
| Belly drag, 2.5 Hz | 2.2 mm/s | 1.7 mm | 51% | 6.7 mJ/s | 3,025 µJ |
| Belly drag, 6 Hz | 8.1 mm/s | 1.7 mm | 54% | 12.0 mJ/s | 1,490 µJ |

The check of realism passes: walking raised costs 6 to 12 times less per
millimetre than dragging the belly, and a still centipede saves by resting
on its belly, as a real one does. Two lessons: the scripted walks' work is
mostly the joints' damping (fast leg swings), not the body's motion; and
with force dominant, a faster walk is cheaper per millimetre up to the
fastest tried, 1.7 body lengths a second, so the preferred speed lies
higher, and the task's distances, not energy, will set the pace at first.

### Earlier cycles, as built

Removed from the code in cycles 6 and 7, with their settings; kept as the
record of what runs 01 to 12 trained with. Every earlier cycle also paid a
follower progress toward where the segment ahead's centre was before the
step (until cycle 3) and shared energy by wᵢⱼ = 0.5^|i−j| over the
segments it sees (until cycle 5). Cycle 6's goal channel (runs 11 and 12),
its health being today's:

**The goal channel**, in millimetres, with q = `squeeze_price_mm_per_s` ×
0.02 s = 6 mm per unit weight. The head:

> goal = 1000 · (d_before − d_after) + arrival − q · max(ḡ_along, 0)

with d the flat distance from its tip to the target; the arrival, when the
target comes under the head, is the middle distance of the level's
targets, (near + far) / 2, in millimetres (`arrival_approaches` = 1
average approach). A follower:

> goal = 1000 · Δx · Q − q · max(−f̄, 0)

with Δx how far its front end, its spine joint, moved along its own
forward axis in the step, in metres, and Q the quality of its trail:

> Q = exp(−(max(e − `trail_dead_zone_deg`, 0) / `trail_width_deg`)²), e = √(s(θᵢ₊₁) + k · s(θᵢ))

with θᵢ segment *i*'s offset from its averaged trail, s(θ) = (180/π)² · 2
(1 − cos θ), the square of θ in degrees while it is small, and k =
`trail_own_weight`; the rear's e is √s(θᵢ). Each segment's front end, the
head's tip or its spine joint, adds a point to its trail each time it has
moved an eighth of the shortest segment length; θᵢ is the angle between
the segment's forward axis and the line from the newest point at least
one segment length (`BodyMeasures.segment_lengths_m`) behind its front end
to the front end (`environment/cooperative_trail.py`).


**The goal channel**, in millimetres. For segment *i*, with *d* its flat
distance to its target before and after the step (the head's tip to the
target; a follower's centre to where the segment ahead's centre was before
the step):

> goal = 1000 · (d_before − d_after) + arrival

The arrival is the head's alone: when the target comes under the head, the
middle distance of the level's targets, (near + far) / 2, in millimetres
(`arrival_approaches` = 1 average approach).

**The cost channel**, in millijoules. Each segment's own energy, over its
leg motors and the spine motor behind it:

> e = 1000 · (W⁺ + ¼ W⁻) + p · Σ a²

with W⁺ and W⁻ the step's pushing and braking work in joules (torque times
joint speed times the timestep, by its sign), a each motor's command, and
p = 1.98 mJ/s × 0.02 s = 0.0396 mJ, the force price of one motor fully on
for a step. A segment pays its neighbourhood's energy, shared, and its own
pain:

> cost = −Σⱼ wᵢⱼ eⱼ − p · mean(F²) / F_push²

where wᵢⱼ = 0.5^|i−j| over the segments it sees, scaled to add up to one,
F is the force with which its legs press other legs, and F_push the
strongest push of a leg's sweep at its foot, 0.94 mN on v4. With the
chain (`chain_price` > 0), each segment also pays its own two joints:

> chain = −p · c · (|f_front| + max(f_rear, 0)²)

with f_front the force of the segment ahead on it and f_rear its own pull
on the segment behind (the force on that segment's front joint), both
along the body, in unit weights, as means over the step's physics steps,
and c = `chain_price`. The rear pull's square is paid by every segment
(`chain_rear_pull = "every"`, cycle 2), by the head alone (`"head"`) or by
none (`"none"`). Without `follower_goal` (cycle 3) the followers' progress
is measured but not paid, and their critics learn from the cost channel
alone (the environment's `segment_reward_channels`).

**The trail channel** (cycle 4, `follow_trail`, which needs `follower_goal`
off), in squared degrees. With θᵢ segment *i*'s offset from its trail
(`environment/cooperative_trail.py`), and s(θ) = (180/π)² · 2 (1 − cos θ),
the square of θ in degrees while it is small:

> trail = −(s(θᵢ₊₁) + k · s(θᵢ))

with k = `trail_own_weight` for the followers and 1 for the head, and no
first term for the last segment. Each segment's front end, the head's tip
or its spine joint, adds a point to its trail each time it has moved an
eighth of the shortest segment length; θᵢ is the angle between the
segment's forward axis and the line from the newest point at least one
segment length (`BodyMeasures.segment_lengths_m`) behind its front end to
the front end. The head's critics are weighed 1 : 0.25 : 1 by
`[agents] head_advantage_weights`.

**Carrying oneself** (cycle 4, `carry_price` > 0), in the cost channel. With
g the net force of a segment's two spine joints on it, its legs included,
as a mean over the step's physics steps, in unit weights, g_along along its
forward axis and g_up up:

> carry = −p · c_carry · (|g_along| + max(g_up, 0))

with c_carry = `carry_price`, and no along term for the head. The parts are
logged apart: progress, arrival, trail, pushing, braking, force, pain,
chain, carry, coordination.

**Coordination** (cycle 5, `coordination`, which needs `follow_trail`):
four channels, goal, coordination, trail and health; the head learns from
goal, trail and health, the followers from coordination, trail and health,
weighed by `[agents] advantage_weights` and `head_advantage_weights`, one
per channel. Each step the averages move a share r = 0.02 s / `averaging_s`
of the way to the step's values (`AveragedForces`; the trail's front ends
and forward axes in `TrailMemory`), starting at the current values when a
world restarts. With ḡ the averaged net joint force (along, up), f̄ the
averaged front joint force and b̄ the averaged floor push on the body, in
unit weights, a follower's coordination is

> coordination = −(|ḡ_along| + (`push_factor` − 1) · min(max(−ḡ_along, 0), max(−f̄, 0)) + max(ḡ_up, 0) + b̄)

and the head's carry, in its health,

> carry = −p · c_carry · (max(max(ḡ_up, 0) − `head_free_support`, 0) + b̄)

The trail channel becomes

> trail = 0.02 · exp(−(max(e − `trail_dead_zone_deg`, 0) / `trail_width_deg`)²), e = √(s(θᵢ₊₁) + k · s(θᵢ))

with k = `trail_own_weight` for every segment but the rear, whose e is
√s(θᵢ), and the offsets θ from the averaged trail. Energy is shared by
`energy_sharing_weights(..., sides=True)` with `energy_sharing = "sides"`.
Each observation block holds the front joint force and the net joint
force, averaged ([observations.md](observations.md)).

**The chain's price, measured** (`analysis/chain_check.py` in
`../CentipedeTraining`, 2026-10-15): the force on each joint along the
body, in unit weights, means over each step, on the CPU.

| | Run 01 at cycle 64 (followers pushing) | Scripted raised walk (each segment carries itself) |
| --- | ---: | ---: |
| Force on the head's joint | −0.66 (pushed) | +0.02 |
| Strongest mean force | −0.77 (pushed, segments 2-3) | +0.26 (pulled, segments 4-5) |
| Mean size over the joints | 0.80 | 0.19 |
| Energy per segment | 1.66 mJ/s | 0.78 mJ/s |
| Front joints' chain at c = 1 | 1.58 mJ/s, about the energy | 0.38 mJ/s |

Setting run 01's followers' chain equal to their energy gives c = 1.05, so
the force price itself was kept: pushing as run 01 did doubles a
follower's cost. The scripted walk's chain is an upper bound, since its
segments' efforts are not evened out. In run 01 the push grows from the
tail toward the front: segments 4 to 11 add it, a tenth to a fifth of a
unit weight each, and segment 1, which adds none, carries 0.66 of it. A
risk to watch: before the followers walk, a head that walks drags them,
and the square of a pull of several unit weights could teach it to stand
still.

## Left out

- Step-quality terms (foot hold, carry, grip, the step value): they
  prescribe a gait, and runs 8 to 57 disproved them.
- Rhythm and tempo terms.
- Rewards for sameness or for beating the others.
- A separate cost of time: paying for progress already makes standing still
  worse.
