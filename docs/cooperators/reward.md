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
([Cycle 6](#cycle-6-distance-on-the-trail-the-squeeze-its-price-the-users-design)).**

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
From cycle 5 the shares are also the same at the ends: every segment
counts itself 40% and each side 30%, a missing side's share going to the
other side ([Cycle 5](#cycle-5-coordination-first-the-users-design)).

α applies to energy only.

## Critics and balances

| Agent | Goal critic | Trail critic (cycles 4 and 5) | Cost critic (health from cycle 5) |
| --- | --- | --- | --- |
| Head | Task; from cycle 6 less being pushed | Its own offset and segment 1's | Energy, pain, chain, and from cycle 4 being held up; from cycle 5 its own weight beyond half; from cycle 6 hanging |
| Each follower | Path; none from cycle 3; from cycle 5 coordination; from cycle 6 the distance on its trail less its squeeze | The offset of the segment behind, and its own at `trail_own_weight` | Energy, pain, chain, and from cycle 4 carrying itself (cycle 5: in coordination); from cycle 6 energy, pain and hanging |

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

## As built

Every quantity is per segment and per 20 ms step; the simulation measures
forces and work after each of the step's physics steps (40 on v4). From
cycle 6 the reward has two channels, goal and health
(`environment/cooperative_reward.py`), and every segment learns from both
(the environment's `segment_reward_channels`), weighed by `[agents]
advantage_weights`. Its parts are logged apart: progress, arrival,
distance and squeeze in the goal, pushing, braking, force, pain and
hanging in health.

**Averaged measures.** Each step the averages move a share r = 0.02 s /
`averaging_s` of the way to the step's values (`AveragedForces`; the
trail's front ends and forward axes in `TrailMemory`), starting at the
current values when a world restarts. With them, ḡ is the net force of a
segment's two spine joints on it, its legs included, along its forward
axis and up; f̄ the force of the segment ahead on it along the body,
positive when it pulls, negative when the segment pushes it, zero for the
head; b̄ the floor's push on its body; all in unit weights, means over the
step's physics steps before averaging.

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

Removed from the code in cycle 6, with their settings; kept as the record
of what runs 01 to 10 trained with. Every earlier cycle also paid a
follower progress toward where the segment ahead's centre was before the
step (until cycle 3) and shared energy by wᵢⱼ = 0.5^|i−j| over the
segments it sees (until cycle 5).

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
