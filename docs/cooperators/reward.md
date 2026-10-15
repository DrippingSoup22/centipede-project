# Reward

**Status (2026-10-15): agreed with the user and built; the force price
measured on the model ([As built](#as-built)). The chain was added after
runs 01 and 02 ([The chain](#the-chain-push-and-pull-between-neighbours-the-users-design)).
After run 04, cycle 3 takes the followers' goal away: they follow, paying
the chain at ten times the force price
([Cycle 3](#cycle-3-followers-that-follow-the-users-design)).**

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
| Head | Task | Energy, pain and chain |
| Each follower | Path; none from cycle 3 | Energy, pain and chain |

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
| Pushing : braking work | About 4 : 1, from muscle data, agreed |
| Work : force, inside energy | Force dominant, as in animals of a gram, agreed: 90% of a raised walk's energy at the level-1 pace, which sets the force price at 1.98 mJ per second of one motor fully on ([As built](#as-built)) |
| Pain : energy | 1 : 1 in activation, agreed |
| Chain : energy | A unit weight on a joint as one motor fully on, agreed: in run 01's push about the followers' energy ([As built](#as-built)) |
| Progress : arrival, for the head | Progress in millimetres; arrival worth one average approach of the current level, agreed |

## As built

Every quantity is per segment and per 20 ms step; the simulation measures
forces and work after each of the step's physics steps (40 on v4).

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
alone (the environment's `segment_reward_channels`). The parts are logged
apart: progress, arrival, pushing, braking, force, pain, chain.

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
