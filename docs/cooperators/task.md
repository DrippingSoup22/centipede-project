# Task

**Status (2026-10-14): being designed with the user.** The
task is the environment's side of the problem: the targets, the episodes,
how the difficulty grows, and what happens after an arrival.

## The starting point

Main's task ([environment.md](../environment.md#episodes)), the reference,
not adopted by default: a target on the ground in front of the head, an
episode that ends on arrival, at the time limit, or when the head leaves
the range circle. The oscillators added a curriculum of growing distances
(the circle task), walking on to a new target after an arrival, and a
plateau stop.

## The user's ideas (2026-10-14)

- **Speed comes from the task.** No commanded speed: the body must find
  the cheapest way of moving that still reaches the goal, and as the levels
  raise the distance, reaching it asks for more travel and more speed.
- **An arrival worth more the farther it was.** Effort grows with the
  distance walked, so with a fixed arrival reward, targets beyond some
  distance would not be worth their energy. Scaling the arrival with the
  distance travelled keeps every arrival worth reaching.
- **A fixed episode length.** As targets move farther, the same time asks
  for a faster walk; a slow one no longer arrives.
- **Walking on after an arrival.** The episode does not stop at a target:
  a new one follows, so the more targets reached, the more reward, which
  also pays speed.

## The targets (proposed, 2026-10-14)

| | Level 0 | Level 0.55 | Level 1 |
| --- | --- | --- | --- |
| Distance from the head's tip | 20 to 30 mm | 89 to 134 mm | 146 to 219 mm (2.7 to 4 body lengths) |
| Direction | Within 45° of straight ahead | All around | All around |

**The far end is always 1.5 times the near end** (the user: half the
spread of a doubling). Every target of a level then asks for about the
same walk, within ±20% of the middle, so the effort per target and the
arrival share stay steadier while the task still varies. Both ends grow in
a straight line with the level, so the ratio holds at every level.

- **The start, 20 to 30 mm** (the user). 10 mm would be less than one and
  a half head lengths (the head's outline is 7.2 mm long and 8 mm wide): a
  single lurch would reach it, and near targets teach shuffling, not
  walking (the oscillators, 2026-10-13). From 20 mm the head needs several
  strides. The start is set by the head and the stride, which the
  21-segment model shares, so it stays in millimetres.
- **The end, 4 body lengths at most.** Far enough that only a sustained,
  coordinated walk arrives. Stated in body lengths, since the whole body
  must walk it, so it grows with the 21-segment model.
- **A range whose near end grows too.** The oscillators left a range's
  near end at 30 mm, so a third of the targets lay close enough to arrive
  without a wave, which led them to the circle task (one distance per
  level). Here the near end grows with the far one, so no level keeps easy
  targets.

**The direction opens to the full circle before the distance is at its
largest** (the user: at level 0.5 to 0.7). Turning toward a target behind
is a skill of its own, and it tests the body's following hardest: every
follower must trace the head's U-turn. The level where the circle closes
follows from the body's geometry. With every spine joint at its 15° limit
the body turns on a circle about 33 mm across, so a target straight
behind lies clear of the body only beyond its length plus that circle,
about 88 mm (1.6 body lengths); closer, it lies among the body's own legs.
The near end passes 88 mm at level 0.55, so the direction grows from ±45°
at level 0 to the full circle at level 0.55, and from there every target
behind can be reached by turning around. Stated as that rule (the full
circle once the near end clears one body length plus a turning circle),
it carries over to the 21-segment model.

## The starting heading (proposed, 2026-10-14)

**Any direction, from the start, not growing with the level.** The floor
is the same everywhere and every target is placed relative to the head, so
the direction the body faces when an episode starts changes nothing in the
task: it is not a difficulty. It matters only if a segment observes its
heading in the world, through its orientation, which main's observations
do: with one starting heading, the networks see only the few world
headings near it, and a body that later turns far gives them inputs they
never trained on ([environment.md](../environment.md#episodes)).
Randomising it fully makes the heading uninformative everywhere, so the
networks learn to ignore it. If the observations leave out the world
heading altogether ([observations.md](observations.md)), the starting
heading stops mattering, and randomising it still costs nothing.

## The levels (proposed, 2026-10-14)

**The user's rule:** the level starts rising once the arrivals pass 10%,
and rises faster the more they exceed it, episode after episode; the level
bonus is no longer needed.

It is a controller with a set point: the level moves by a step
proportional to (arrival share − 10%), up when above, down when below, so
the task always sits where the centipede reaches about one target in ten.
The oscillators' rule was the same with a set point of 50%, too slow to
leave level 0 (runs 49 and 50: no arrival until cycle 170), hence their
bonus. Two additions: the level must also come down below 10%, or a bad
stretch would leave the task out of reach; and the arrival share is read
over the last episode length, so one lucky cycle does not move it.

Why 10% works here: training goes best on goals of intermediate
difficulty, those reached between 10% and 90% of the time (Florensa et
al., ICML 2018, robust to the exact bounds). Their lower bound exists
because with a sparse reward, rarer successes leave nothing to learn from;
our progress is paid every step, so the walk is learned even from targets
missed, and the low end of the band costs little. It keeps the centipede
on the hardest targets it can still reach. The price: arrivals are rare, so
the arrival part of the goal is learned from few events, which thousands
of worlds make up for.

**The stages blur, and that is fine:** early is level 0 until arrivals
reach 10%, the walk being learned from progress; middle is the level
rising; late is the level at 1 or no longer rising. They are measured,
not set.

## A point to weigh (answered in the reward)

How progress is counted matters for the same question. Main counts it in
halvings of the distance, which makes a millimetre near a far target worth
less than one near a close target, while a millimetre's effort costs the
same everywhere. The reward now proposes millimetres, so a farther target
pays more progress in proportion to its walk, and a fixed arrival
([reward.md](reward.md#task-the-head-steers)).
