# Task

**Status (2026-10-14): agreed with the user and built
([As built](#as-built)).** The
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

## The targets (agreed, 2026-10-14)

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

## The starting heading (agreed, 2026-10-14)

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

## How a world runs

```text
episode starts (fresh pose, random heading) ─┬─ target 1 appears, placed from the head; a 7 s stopwatch starts
                                             │    reached within 7 s → arrival bonus; counted a success for the level
                                             │    7 s pass           → counted a miss for the level; the target stays
                                             │    reached later      → arrival bonus all the same
                                             ├─ the moment a target is reached, the next appears, placed from the head
episode ends at 20 s ────────────────────────┴─ the world restarts from a fresh pose
after every update, one level for all worlds moves toward 1 target in 10 reached within 7 s
```

- **A target stays until it is reached** (the user, 2026-10-14). Food does
  not vanish, and a centipede that overshoots or wanders off must turn
  back, which is practice at recovering.
- **The 7 s only measure.** They tell the level controller whether a
  target was reached in time; the reward never depends on them, so the
  head needs no clock among its inputs, and the time limit stays outside
  the task, where bootstrapping handles it (Pardo et al., 2018).
- **A late arrival is worth less all the same, smoothly.** The discount
  (0.998 at 50 decisions a second, a half-life of 7 s) makes an arrival
  count less the later it comes, seen from when its target appeared, with
  no cliff at 7 s:

  | Arrival after | 2 s | 3 s | 7 s | 8 s | 14 s |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Its worth | 82% | 74% | 50% | 45% | 25% |

  The progress on the way is discounted alike, so a faster walk is worth
  more at every moment. Scaling the bonus by the time taken as well would
  pay speed twice, and the head would need a clock to see why.

  The bonus itself is paid in full whenever it comes; the percentages are
  how much the centipede values it in advance. A late arrival can never
  teach it to give up: the discount ranks later below sooner, but never
  below never, which is worth nothing. And it always values from the
  present: at second 6, a head one second from its target sees the
  arrival at about 90%.
- **The episode's 20 s are simulated time**, not waiting time. A training
  cycle simulates the same number of steps in every world (the rollout
  window), whatever the episode's length; the length only decides when a
  world restarts. A shorter episode does not make training faster or
  slower.

## Walking on (agreed, 2026-10-14)

**An arrival never ends the episode** (the user): a new target appears at
once, by the current level's rules, **placed from the head where it
stands**. The alternative, every target around the episode's starting
point, keeps the walk in one area, easier to watch (the user's lean), but
the agents never see that origin: they see only the target relative to
the head. From the origin, the next target would lie anywhere from almost
under the head to twice the range away, in any direction, so the 1.5 ratio,
the opening of the circle and the level would stop controlling the task,
and near targets would come back. Placed from the head, every target obeys
the level exactly. Watching is the viewer's job: a camera that follows the
head, or the trail drawn on the floor. The floor is the same everywhere,
and a walk of metres loses nothing in precision.

**A stopwatch per target.** The level needs a share of targets reached in
time, so each target's time is measured from its appearance: reached
within 7 s is a success, otherwise a miss, counted when the 7 s pass.
The target stays either way (above). A target whose 7 s cannot pass before
its world restarts is not counted at all, reached or not: a restart cuts
its stopwatch, and counting only the ones reached before the cut would
leave out the hard cases and read too high (as built, 2026-10-14). A fixed time is the user's earlier
idea (a fixed length asks a farther target for a faster walk), moved from
the episode to each target; it acts through the level, which rises only
while one target in ten is reached within it. The oscillators' circle task
allowed 10 s per target, so its level read as speed.

## The episode (agreed, 2026-10-14)

**Fixed, generous, and never growing with the level.** With walking on,
the task has no end: the centipede is judged over an indefinite time, and
the episode's time limit only restarts a world from a fresh pose. In that
case the learning must bootstrap at the time limit, valuing the state it
was cut in, rather than treat it as an end (Pardo et al., ICML 2018), as
main already does. Then the episode's length is not part of the task: it
changes nothing in what is best to do, only how often each world starts
afresh. Growing it with the level would change nothing the levels do not
already change; the difficulty is the targets' job.

What the length trades:

- **Longer:** more practice at walking on and at recovering from bad
  positions, which the policy must learn anyway, and fewer restarts.
- **Shorter:** more fresh starts, and a world stuck early in training (a
  body tangled or sunk) waits less before its reset (the user's concern).

Since nothing else ends an episode (resting on the belly is allowed, and
the body never turned over), the time limit is the only reset. A length of
an episode of a few targets' 7 s keeps both sides: several targets per episode,
and no world stuck for long. Rudin et al. (CoRL 2021), training thousands
of quadrupeds in parallel, used 20 s episodes, bootstrapped at the time
limit and reset only there, so one episode spans many updates; with too
short a horizon per robot, their performance dropped sharply.

**The discount follows the target, not the episode.** Main ties it to the
episode's length ([agents.md](../agents.md#settings)); with walking on and
bootstrapping, the time that matters is a target's: the discount should
see about 7 s ahead, one stopwatch.

## The numbers (agreed, 2026-10-14)

Times are in seconds, so they hold whatever the decision rate; in steps of
20 ms they are 50 per second, as in legged_gym.

| Setting | Value | Why |
| --- | --- | --- |
| Targets, level 0 | 20 to 30 mm, within ±45° | [The targets](#the-targets-agreed-2026-10-14) |
| Targets, level 1 | 2.7 to 4 body lengths, all around | The same; the full circle from level 0.55 |
| Starting heading | Any direction | [The starting heading](#the-starting-heading-agreed-2026-10-14) |
| Stopwatch per target, for the level | **7 s** (350 steps) | At level 1, the middle target, 3.3 body lengths, plus about one body length of turning, in 7 s asks for 0.64 body lengths per second: the slower speed measured for *S. polymorpha* on flat ground (0.64 to 1.44; Diaz et al., Georgia Tech). Level 1 means walking like a real centipede at an easy pace. At level 0 the same 7 s is very generous. The target stays after the 7 s; only the count changes |
| Episode | **20 s** (1,000 steps), about three targets' 7 s | The user's choice: about three far targets at a good walk, as legged_gym's 20 s; a world stuck early waits at most 20 s |
| First episode of each world | A random length up to 20 s | Otherwise, with no other end to an episode, all worlds would restart on the same step; legged_gym starts every world at a random point of its episode for the same reason |
| Discount | **0.998**, half-life 7 s | An arrival 7 s after its target appeared is still worth half, seen from the target's appearance (main's rule, tied now to the target instead of the episode). legged_gym's 0.99 sees about 1.4 s ahead: right for its velocity task, but a 7 s arrival would be worth 3% |
| Level | One for all worlds; after every update, level += 0.02 × (arrival share − 0.10), kept between 0 and 1 | The user's rule. The arrival share counts, over the last episode length, the targets whose 7 s ended: reached within them, or not. At 50% arrivals the level climbs 0.008 per update, 0 to 1 in about 125 updates; the oscillators' step was the same 0.02 |
| Range circle | None | Walking on makes it needless: a head that wanders off still has its target, and the episode's 20 s bound the time it can lose |

For the 21-segment model the distances grow in body lengths and the
times stay, since the speed asked is in body lengths per second too.

## The levels (agreed, 2026-10-14)

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

**An option from the research: a level per world.** Rudin et al. gave each
robot its own level: up when it crossed its terrain, down when it covered
less than half of what was asked, and robots at the top sent back to a
random level, for variety and against forgetting. The population then
spreads across levels by itself, each world at its own edge. A single
level, as above, is simpler to read; this stays an option.

**The stages blur, and that is fine:** early is level 0 until arrivals
reach 10%, the walk being learned from progress; middle is the level
rising; late is the level at 1 or no longer rising. They are measured,
not set.

## As built

- **The full circle from the model.** The near end must clear the body's
  length plus its turning circle, both read from the model: on v4, 54.75 mm
  and 32.9 mm (units 4.3 mm apart, each spine joint bent to 15°), so 87.7 mm,
  reached at level 0.53. The estimate above said 0.55.
- **The level** moves after every update cycle, by the stopwatches of the
  last episode length of steps (1,000), and not before one episode length
  is recorded, so a lucky cycle cannot move it. It is logged as `task`,
  kept in each checkpoint, and a continued run or an evaluation starts from
  it.
- **Staggered first episodes**: after a full reset each world starts at a
  random step of its episode. Evaluations turn it off
  (`stagger_first_episodes = false`), to judge whole episodes.
- **What the logs call an episode** is the 20 s between two restarts (the
  user's view: an episode is a time, not a target): the targets reached in
  it and within their stopwatch, how far the head walked, and what it
  spent (`environment/cooperative_diagnostics.py`); the stopwatch's share
  is the `task` log's.
- The settings are `[environment.cooperators]`, with the values above as
  defaults; the episode is main's `max_episode_steps`, 1,000.

## A point to weigh (answered in the reward)

How progress is counted matters for the same question. Main counts it in
halvings of the distance, which makes a millimetre near a far target worth
less than one near a close target, while a millimetre's effort costs the
same everywhere. The reward now proposes millimetres, so a farther target
pays more progress in proportion to its walk, and a fixed arrival
([reward.md](reward.md#task-the-head-steers)).
