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

- **Where a target appears:** within 45° to either side of straight ahead
  (the user: a 90° spawning angle), at a distance drawn from a range, so
  the task varies (the user).
- **How far: one to two body lengths**, 50 to 100 mm for the twelve-unit
  model (54.8 mm). The user's 5 to 10 is read in centimetres: in
  millimetres a target would lie within one or two strides, partly under
  the head already (the head's outline is 7.2 mm long and 8 mm wide), and
  near targets teach shuffling, not walking (the oscillators, 2026-10-13).
  At 40 to 80 mm within ±45°, walking straight reaches only about 9% by
  luck ([environment.md](../environment.md#episodes)). Stated in body
  lengths, the range carries over to the 21-segment model.
- **To discuss next:** how the range grows with the levels, the episode's
  length, walking on after an arrival, and the starting heading (the
  user). The oscillators' curriculum began at 15 to 30 mm within ±45° and
  grew to 30 to 200 mm all around; their circle task then put every target
  of a level at one distance, because with a range a third of the targets
  lay close enough to arrive without a wave.

## A point to weigh (answered in the reward)

How progress is counted matters for the same question. Main counts it in
halvings of the distance, which makes a millimetre near a far target worth
less than one near a close target, while a millimetre's effort costs the
same everywhere. The reward now proposes millimetres, so a farther target
pays more progress in proportion to its walk, and a fixed arrival
([reward.md](reward.md#task-the-head-steers)).
