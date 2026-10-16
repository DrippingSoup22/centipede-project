# The tow

Training wheels that fade (cycle 10, the user's choice on 2026-10-16, from
the plan two sessions worked out with the user). In training every segment
is pushed gently forward; the push shrinks as the agents' mean action
learns to walk, and every evaluation runs without it.

## Why

Through cycles 7 to 9 the body walked only under exploration noise, and
its mean action stood ([actions.md](actions.md#how-actions-explore)). The
cause is the body itself. The floor's friction is 0.8 on feet and belly
alike, and the chain does not stretch, so a segment whose feet are planted
holds about 0.8 of its weight back, and one that steps pushes at most about
as much. A standing body moves only when most of its segments step at once.
A train car that does nothing rolls on its wheels; a segment that does
nothing is a parked car with its brake on.

Measured on the scripted wave walk (`analysis/tow_check.py` in
`../Training`, v4, CPU): the smallest number of stepping segments that
moves the body faster than 5 mm/s, the steppers spread along the body or
gathered at the rear, under a forward push of t times each segment's
weight:

| Tow t | 0 | 0.3 | 0.5 | 0.7 |
| --- | ---: | ---: | ---: | ---: |
| Steppers spread along the body | 8 | 3 | 1 | 1 |
| Steppers at the rear | 12 | 12 | 2 | 1 |

So one segment's own step changes nothing it can feel while its
neighbours stand, and its goal cannot reward the step. Measured on run 16's
checkpoints (`analysis/own_step_check.py` in `../Training`): the change in
one segment's own goal over 3 s when its noise is switched on, every other
segment's noise as it was, from the same states (40 pairs per case, mm,
with the mean's standard error):

| Checkpoint, tow | Segment 1 | Segment 5 | Segment 10 |
| --- | ---: | ---: | ---: |
| Cycle 32, no tow | −4.5 ± 4.9 | −3.4 ± 2.0 | −1.4 ± 2.4 |
| Cycle 32, tow 0.4 | +45 ± 15 | +141 ± 32 | +18 ± 7 |
| Cycle 128, no tow | −1.2 ± 8.3 | +5.9 ± 5.6 | −2.0 ± 1.9 |
| Cycle 128, tow 0.4 | +137 ± 24 | +105 ± 22 | +29 ± 21 |

Without a tow a segment's exploring earns it nothing, even in the body
that walks under noise at cycle 128; its health always pays for it (−0.14
to −2.23 mJ). With a tow of 0.4 the same exploring pays clearly. Under the
tow the followers' noisy body erased 22 to 24 mm/s of line, against 0.8 and
4.2 without.

## The design

- **The push.** Each segment is pushed by its world's tow level times its
  own weight, its body and legs (v4: the head 1.09 unit weights, every
  other segment 0.99), along its forward axis laid flat on the floor, at
  its centre of mass, with no torque. Proportional to the weight, it acts
  as a slope would, the same share of every segment's friction, without a
  slope's loss of load. It is set before each 20 ms step from the
  segment's heading and held through the step.
- **Below the friction.** A tow under 0.8 cannot drag a still body by
  itself: a body that holds its feet creeps 0.94 mm/s at 0.4 and 1.59 at
  0.7, a limp one 0.18 at 0.4 (`tow_check.py`); at 0.8 the body slides.
  Right after a restart, while the body settles onto the floor for about
  0.2 s, a tow of 0.4 slides it about 2 mm.
- **Every world draws its own.** At each episode start a training world
  draws its level uniformly from 0 to the tow's top, so the agents always
  train at small tows too, as Yu, Turk and Liu's lessons overlap. Each
  segment observes its world's level (`observe_tow`).
- **Mean-action worlds measure.** The last worlds (128 and 128 in the
  pilot) act with the agents' mean action and are neither stored nor
  learned from: the first half always at the tow's top, the second never
  towed, the honest measure. The target level ignores them.
- **The fade.** After each update cycle, once `tow_window_steps` steps
  (256 by default; 128 in the pilot, the user's choice) have passed since
  the top last moved, the top falls by 0.02 if the towed mean-action
  worlds walked, else it rises by 0.02, between 0 and its start of 0.4.
  Walked means both: their followers erased more than 5 mm/s of line each,
  far above the creep, and their feet lifted off more than once a second
  after stances of at least 60 ms. A rolling window would lag behind the
  top's own moves and swing it from end to end, so each window starts
  fresh when the top moves.
- **The price.** A tow helps every segment along, so the follower's force
  price must not let it hide a drag: it charges the own ground force, the
  joints' net force plus the tow
  ([reward.md](reward.md#cycle-10-each-segment-pays-its-own-ground-force-the-user-agreed)).
- **The reflex inputs.** Each block adds its front joint's force and its
  net joint force of the step itself beside the 1 s averages
  (`observe_instant_joint_forces`), so that a segment feels at once that it
  is being pulled ([observations.md](observations.md)).
- **Evaluations never tow.** Reviews and evaluations measure the mean
  action without tow, as before.

## Reading a run

- **The spread rising, then the untowed mean-action worlds stepping and
  erasing line:** the gait moves into the mean, and the tow works.
- **The spread rising, but the untowed mean-action worlds still standing
  by cycle 128:** the noise is the gait again, now under the tow. The next
  lever is then how the mean averages a rhythm away, not credit.
- **The tow's top hovering above 0:** the towed mean walks there and not
  below. A top stuck while the towed worlds slide without stepping would
  call for a measure of slipping feet, which the branch does not have yet.

## Risks

- **A gait made for the tow.** Under a tow a segment that carries itself
  brakes by the tow's share (its own ground force is zero when its feet
  hold back exactly the push), and a learned walk at t > 0 is a braking
  walk; it must turn into a pushing one as the tow fades. The overlapping
  levels down to 0 are meant to keep that change small.
- **Sliding instead of stepping.** Under noise the body moves without
  steps: with a spread of 0.25 on holding commands it moves 17 mm/s at a
  tow of 0.3 and 37 at 0.5 (0.5 without tow), its random lifts unloading
  the feet. The own ground force charges sliding feet, and the fade asks
  for lift-offs, but a tapping gait could count as stepping.
- **Steering.** The tow pushes the head along its heading: closing on a
  target ahead comes free, which pays turning toward it, and the heading
  error (70° to 117° in runs 15 to 18) is read in every review.

## In the code

| Part | Where | Settings |
| --- | --- | --- |
| The tow | `environment/cooperative_tow.py`: each world's level, the forces, the measuring worlds' sums and the fade; started and reserved by the experiment in training only | `[environment.cooperators]` `tow_start`, `tow_step`, `tow_window_steps`, `tow_walk_mm_per_s`, `tow_lift_off_hz`, `observe_tow` |
| The push | `environment/simulation`: `step`'s `body_forces`, MuJoCo's applied force on each segment's body, CPU and GPU | |
| Mean-action worlds | `agents/agents.py` acts and stores; `interaction_loop` counts only the learning worlds; `experiment/` reserves them, moves the tow, logs the `tow` category and keeps the top in checkpoints | `[run] mean_action_worlds_per_group` |
| Each segment's own weight | `environment/simulation/body_measures.py`: `segment_weights_n` | |

Every setting defaults to no tow and no mean-action worlds, so earlier
configurations keep their meaning.

## Sources

- Yu, Turk and Liu, "Learning Symmetric and Low-energy Locomotion", ACM
  Transactions on Graphics 37(4), SIGGRAPH 2018 (arXiv 1801.08093): a
  curriculum of physical assistance toward balance and forward movement,
  relaxed until the character moves without help.
- Cao, Zhang, Nie, Lin, Li and Gao, "Learning Motion Skills with Adaptive
  Assistive Curriculum Force in Humanoid Robots", arXiv 2506.23125 (2025,
  preprint): an agent that applies guiding forces and lowers them as the
  robot improves.
- Rudin, Hoeller, Reist and Hutter, "Learning to Walk in Minutes Using
  Massively Parallel Deep Reinforcement Learning", CoRL 2021: a curriculum
  moved by each robot's own success.
