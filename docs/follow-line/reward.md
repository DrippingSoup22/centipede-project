# Reward

From run 08 (the user's decision, 2026-10-21, after run 07): the team is
paid the line it walks, up to a pace of 100 mm/s, and the joints'
accelerations take a share of that pay; each unit's own account is charged
the weight its feet do not carry. Runs 06 and 07 subtracted the
accelerations as a price instead, and run 05 charged the joints' speeds;
runs 01 to 04 used the cooperators' run 28 reward (`docs/cooperators/reward.md`
on that branch). Every run's lessons are in the study's
`Training/results/follow-line/README.md`.

## Two accounts

Each 20 ms step every agent receives:

```text
team = forward line × 1 / (1 + J / h) − steps back   (the same for all ten)
own  = − carry                                         (each unit's own)
```

Each account has its own critic ([critic.md](critic.md)). Everything is
counted in millimetres of line.

## The line

For each unit, with Δs how far its centre went along the line on the step
(mm), d its centre's distance from the line (mm) and θ its body's angle from
the line's direction:

> forward line = Δs · c · max(0, 1 − (d / b)²) · max(0, cos θ) for Δs ≥ 0,
> and a step back, Δs < 0, charged in full

The team receives the mean over the units.

- **The cap, 100 mm/s**: c is 1 while the unit's speed along the line,
  averaged over 1 s, is at most 100 mm/s, and 100 mm/s over that speed
  above it, so a faster walk earns no more. The average starts each
  episode at the cap, and a stride's surges and dips cancel in it. 100
  mm/s is 1.5 body lengths a second, the fastest pace measured in
  *Scolopendra heros* (Anderson, Shultz & Jayne 1995).
- **The band b**, the sight radius, 16.5 mm on v5: a unit is paid for the
  line only while it can see it ([observations.md](observations.md)).
- **Facing** along the line, and **a step back charged in full**, so that a
  body walking sideways earns less and rocking never pays.
- **The mean**, because the chain moves together: sharing the gain is what
  first moved the walk into the policy (cooperators, run 23).

## The accelerations

J is how much the joints change their speed: for each unit, each of its
joints' (its legs' and the spine joint behind it) change of speed over the
step, squared and added up, averaged over the last second, and the team's
J the units' mean. The team keeps 1 / (1 + J / h) of its forward line, h
being 3.8·10⁶ (rad/s²)²:

| J | 0 | h / 3 | h | 3h | 9h |
| --- | --- | --- | --- | --- | --- |
| Share of the forward line kept | 100% | 75% | 50% | 25% | 10% |

- **A share, never more.** Run 07 subtracted the accelerations as a price.
  Once that price passed what the line paid, about 0.015 of its full
  weight, every walk the agents knew paid less than standing still, which
  costs no accelerations; the agents learnt it and stopped walking. A share
  cannot outgrow the pay: standing keeps nothing, any forward walk keeps
  something, whatever the gait and the noise.
- **Free while there is no walk.** Before the body moves forward there is
  no pay to take a share of, so the untrained agents' jerky exploring costs
  nothing; the share grows with the walk, with no ramp and no schedule.
- **The exploring noise.** Its jerk is there at every speed. Subtracted,
  it was a toll that grew with the price until it swamped the line, and the
  team critic could no longer predict the return (run 07: its explained
  variance fell from 0.9 to about 0 once the cost passed the pay). As a
  share it lowers what the team keeps, but never by more than the line.
- **Wide, slow strokes keep more.** At the same pace the line is the same,
  so the smoother gait keeps the larger share. A joint's acceleration grows
  with a stroke's width but with its frequency squared, so a stroke twice
  as wide and half as fast is four times smoother for the same speed. The
  metabolic cost of swinging a leg rises about as its frequency's fourth
  power (Doke, Donelan & Kuo 2005), and legged_gym charges joint
  accelerations for smooth gaits (Rudin et al. 2021).
- **Averaged over a second**, so that the share follows how much the joints
  jerk, not one step: at each step alone a single snap could take at most
  that step's pay, and a snapping gait would come cheap.
- **A step back in full**, not discounted: a jerky step back would
  otherwise cost less than a smooth one.
- **h** keeps 76% of the line on the ledger's scripted walk and about half
  on run 07's best learned gaits (cycle 256, 120 to 131 mm/s, noise-free),
  which slowed to the cap would keep 56 to 71%.

The DeepMind Control Suite builds its rewards the same way: the humanoid's
task terms are multiplied by a control factor between 0.8 and 1
(`small_control`, Tassa et al. 2018). Here the factor can take more than a
fifth, but never all.

## Why the cap sets the pace

If a gait's J grows as its speed to the power n, a cost of J, subtracted or
as a share, is best paid at the speed where the line keeps (n − 1) / n of
itself. The pace a cost of J sets therefore depends on how fast the jerk of
the agents' gait grows with speed, which nothing controls: the ledger's
scripted walk, whose lift snaps once a step, grows as about the speed to
the 1.5, and only a share of two thirds held its pace, by 1 to 2 mm/s
(2026-10-21). So the task sets the pace with the cap, and the accelerations
judge only how smoothly the body reaches it. Runs 05 to 07 had no cap and
their pace ran with their gaits: past 350 mm/s in run 05, to a creep in
run 06, and from 150 mm/s to a creep in run 07.

## The carry

Each unit's own account is charged, at 50 mm a second for each unit weight,
what its two feet, on average over 1 s, do not carry of its weight beyond
its free support (on the headless tester, all of its weight). Run 05's
agents walked on a few units' legs: the others held theirs crossed over
their neighbours' and lifted, and saved their share of the cost while
others carried them. A unit must carry itself on its own feet whatever
else holds it up: its neighbours through the joints, its belly, or a leg
resting on another. It is its own duty, in its own account ("shared gains,
own duties", [README.md](README.md)): the charge falls on the unit that
leans on the others, not on the team. Holding a leg up still saves the
team a little of the accelerations' share (about 1 mm/s); the carry is
what makes it lose. Feet that carry while skidding along pass: a unit
carried along on skidding feet is a known gap, left open until a run shows
it.

## The ledger

Before a run, scripted walks on the built code check that the good wave walk
pays more than every fault we know (the study's `Training/analysis/ledger.py`).
On run 08's reward (2026-10-21) the good walk, 103 mm/s, keeps 74 of its
98 mm/s, and every fault loses: twice the step rate by 9 mm/s, 1.4 times
by 10, 0.7 times by 20, the wave running backward by 3, legs in random
phase by 43, a leg held up by 14 to 28, six by 69, standing by 66 to 73.
Without the cap the faster walks won (twice the step rate by 17 mm/s).
Fresh networks under the tow keep about half of the little line they walk
(+3.7 of 7.6 mm/s), where run 06's price at full weight cost them about
400 times what they earned.
