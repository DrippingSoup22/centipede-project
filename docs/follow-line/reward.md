# Reward

Agreed with the user on 2026-10-20: the cooperators' run 28 reward
(`docs/cooperators/reward.md` on that branch, cycles 14 to 18, where every
term's history and measurements are), with nothing left of the head's
special role. Changed after run 01 and run 03 (the user's decisions,
2026-10-20): the line is paid against a target speed on each unit's speed
averaged over 1 s, pain is removed, and energy and the squeeze start at a
quarter of their weight and reach it in 160 cycles.

## Two accounts

Each 20 ms step every agent receives two numbers:

```text
team = line − squeeze − energy                (the same for all ten)
own  = − force share − hanging              (each unit's own)
```

Both are counted in millimetres of line, a millijoule worth 1.59 mm. Each
has its own critic, and learning weighs them 1 : 0.8 ([critic.md](critic.md)).

## The team account

**The line.** For each unit, with Δs how far its centre went along the line
on the step (mm), v̄ its speed along the line averaged over 1 s as the forces
are (mm/s), d its centre's distance from the line (mm) and θ its body's
angle from the line's direction:

> line = Δs · s(v̄) · max(0, 1 − (d / b)²) · max(0, cos θ) for Δs ≥ 0, and
> Δs for a step back

where the speed factor s is 1 up to the target speed v\*, and
(v\* − k (v̄ − v\*)) / v̄ above it. A second's pay is therefore the speed up
to the target, then falls by k mm/s for every mm/s too fast: with k = 1,
none at twice the target and a charge beyond. The team receives the mean
over the units.

- **The target v\***, 90 mm/s, 0.6 of the full body's length a second. *S.
  heros* walked a treadmill at 0.5 to 1.5 body lengths a second (Anderson
  et al. 1995); a scripted wave on v5 reaches 90 mm/s at 3 steps a second.
  It is set in millimetres, so it asks every leg the same on every body.
- **A target, not a cap.** Run 03 paid at most the cap (s = v\* / v̄): above
  it speed earned nothing but cost nothing either while energy was free, and
  the body learned a run at 10 Hz and 230 to 260 mm/s, which energy added
  later did not undo. Paid less above the target, a unit gains by slowing
  down, and since k = 1 makes being too fast cost as much as being too slow,
  the noise centres the walk on the target. Hwangbo et al. (2019) likewise
  charge the error from a commanded velocity either way.
- **On the averaged speed**, like every force the reward weighs. Run 01
  capped each step at 1.8 mm: a stride's surges were cut, its dips paid in
  full, so a faster stride, with higher dips, still earned more, and seed 2
  walked 119 mm/s. On the averaged speed a stride's surges and dips cancel.
  Each episode starts the average at the target, so a slow start is paid in
  full; a fast start is charged a little late while its average climbs.
- **The band b**, the sight radius, 16.5 mm on v5: a unit is paid for the
  line only while it can see it ([observations.md](observations.md)). Run 28
  used 20 mm; within 10 mm walking faster up to the cap did not pay, the
  body's own sway costing too much.
- **Facing** along the line, and **a step back charged in full**, so that a
  body walking sideways earns less and rocking never pays.
- **The mean**, because the chain moves together: sharing the gain is what
  first moved the walk into the policy (run 23).

**The squeeze.** At each joint between units, how hard the chain is pushed
together along the body, the tows' push taken out, squared against 0.3 unit
weights and averaged over the joints, 53 mm/s at that force; a pull is free.
A chain pushed from behind buckles (run 04), and many small pushes, each
nearly free, add up (run 24).

**Energy, the body's mean.** Each unit's motors: pushing work, braking work
at a quarter of its price, and the force cost, each command squared for as
long as it is held, priced so that force makes 90% of a walk's energy, as
in muscles of a gram-sized animal (Kram & Taylor 1990). It replaces gait
rules: standing costs force, dragging costs work, and sharing a load among
legs costs less than one leg carrying it. In the team account, beside the
line, it no longer pulls against exploring (cycle 14); paid by each unit,
it pays a unit to do less while the slower walk is shared (cycle 16).

## The own account

**Force share.** The net force its two joints put on the unit along the
body, plus its tow, averaged over 1 s, squared, at the squeeze's price.
Positive: it is dragged; negative: it pushes the others. The chain's force
at any joint is the sum of the shares behind it, so this is each unit's
exact part (cycle 14).

**Hanging.** The weight its neighbours hold up for it, and its belly's load
on the floor, averaged over 1 s, at twice the price of one motor fully on
for a unit weight (cycle 16). A carried unit saves most of its energy while
dragging it costs the others little; hanging closes that hole. A unit may
take for free what it weighs beyond the others' mean, which on the headless
tester is nothing.

**Pain, removed** (its price 0; the measure is still logged). It charged
how hard a unit's legs press on other legs. In run 01 the middle units held
their legs raked back, never reaching forward of neutral: pain taught them
to keep apart rather than to step in turn, and to find a wave they must be
free to fail. It may return as its own run once a wave forms.

## The ramp

Energy and the squeeze weigh 0.25 at the first update cycle and rise in a
straight line to their full weight at cycle 161. This is Hwangbo et al.
(2019)'s curriculum: every cost but the objective is multiplied by a factor
that starts at 0.3, chosen "to prevent the initial tendency to stand still",
and rises with the iterations toward 1; they note that low penalties on
effort give unnatural motion and high ones a standing body. Both happened
here. Run 01 started at half (with pain), and for untrained agents the costs
outweighed the line (the fresh check: −9.7 mm/s a unit), which taught timid
legs. Run 03 started at 0 until the walk reached the cap, and the walk became
a run that the full costs did not undo. On this body an untrained unit's
team account breaks even at 0.25 (−0.6 mm/s). The ramp follows time rather
than the walk's progress: it is predictable, the seeds stay comparable, and
no loop forms between the cost and the walk it waits for; holding the speed
is the target's job. In runs 01 and 03 the walk formed between cycles 74
and 100, when this ramp weighs 0.6 to 0.7.

## The ledger

Before a run, scripted walks on the built code check that the good wave walk
pays more than every fault we know (the study's `Training/analysis/ledger.py`).
On this branch's code (2026-10-20, the design above) the good wave walk at
the target, 3 steps a second and 90 mm/s, earns +84 mm/s a unit from the
line, and the team +65 with the ramped costs in full (+80 at 0.25). Every
fault pays less, own weighed 0.8 or 4 and the ramped costs at 0.25 or in
full, by 15 mm/s and more for walking faster (1.4 times the step rate,
111 mm/s) and 37 to 51 for a run at twice it. The one exception is the wave
running backward, tail to head, a near tie: it loses at own weight 0.8
(−2.6 mm/s in full, −6.0 at 0.25) and ties only in the stress test at 4
(+2.6 in full): the reward does not care about the wave's direction. On the
half body, whose first unit carries the head, the same walk squeezes the
chain (−22 mm/s a unit): the others push the heavy front unit along, one
more reason to train headless first.
