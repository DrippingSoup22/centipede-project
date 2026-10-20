# Reward

Agreed with the user on 2026-10-20: the cooperators' run 28 reward
(`docs/cooperators/reward.md` on that branch, cycles 14 to 18, where every
term's history and measurements are), with nothing left of the head's
special role. Changed after run 01 (the user's decisions, 2026-10-20): the
cap holds on each unit's speed averaged over 1 s, pain is removed, and
energy and the squeeze wait until the body walks the line.

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

> line = Δs · min(1, c / v̄) · max(0, 1 − (d / b)²) · max(0, cos θ) for
> Δs ≥ 0, and Δs for a step back

The team receives the mean over the units.

- **The cap c**, 90 mm/s, 0.6 of the full body's length a second. *S.
  heros* walked a treadmill at 0.5 to 1.5 body lengths a second (Anderson
  et al. 1995); a scripted wave on v5 reaches 0.6 at 3.5 steps a second.
  Above the cap speed earns nothing, so only walking better pays. It is set
  in millimetres, so it asks every leg the same on every body.
- **On the averaged speed**, like every force the reward weighs. Run 01
  capped each step at 1.8 mm: a stride's surges were cut, its dips paid in
  full, so a faster stride, with higher dips, still earned more, and seed 2
  walked 119 mm/s. Scaled by the averaged speed, a unit earns at most the
  cap per second however its stride surges. Each episode starts the average
  at the cap: only the speed above the cap counts, so a slow start is paid
  in full. A fast one still gains a little while its average climbs, about
  c · 1 s · ln(v / c) per episode: at twice the cap 62 mm, 3 mm/s over a
  20 s episode (run 03 measured it: its noise-free walks at 136 to 187 mm/s
  were paid 92 to 95 mm/s).
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

## The ramp: the walk first, then its cost

Energy and the squeeze weigh nothing until the untowed noise-free worlds
earn 0.9 of the cap from the line (81 mm/s); from the next update cycle
their weight rises to full over 64 cycles. The body first learns to follow
the line however it can, a messy walk included, and energy then refines
the walk toward the cheapest one, the wave (the user's idea). In run 01
both started at half their weight, and for untrained agents they cost more
than the line paid (the fresh check), which teaches doing less (Hwangbo et
al. 2019's curriculum factor; cooperators, cycle 15). In run 01 the line
reached 84 mm/s by cycle 96 and held 84 to 89 after it, so the gate opens
once the walk reaches the cap. A continued run keeps the cycle its ramp
rose from.

## The ledger

Before a run, scripted walks on the built code check that the good wave walk
pays more than every fault we know (the study's `Training/analysis/ledger.py`).
On this branch's code (2026-10-20, the design above, pain at 0) the good
wave walk, 103 mm/s, earns the line's cap, +90 mm/s a unit, and the team
+71 with the ramped costs in full. Every fault pays less, own weighed 0.8
or 4 and the ramped costs at 0 or in full, by 3 to 1,500 mm/s; walking
faster than the cap loses even with the costs at 0 (−2.6 mm/s). The one
exception is the wave running backward, tail to head (+1.0 mm/s at 0.8 in
full, −0.7 at 0), a near tie: the reward does not care about the wave's
direction. On the half body, whose first unit
carries the head, the same walk squeezes the chain (−22 mm/s a unit): the
others push the heavy front unit along, one more reason to train headless
first.
