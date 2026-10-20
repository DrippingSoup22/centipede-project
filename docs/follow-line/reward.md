# Reward

Agreed with the user on 2026-10-20: the cooperators' run 28 reward
(`docs/cooperators/reward.md` on that branch, cycles 14 to 18, where every
term's history and measurements are), with nothing left of the head's
special role.

## Two accounts

Each 20 ms step every agent receives two numbers:

```text
team = line − squeeze − energy                (the same for all ten)
own  = − force share − hanging − pain        (each unit's own)
```

Both are counted in millimetres of line, a millijoule worth 1.59 mm. Each
has its own critic, and learning weighs them 1 : 0.8 ([critic.md](critic.md)).

## The team account

**The line.** For each unit, with Δs how far its centre went along the line
on the step (mm), d its centre's distance from the line (mm) and θ its
body's angle from the line's direction:

> line = min(Δs, c) · max(0, 1 − (d / b)²) · max(0, cos θ) for Δs ≥ 0, and
> Δs for a step back

The team receives the mean over the units.

- **The cap c**, 1.8 mm a step: 90 mm/s, 0.6 of the full body's length a
  second. *S. heros* walked a treadmill at 0.5 to 1.5 body lengths a second
  (Anderson et al. 1995); a scripted wave on v5 reaches 0.6 at 3.5 steps a
  second. Above the cap speed earns nothing, so only walking better pays.
  It is set in millimetres, so it asks every leg the same on every body.
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

**Pain.** How hard its legs press on other legs, averaged over 1 s, squared,
priced like force. Legs could otherwise push off their neighbours' legs, and
it ties each unit's steps to its neighbours': two legs of a side touch when
the rear one sweeps into the front one.

## Ramps

Energy, the squeeze and pain start at half their weight and reach it fully
at update cycle 129, so that exploring stays worth it while the body learns
(Hwangbo et al. 2019's curriculum factor; cooperators, cycle 15).

## The ledger

Before a run, scripted walks on the built code check that the good wave walk
pays more than every fault we know (the study's `Training/analysis/ledger.py`).
On this branch's code (2026-10-20) the good wave walk, 103 mm/s, earns the
team +69 mm/s a unit; every fault pays less, own weighed 0.8 or 4 and the
ramped costs at half or in full, by 6 to 1,900 mm/s, but the wave running
backward, tail to head (+1.2 to +2.1 mm/s at 0.8), a near tie: the reward
does not care about the wave's direction. On the half body, whose first unit
carries the head, the same walk squeezes the chain (−22 mm/s a unit): the
others push the heavy front unit along, one more reason to train headless
first.
