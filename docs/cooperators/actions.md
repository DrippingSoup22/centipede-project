# Actions

**Status (2026-10-14): open.** Nothing is decided yet beyond the
constraints.

## What the constraints fix

Each segment commands only its own part of the body and chooses its action
on its own ([README.md](README.md)).

## The starting point

Main's action ([environment.md](../environment.md#actions-and-timing)),
the reference, not adopted by default: each segment sends one number per
leg motor of its two legs, eight with model v4's ankles, each the share of
that joint's maximum torque, held for 20 ms. With spine control, every
segment but the rear also commands the spine joint behind it.

## Open

- **Torque motors** (main's v4) **or position actuators**, which the
  oscillators' models used.
- **The spine:** commanded by the segments, or left passive.
- **The decision rate:** an action every 20 ms, or another.
