# Observations

**Status (2026-10-14): what each segment may see is decided; what each
segment's description holds is open.**

## Decided

- **Only the neighbourhood.** When it acts, a segment sees only itself and
  its neighbours within the observation radius (the constraints in
  [README.md](README.md)). The head also sees the target, which only it can
  know.
- **Radius 2 to start.** A segment sees five segments: itself, two ahead
  and two behind. The tail-ward wave of a real centipede's legs spans a few
  legs, about five in the Tohoku model (Yasui et al., Frontiers 2022), so a
  window of five covers about one wavelength around each segment: what it
  needs to sense where it sits in a wave. A coincidence of scales, not
  proof, but it points the same way.
- **The critics see a little more, in training only:** what their actor
  sees, plus the neighbours' actions of the same step
  ([critic.md](critic.md)).

## Open

- **What describes a segment:** which of its quantities its neighbours and
  itself see. Main's block (height, orientation, leg angles and speeds,
  body speeds, contacts; [environment.md](../environment.md#what-each-segment-sees))
  is the starting reference, not adopted by default. The research points at
  the neighbours' last actions (the hexapod, Schilling et al. 2020) and the
  load and contact of the leg ahead (biology, [research.md](research.md)).
- **What the head sees of the target**, including the time left on it
  ([task.md](task.md#the-numbers-proposed-2026-10-14)).
- **Whether a segment sees its heading in the world.** Its orientation
  includes the direction it faces, which carries no information about the
  task ([task.md](task.md#the-starting-heading-proposed-2026-10-14));
  gravity's direction and the neighbours' relative orientations may be
  enough.

## Later experiments

The radius changed, to show how much seeing the neighbours matters for
cooperation to emerge; then the final model of 21 segments with several
radii ([README.md](README.md#the-plan)).
