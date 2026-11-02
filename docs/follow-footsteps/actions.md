# Actions

Agreed with the user on 2026-11-02.

| | Units 2 to 10 | Unit 1 |
| --- | --- | --- |
| Leg motors | 8: left sweep, lift, knee, ankle, then the right leg's | The same 8 |
| Spine | 1: the yaw motor of the joint **ahead** of it, between it and its leader | None: ahead of it is the head, whose neck no walker commands |

- **The joint ahead.** In follow-line each unit commanded the joint behind
  it, which places the unit behind, so it was charged for how far the units
  behind it kept from the line. Here each unit steers itself onto its
  leader's path: its marks depend on its own legs and its own spine, and it
  needs to see nothing behind. The model stores each spine motor with the
  unit behind its joint (`docs/model.md`, "Motors"), so each unit now
  commands its own; the physics is the same, since a joint's motor pushes
  both units equally, and only who decides changes. There are still nine
  spine motors: unit 1 commands none and unit 10 one.
- **Target motors**, v6's: each command u, from −1 to 1, is a target angle
  across its joint's range, and the motor pulls the joint toward it as a
  spring and a damper (follow-line's [actions](../follow-line/actions.md),
  `docs/model.md`, "Target motors"). A spine target spans its joint's ±15°.
- Commands are a Gaussian squashed by tanh, never clipped.
- **Every 20 ms**, 50 decisions a second, as in follow-line.
- **The neck** is passive in v6: no agent commands it ([README.md](README.md#open)).
