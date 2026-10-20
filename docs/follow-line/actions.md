# Actions

Agreed with the user on 2026-10-20, as the cooperators' (`docs/cooperators/actions.md`
on that branch).

| | Every unit but the rear | The rear |
| --- | --- | --- |
| Leg motors | 8: left sweep, lift, knee, ankle, then the right leg's | The same 8 |
| Spine | 1: the yaw motor of the joint behind it | None (its ninth value goes to no motor) |

- **Torque motors.** Each command, from −1 to 1, is a share of its motor's
  maximum torque: the action is the muscle's activation, so energy's force
  cost is the command squared. Commands are a Gaussian squashed by tanh,
  never clipped.
- **The spine at 0.35.** A spine motor reaches its joint's limit for any
  command above about 0.35, so each receives 0.35 of its command and the
  whole range spans the joint's (cycle 6: runs 09 and 10 pinned their
  spines at the limits).
- **The up–down joints are passive springs**, ±8°: a flat line never asks
  the body to arch. Lifting the front is for the head's later model.
- **Every 20 ms**, 50 decisions a second, as legged_gym. A centipede steps
  about six times a second, so a unit decides about eight times a step.
