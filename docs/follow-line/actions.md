# Actions

Agreed with the user on 2026-10-20, as the cooperators' (`docs/cooperators/actions.md`
on that branch); from run 11 (the user's decision, 2026-10-21) each command
is a target angle.

| | Every unit but the rear | The rear |
| --- | --- | --- |
| Leg motors | 8: left sweep, lift, knee, ankle, then the right leg's | The same 8 |
| Spine | 1: the yaw motor of the joint behind it | None (its ninth value goes to no motor) |

- **Target motors** (from run 11, model v6 headless). Each command u, from
  −1 to 1, is a target angle across its joint's range, m + c·u, with m the
  range's middle and c half its width. The motor pulls the joint toward it
  as a spring and a damper, as a muscle pulls a limb toward the position
  its activation sets (the equilibrium-point hypothesis: Feldman 1966;
  Bizzi et al. 1992):

  > force = kp · (m + c·u − q) − kv · q̇, within ±1 of the motor's strength

  kp gives the full strength a quarter of the range away from the target,
  and kv damps the joint's own inertia critically, so a held target is
  reached without ringing. Bodies, joints, masses and each motor's
  strength are v5's (`models/assembly_v6_headless.xml`, made by
  `archive/model_v6_targets/make_model_v6.py` in the study's folder).
  The spring and the damper give each leg a posture to return to and
  resist quick changes, which a torque does not (Peng & van de Panne 2017:
  on simulated legged bodies, targets learned faster and more robustly,
  with better motion, than torques).
- **Torque motors** (runs 01 to 10, model v5 headless): each command was a
  share of its motor's greatest torque, the muscle's activation; the
  angle followed from the physics.
- Commands are a Gaussian squashed by tanh, never clipped.
- **The spine's reach.** A spine target spans its joint's ±15° by itself,
  so its command is not scaled (`spine_action_scale` 1). With torques, a
  spine motor reached its joint's limit for any command above about 0.35,
  so each received 0.35 of its command and the whole range spanned the
  joint's (cycle 6: runs 09 and 10 pinned their spines at the limits).
- **The up–down joints are passive springs**, ±8°: a flat line never asks
  the body to arch. Lifting the front is for the head's later model.
- **Every 20 ms**, 50 decisions a second, as legged_gym. A centipede steps
  about six times a second, so a unit decides about eight times a step.
