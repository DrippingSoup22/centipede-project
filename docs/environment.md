# Environment, rewards, and evaluation

This document owns target generation, episode behavior, reward design, and
training difficulty. Confirmed requirements and unaccepted proposals remain
separate. Target sensing is defined in [control.md](control.md); physical body
and contact geometry are defined in [model.md](model.md).

## Navigation and episode design

- Sample a target point stochastically within defined boundaries on the ground
  plane. Boundary values and the sampling distribution remain to be agreed.
- Start in a simple environment with targets relatively close to the centipede.
  The initial distance range and any later difficulty schedule remain open.
- The head must reach the target to obtain an arrival reward. Use the head as
  the reference for target arrival, not the body's center or an arbitrary segment.
  The exact head reference point, arrival tolerance, and reward magnitude remain
  open; all agents receive the shared arrival reward.
- The task must accommodate changes in direction, including targets behind the
  centipede that require it to turn around.
- A proposed later training stage continues the same episode after reaching a
  target, assigns another target, and lets the centipede navigate to it without
  resetting its body. Target arrival must therefore be distinguishable from
  episode termination; do not hard-code arrival as always ending an episode.
- The target sampling procedure, arrival criterion, target information available
  to each agent, initial training stages, episode limits, and failure conditions
  remain open. Continued target sequences are a proposed later stage, not an
  instruction to run indefinitely or implement a curriculum now.

## Reward design

Agreed requirements:

- Give all agents a shared positive reward when the head reaches the target.
- Apply a small negative reward per time step to encourage faster arrival.
  Interpret this as elapsed simulation/control time, not individual footfalls;
  exact timing, scaling, and allocation of the penalty remain to be agreed.
- Keep the reward minimal and avoid prescribing how legs must move. The user
  wants exploration and cooperation without unnecessary behavioral bias.
- Reward magnitudes and all other terms remain undecided.

Assistant proposals, pending agreement:

- Expand target distances and directions as performance improves. Nearby initial
  targets are agreed, but this expansion schedule is still a proposal. Keep the
  eventual evaluation target distribution separate.
- If dense feedback is needed, consider a shared potential-based shaping term
  `beta * (gamma * Phi(s_next) - Phi(s))`, with a fixed bounded potential based
  on negative head-to-target distance. Match the learner's discount. This is not
  simply raw distance reduction when `gamma < 1`.
- Before adopting shaping, define correct terminal potentials, time-limit
  bootstrapping, and target-change transitions with the target included in state.
  Do not assume a single-agent policy-invariance result guarantees convergence
  for partially observed cooperative learners, or that shaping solves exploration.
- Initially avoid action-effort penalties and explicit gait/phase rewards. Add
  terms only to address an observed problem and with the user's agreement.

Body contact and falling (under discussion, not agreed reward terms):

- The user proposes penalizing a segment's main body touching the ground and
  tentatively favors charging only that segment's agent.
- Distinguish a brief body-ground contact from sustained dragging or collapse;
  normal foot-ground contact must not count as falling. Define the contact
  criterion and any persistence threshold after inspecting the physical model.
- Assistant recommendation: consider a small shared cost based on the fraction
  of segments with sustained body-ground contact, rather than assigning blame
  solely to the touching segment. Connected neighbors can cause or prevent a
  segment's fall, so contact location does not identify the responsible agent.
- A candidate shared term is `-lambda_contact * mean(contact_flags)`, applied
  to every agent per control step; flags indicate sustained body-ground contact.
  Averaging keeps its maximum scale independent of segment count. This adds an
  explicit preference for body clearance; it is not policy-invariant shaping.
- A local-only cost is an alternative for comparison, not a confirmed decision.
  It provides segment-specific feedback but gives agents different objectives
  and may discourage helping a neighbor when doing so risks their own contact.
- Do not terminate on the first body contact. Provisionally allow recovery and
  consider termination only for an agreed sustained collapse condition. Neither
  a termination criterion nor a contact penalty weight is settled.
- Keep any contact cost modest enough to test movement and recovery rather than
  favor standing still; audit it jointly with progress, arrival, and time costs.

Leg-leg contact (agreed structure; scale still open):

- Leg collisions remain enabled in the physical model and joint limits are not
  reduced solely to eliminate them. Avoiding harmful contacts is part of the
  cooperative policy problem.
- Give each segment a leg-leg contact flag distinct from normal foot-ground and
  body-ground contact. A segment learner receives a local penalty when its flag
  is active; a collision involving legs from two segments therefore activates
  the flag and local cost for both owners.
- Whether to count contact onset, duration, impulse, or any contact during one
  control interval remains undecided. The penalty weight is also open and must
  stay small enough that avoiding movement is not preferable to exploration.

Design checks:

- With only a constant time cost and no success, equal-duration trajectories
  receive equal returns whether they move or idle. The problem is missing useful
  feedback, not an intrinsic reward advantage for zero action.
- In fixed-duration continuing episodes, a constant time cost is also constant
  across policies; repeated arrival rewards, rather than that cost alone, must
  incentivize reaching more targets sooner.
- Ensure failure termination cannot become an easy way to avoid time costs.
  Award arrival only once per target and audit target changes for reward exploits.
- Reaching targets does not establish walking or a wave-like gait. Measure and
  inspect these separately; do not promise biological motion from navigation
  rewards alone. Compare any shaping against the agreed base reward.

## Evaluation questions still open

Walking ability, target-reaching performance, and resemblance to a wave-like
centipede gait must be assessed separately. Reward increases or successful
navigation alone do not establish either walking or biological gait resemblance.
Exact success thresholds, phase relationships, the roles of leg and body waves,
and evaluation scenarios remain to be agreed. No metric values have been measured.

