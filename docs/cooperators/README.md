# The cooperators design

This folder holds the design of the `cooperators` workspace branch, one file
per part: what is decided, how it is built, and what is still open. The
other documents in `docs/` describe the program this branch started from
(`main` at d8b8671); where this folder decides differently, this folder
holds on this branch.

## The question

Can independent agents, each seeing only its neighbours, learn to cooperate
so that one body walks like a centipede? The gait is the evidence of
cooperation, never something the reward describes.

## Why a new line

The earlier workspace, `oscillators` (runs 8 to 58), paid each segment for a
good step as we defined it: foot hold, carry, grip, the step value. The gait
was designed through the reward, every rule about how to move was gamed, and
the followers ended up selfish by construction. When the followers did share
the task, their critics could not see the target and explained none of it,
so the shared goal was dropped to make the critics learn. This line undoes
that trade of cooperation for a learnable critic.

## The constraints

Set by the user on 2026-10-14.

1. **Independent agents.** Each segment has its own actor, its own critic
   and its own experience. Nothing is shared between segments: no
   networks, no estimates, no data.
2. **A local view.** When it acts, a segment sees only itself and its
   neighbours within the observation radius.
3. **General rules.** The reward says what to achieve, never how to walk.
   Different solutions are welcome.
4. **The measure of success.** A real centipede gait emerges from learning.

## Principles that follow

- **The pairing rule** (the user's). A critic must see the causes of the
  reward it predicts. A reward that depends on something outside its view
  is an effect without a cause, and the advantage it produces is noise. So
  the critic and the reward are one decision.
- **The goal first.** Moving toward the goal is always worth more than what
  it costs; otherwise standing still becomes the best choice (run 57).
- **Outcomes, never form.** A term may judge what happened, never how the
  legs moved.
- **Cooperation through shared outcomes.** A segment may care about its
  neighbours' results; it is never paid to be like them or to beat them.

## The parts

| Part | File | Status |
| --- | --- | --- |
| Critic | [critic.md](critic.md) | Built: a value and a baseline per critic; how each update learns (batch, minibatches, epochs, a learning rate adapted to KL); from cycle 3 the followers have the cost critic only; from cycle 4 a trail critic for every segment, the head's weighed 1 : 0.25 : 1; from cycle 5 three critics each, one per priority, weighed 40 / 35 / 25; from cycle 6 two again, goal and health, 1 : 0.8; from cycle 14 the team's and each one's own; from cycle 16 each critic's scale forgets over about 10 updates |
| Reward | [reward.md](reward.md) | Built: goal in millimetres, energy, pain and the chain in millijoules; force and chain prices measured; cycle 3: only the head has a goal, the followers pay the chain at ten times the force price; cycle 4: run 01's design without the chain, the followers' goal the trail (how far off it, not how far along), each segment paying its own offset and the offset of the one it steers, and carrying itself; cycle 5: the followers' coordination first (the averaged net force, pushing ahead counted twice), the trail paid by time, then health; the head's own weight; energy shared by sides; cycle 6: the followers paid the distance they walk on their trail, less their squeeze on the segment ahead, the head its target less being pushed, health energy, pain and hanging; cycle 7: the head's line, each follower paid the line its tail erases, less, squared, its distance from the line and its front joint's force; the head its target alone; cycle 8: the forces averaged over 1 s; cycle 10: each follower's own ground force, its joints' net force plus its tow, in place of its front joint's; cycle 11: its own joint force, the net force alone, so that each segment brakes its own tow; cycle 13: the reward shared, each follower paid the followers' mean goal, pain and hanging shared with the neighbours like energy, pain ramped in over the first cycles; cycle 14: split by owner, the team paid the followers' mean line and distance and the body's mean energy, each segment its own force share, pain (from half its weight) and hanging, a millijoule worth 10 mm; cycle 15: the line paid near it (times its closeness, no distance cost) and drawn where the head keeps pointing (its sideslip drifts it), the chain's squeeze charged to every follower, energy and the squeeze ramped in with pain; cycle 16: energy each segment's own, shared with its neighbours (40% itself, 25% each nearest, 5% each next), pain on the legs' averaged press |
| Observations | [observations.md](observations.md) | Built: radius 2, a 42-value block, 43 with the joint forces, 45 with the offset and the net joint force (cycle 4), 46 with both joint forces averaged (cycle 5), 48 with the tail's place on the head's line (cycle 7), 50 with the joint forces of the step and the tow level (cycle 10) |
| Actions | [actions.md](actions.md) | Built: 8 leg torques and the spine joint behind, every 20 ms; from cycle 6 the spine motors receive 0.35 of their command; how actions explore: pink noise, gSDE from cycle 8, a fixed spread schedule in cycle 9, gSDE's learned spread again from cycle 10 |
| Tow | [tow.md](tow.md) | Built (cycle 10): every segment pushed forward by a share of its weight in training, fading as the mean action, measured in worlds of its own, learns to walk; cycle 12: the push shrinks with each segment's speed to none at a help speed, the fade reads the untowed walk, and the training worlds draw from a floor; cycle 15: braking a tow measured the dearest answer to it, a gentle top of 0.2 |
| Forces | [forces.md](forces.md) | Model v4's forces checked against biology; no change proposed |
| Task | [task.md](task.md) | Built, with numbers |
| Research | [research.md](research.md) | The projects and mechanisms the design draws on; how a long body turns (cycle 15) |

## In the code

Built on 2026-10-14 (cycle 1). The configuration chooses the design with
`[environment] design = "cooperators"`; every new setting defaults to main's
behaviour, so main's configurations keep their meaning.

| Part | Where | Settings |
| --- | --- | --- |
| Environment | `environment/cooperative_environment.py`, made by the front file's `create_environment` | `[environment] design`, `max_episode_steps`, `observation_radius` |
| Task | `environment/cooperative_task.py` | `[environment.cooperators]` |
| Observations | `environment/cooperative_observations.py`: the 48-value block, 50 with the joint forces of the step | `[environment.cooperators] averaging_s`, `observe_instant_joint_forces`, `observe_tow` |
| Reward | `environment/cooperative_reward.py`: goal and health (cycles 7, 10, 11 and 13), or team and own (cycle 14) | `[environment.cooperators]`: `reference_pay_mm_per_s`, `line_distance_scale_mm`, `line_distance_cap_mm`, `joint_force_scale`, `force_price`, `hanging_price`, `head_free_support`, `energy_sharing_decay`, `averaging_s`, `goal_team_share`, `health_sharing`, `pain_ramp_cycles`, `pain_ramp_start` (the experiment sets the pain's weight before each update cycle), `reward_channels`, `mm_per_mj`, `line_closeness_mm`, `squeeze_weight`, `ramp_team_costs`, `energy_side_shares`, `pain_averaged` |
| Tow | `environment/cooperative_tow.py`; the push through the simulation's `step` (`body_forces`, CPU and GPU) | `[environment.cooperators]` `tow_start`, `tow_step`, `tow_window_steps`, `tow_walk_mm_per_s`, `tow_lift_off_hz` |
| Mean-action worlds | `agents/agents.py`, `interaction_loop/`, `experiment/`: the last worlds act with the mean action, unstored; logged as `tow` | `[run] mean_action_worlds_per_group` |
| The head's line | `environment/cooperative_line.py`: the line the tip draws, each segment's erase point, and its tail's place on the line after every step; from cycle 15 the head's sideslip and the drift it gives the line | `[environment.cooperators] line_drift_leak_s` |
| Reward channels | `environment/cooperative_environment.py`'s `segment_reward_channels`: every segment learns from both channels | |
| Spine | `environment/cooperative_environment.py`: each spine motor receives a share of its command | `[environment.cooperators] spine_action_scale` |
| Forces and work | `environment/simulation`, measured after every physics step (`measure_effort`), CPU and GPU; the joint forces too, each segment's front joint force and net joint force (`measure_joint_forces`, MuJoCo's `cfrc_int`); each segment's front end, the head's tip or its spine joint | |
| Critics | `agents/`, with RL_lib's PPO on its `cooperators` branch | `[agents] advantage_weights`, `critic_baselines`, `value_normalization`, `value_normalization_horizon` |
| Exploration | `agents/segment_agent.py`, with RL_lib's gSDE (`GaussianPolicyNetwork`'s `"gsde"`, `SquashedGaussianPolicy`) on its `cooperators` branch (cycle 8) | `[agents] state_dependent_noise_steps`, `actor_activation`, `initial_action_std` |
| Level | `experiment/`: moved after every update cycle, logged as `task`, kept in checkpoints, the mean-action worlds not counted | |
| Diagnostics | `environment/cooperative_diagnostics.py`: each step, each 20 s episode, and the gait read from the feet (each foot's lag behind the foot ahead and the foot opposite); the terminal's own line in `experiment/progress.py`. No training report, since the monitor shows the log; the evaluation page has its own layout (section 2b of `experiment/report_page.html`) | |
| Recordings | `interaction_loop/recording.py`, for both designs: one episode of each of the first worlds, each from its own episode start, ending by 25, 50, 75 and 100% of the update cycles; worlds ranked by targets reached | `[run] record_worlds`, 0 (nothing) by default |

Recordings are the one change to main's behaviour: on the user's request
(2026-10-14) nothing is recorded unless asked, and the earlier recording
settings are retired.

The first run's configuration is `configs/cooperators/01_baseline.toml` in
`../CentipedeTraining`, and the energy check that set the force price is
its `analysis/energy_check.py`.

## The plan

1. **Twelve segments first**, with main's model v4 (twelve units, legs with
   an ankle, torque motors; [model.md](../model.md)). A centipede of twelve
   that learns to walk is the precondition for everything after it.
2. **The observation radius changed**, to show how much seeing the
   neighbours matters: whether cooperation still emerges, more slowly, or
   not at all.
3. **The final model: 21 segments**, trained again, with several radii.

Training runs live in the sibling project `../CentipedeTraining`
(`results/cooperators/`).
