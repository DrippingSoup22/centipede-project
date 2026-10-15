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
| Critic | [critic.md](critic.md) | Built: a value and a baseline per critic; how each update learns (batch, minibatches, epochs, a learning rate adapted to KL) |
| Reward | [reward.md](reward.md) | Built: goal in millimetres, energy, pain and the chain in millijoules; force and chain prices measured |
| Observations | [observations.md](observations.md) | Built: radius 2, a 42-value block, 43 with the joint forces |
| Actions | [actions.md](actions.md) | Built: 8 leg torques and the spine joint behind, every 20 ms |
| Forces | [forces.md](forces.md) | Model v4's forces checked against biology; no change proposed |
| Task | [task.md](task.md) | Built, with numbers |
| Research | [research.md](research.md) | The projects and mechanisms the design draws on |

## In the code

Built on 2026-10-14 (cycle 1). The configuration chooses the design with
`[environment] design = "cooperators"`; every new setting defaults to main's
behaviour, so main's configurations keep their meaning.

| Part | Where | Settings |
| --- | --- | --- |
| Environment | `environment/cooperative_environment.py`, made by the front file's `create_environment` | `[environment] design`, `max_episode_steps`, `observation_radius` |
| Task | `environment/cooperative_task.py` | `[environment.cooperators]` |
| Observations | `environment/cooperative_observations.py` | `[environment.cooperators] observe_joint_forces`, off by default |
| Reward | `environment/cooperative_reward.py` | `[environment.cooperators]`; the chain by `chain_price`, 0 (off) by default |
| Forces and work | `environment/simulation`, measured after every physics step (`measure_effort`), CPU and GPU; the joint forces too when the chain or the observations use them (`measure_joint_forces`, MuJoCo's `cfrc_int`) | |
| Critics | `agents/`, with RL_lib's PPO on its `cooperators` branch | `[agents] advantage_weights`, `critic_baselines`, `value_normalization` |
| Level | `experiment/`: moved after every update cycle, logged as `task`, kept in checkpoints | |
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
