# The cooperators design

This folder holds the design of the `cooperators` workspace branch, one file
per part: what is decided, what is proposed, and what is still open. The
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
| Critic | [critic.md](critic.md) | Decided |
| Reward | [reward.md](reward.md) | Agreed: goal in millimetres, energy and pain |
| Observations | [observations.md](observations.md) | Agreed: radius 2, a 42-value block |
| Actions | [actions.md](actions.md) | Agreed: 8 leg torques and the spine joint behind, every 20 ms |
| Forces | [forces.md](forces.md) | Model v4's forces checked against biology; no change proposed |
| Task | [task.md](task.md) | Agreed, with numbers |
| Research | [research.md](research.md) | The projects and mechanisms the design draws on |

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
