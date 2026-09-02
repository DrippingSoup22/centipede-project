# Working rules for Codex

## Read first

- Read this file before making changes. Use `README.md` for the project goals,
  current status, folder layout, and documentation index.
- Read the relevant design documents before working on their areas:
  `docs/model.md`, `docs/control.md`, and `docs/environment.md`.
- Keep this file limited to working instructions and project boundaries.
  Put model, agent/network, environment, reward, and experiment design in the
  appropriate document, not here.

## Scope and boundaries

- Work in `Centipede/`. Ignore the sibling `backup/` folder: do not inspect,
  modify, or use it as a source.
- Reuse relevant algorithms and utilities from the sibling `../RL_lib` rather
  than copying them here. Keep centipede-specific code in this project.
- Reusable RL changes belong in `RL_lib`; read its `AGENTS.md` and agree on the
  scope before modifying it. Explain missing capabilities instead of silently
  replacing the shared library.
- Keep folders, dependencies, and abstractions simple. Add them only when the
  current task needs them.
- Treat `archive/` as preserved, inactive history. Do not import, load, or use
  archived files as defaults for builds, tests, or experiments. Move a meaningful
  superseded artifact there instead of deleting it; disposable caches need not be
  archived. Keep the entire folder local and outside version control.
- Before proposing a commit, review the active project tree and move meaningful
  superseded artifacts into `archive/`. Do not commit or push automatically after
  making changes. Wait for the user to explicitly request each commit and push so
  the project can be reviewed and cleaned first.

## Decisions and documentation

- Work incrementally within the user's current request. Do not implement future
  stages simply because they appear in a plan or the project goals.
- The eight-unit active-yaw assembly is frozen as the v1 flat-ground physical
  baseline. Its canonical active artifact is `models/assembly.xml`. The isolated
  head/trunk XML models remain as model references. Generators, source
  configurations, tests, and validation evidence are preserved under `archive/`
  and are inactive unless the user explicitly reopens model development.
- Do not change v1 morphology, mass, inertia, contacts, joint physics, timestep,
  or actuator set during environment or learning work. A necessary physical
  change must become a separately named version with new validation artifacts.
- For model revisions, review a small visual/geometry prototype with the user
  before adopting the shape and doing broader physics tuning. Passing mechanical
  checks alone does not settle appearance or biological resemblance.
- A fully passive-yaw spine is a possible later comparison, not an unresolved v1
  requirement. Do not change the v1 actuator set in place.
- In the first learning stage, keep all spine-yaw motor commands at zero and do
  not expose spine actions or state to the agents. This disables learned spine
  control while retaining the frozen XML's passive joint physics. Later active
  spine control is an environment/policy version change.
- Fix the initial neighbor observation radius at one. Include only existing
  immediate chain neighbors; do not add padding or masks for missing end
  neighbors. Independent agent networks may use different observation sizes.
- Keep leg-leg collisions enabled. Do not shrink joint ranges, add collision
  exclusions, or impose coupled action clamps solely to prevent legs touching;
  coordination must remain part of the agents' problem. Any contact reward is a
  separate environment decision and must not be hidden in model mechanics.
- Expose one leg-leg contact flag per segment and apply an agreed local penalty
  to each participating segment learner. Contact aggregation and penalty weight
  remain unresolved environment decisions.
- Distinguish confirmed requirements, proposals, assumptions, and measured results.
  Discuss consequential design choices before implementing them. Do not silently
  turn recommendations or general approval of a plan into fixed parameter choices.
- Update the document that owns a decision when it changes. Keep one detailed
  source per topic and cross-link related material instead of duplicating it.
- Keep the README's status and document index current. Change these working rules
  when the user establishes or revises a rule or boundary.
- Cite biological and technical references where they inform design choices;
  distinguish measurements from estimates and retain attribution when reusing code.

## Implementation and verification

- Check library compatibility before selecting an implementation approach.
- Treat `../RL_lib/src/rl_lib` as the reusable library boundary. Reuse its public
  algorithms, data types, models, policies, normalization, and generic utilities;
  do not import application code from `../RL_lib/experiments`.
- Centipede owns its MuJoCo environment, eight independent learner instances,
  segment-to-learner mapping, rollout coordination, experiment configuration,
  runner, checkpoints, metrics, evaluation, and recording. Do not move this
  project-specific composition into RL_lib.
- Each segment has separate actor and critic networks, optimizers, normalization
  state, rollout samples, advantages, and losses. Never pool parameters, samples,
  gradients, losses, or critic inputs across agents. Shared environment rewards
  may be copied to each learner without combining their training data.
- If Centipede reveals a missing generic algorithm, data, model, or policy
  capability, discuss extending `src/rl_lib` rather than duplicating it here. A
  genuinely Centipede-specific extension may live here and must remain a thin,
  explicit specialization of the library component.
- Use focused checks and short smoke runs to validate changes. Substantial
  training and tuning runs require an explicit user request.
- Reserve `runs/` for reinforcement-learning training and evaluation outputs,
  following the role it has in `../RL_lib`. Model-development diagnostics and
  review renders belong in `archive/`, while physical model definitions stay in
  `models/`.
- Separate evaluation from learning. Inspect physical behavior as well as reward;
  do not present diagnostic scripts, appearance, or reward increases as proof of
  learned walking or a natural gait.
- Keep `tools/view_model.py` as a small static interactive inspection command;
  do not run the 10 kHz physics loop merely to view the model. Do not render
  ordinary training. Add offscreen policy recordings only after the environment
  and checkpoint interfaces exist, and record frozen evaluations.
- Prefer the native Windows environment at
  `C:\Users\danie\.venvs\Centipede` for interactive rendering because the user
  has unreliable GPU/OpenGL behavior through WSL. MuJoCo physics remains CPU
  work; do not claim native rendering accelerates the physics solver.
- Report what was actually tested, the results, and remaining limitations.
