# Working rules for Codex

## Read first

- Read this file before making changes. Use `README.md` for the project goals,
  current status, folder layout, and documentation index.
- Read the relevant design documents before working on their areas:
  `docs/plan.md`, `docs/model.md`, `docs/control.md`, and
  `docs/environment.md`.
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

## Authorship and university review

- The user owns and writes the core reinforcement-learning implementation. Focus
  on discussing designs, explaining tradeoffs, reviewing the user's work, and
  helping with small, explicitly requested edits. Do not implement substantial
  environment, training, agent, or experiment components unless the user clearly
  asks for that exception.
- The completed MuJoCo model is an accepted exception because physical-model
  construction is supporting work rather than the academic focus of the project.
  Do not use that exception as precedent for generating the learning system.
- Treat current design documents as working references. The user plans to rewrite
  the final university-facing documentation in their own voice. Help organize,
  check, and refine it without silently replacing the user's authorship.
- Documentation intended for the repository must be complete, readable without
  conversation history, visually orderly, and written in natural human prose.
  Prefer a few clearly owned documents over fragmented notes, and keep accepted
  decisions distinct from proposals and unresolved questions.
- Minor Codex-created scripts, drafts, or diagnostics are temporary unless the
  user accepts them as project deliverables. After they serve their purpose, move
  meaningful ones to the ignored local `archive/` and remove disposable ones
  during the pre-commit cleanup review.

## Decisions and documentation

- Work incrementally within the user's current request. Do not implement future
  stages simply because they appear in a plan or the project goals.
- Treat `models/assembly.xml` as the frozen v1 physical baseline described in
  `docs/model.md`. Do not change it during environment or learning work. A
  necessary physical change must become a separately named version with new
  validation artifacts and explicit user agreement.
- Let experiment configuration select the accepted model file. The simulation
  loads that file once during construction and derives and caches its model
  dimensions, controlled segment IDs, and named mappings from the compiled
  MuJoCo model. Keep eight segments as the v1 configuration, not an immovable
  assumption inside the reusable simulation class.
- When an accepted model version changes a named joint, actuator, body, geometry,
  or site used by the environment, update its semantic mapping, documentation,
  and schema tests in the same reviewed change. Never let environment mappings
  silently refer to an earlier XML contract.
- For model revisions, review a small visual/geometry prototype with the user
  before adopting the shape and doing broader physics tuning. Passing mechanical
  checks alone does not settle appearance or biological resemblance.
- Treat `docs/control.md` and `docs/environment.md` as the authoritative sources
  for accepted first-version interfaces and unresolved choices. Do not copy their
  detailed constants into this file or implement later variants early.
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
- Base the simulation layer on Gymnasium's `MujocoEnv` and expose simultaneous
  segment interaction through PettingZoo's `ParallelEnv`. Use Ant as a reference,
  but do not subclass `AntEnv`: its single-agent observation, reward, and health
  semantics do not match this project.
- Compose one public `ParallelEnv` around one small internal `MujocoEnv`; do not
  combine the framework classes through multiple inheritance. Let `MujocoEnv`
  own XML loading, model/data state, stepping, frame skipping, rendering, and
  cleanup. Keep targets, partial observations, per-agent rewards, and task episode
  rules in the parallel environment or its pure helpers.
- Use Gymnasium `Box` spaces for every segment's action and observation space.
  Validate the public task with PettingZoo's parallel API test and the internal
  simulation with focused physics tests. Do not invent a duplicate single-agent
  task contract merely to run Gymnasium's full environment checker. MaMuJoCo is
  a composition reference, not an initial dependency.
- Implement identifiers, spaces, action ordering, actuator mapping, and control
  timing from the accepted contract in `docs/control.md`; never infer the policy
  mapping from XML order or silently clip public actions.
- Validate each input or invariant once at its owning boundary, then pass a
  trusted internal representation downstream. Do not duplicate public action,
  XML mapping, configuration, observation, or reward checks across components.
  Fail at the owner instead of silently repairing invalid values later.
- Keep controlled fixtures, invalid-input cases, focused simulation checks, and
  PettingZoo API tests under `tests/`. Production modules must never import test
  or diagnostic helpers. The runner should trust the public environment contract,
  while the parallel environment trusts the initialized simulation mapping after
  its one-time schema validation.
- Follow the state and data ownership table in `docs/plan.md`. In particular,
  keep episode rules in the environment, PPO-window cutoffs in the coordinator,
  learner normalization in `RL_lib`, and configuration/run artifacts in the
  experiment application.
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
- Run focused automated tests to validate code changes. Never launch a training
  run or a full experiment/evaluation command on the user's behalf; provide the
  exact command and let the user execute it, then review the resulting output.
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
