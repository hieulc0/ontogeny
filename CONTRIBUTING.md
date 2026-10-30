# Development conventions

Write code around the behavior it implements. Keep experimental history in research
records and express the current experiment through explicit configuration.
These conventions apply to new and changed code; they do not authorize rewriting
historical evidence or breaking existing checkpoint compatibility.

## Names and organization

- Name modules, classes and functions for domain concepts or algorithms:
  `policy_gradient`, `checkpoint`, `evaluation`, `paired_interval`, `run_episode`.
- Do not name new reusable code after a milestone, phase, date or revision:
  avoid `l1.py`, `phase_two`, `PolicyV2`, `new_adapter`, `final_runner`.
- Use `snake_case` for modules/functions/variables, `PascalCase` for types and
  `UPPER_SNAKE_CASE` for constants. Use clear verbs for operations and nouns for data.
- Prefer descriptive names over generic buckets such as `utils`, `helpers`, `misc`,
  or unexplained abbreviations. Conventional mathematical notation is appropriate
  within a small, documented mathematical calculation.
- Put reusable behavior in `src/ontogeny`. Scripts parse arguments and call package
  interfaces; other scripts must not depend on their internal functions.
- Keep trial-specific seeds, gates, cohorts, artifact identifiers and recovery logic
  with the experiment configuration/driver. Do not add them to general algorithms.
- Keep existing public imports and serialized class paths stable until a deliberate
  compatibility change has been tested. A naming cleanup can break pickle checkpoints.

Milestone labels are appropriate in roadmaps, preregistrations, experiment records,
immutable artifact identifiers and references to historical evidence. Package versions,
checkpoint schema versions, environment definitions, dependency pins and Git revisions
are necessary compatibility/provenance information. Retain them. Do not add cosmetic
version suffixes to implementation names as a substitute for Git history.

## Interfaces and state

- Prefer small explicit functions and cohesive modules. Add an abstraction when it
  expresses an actual boundary or removes demonstrated duplication.
- Use dataclasses for structured records and `Protocol` for replaceable behavior.
  Use `TypedDict` for structured mapping contracts when serialization requires mappings.
- Annotate public interfaces, state records and function return values. Keep `Any`
  contained at genuinely dynamic upstream/serialization boundaries; validate those
  boundaries and give internal callers a typed interface.
- Make state ownership explicit. Avoid mutable global state, surprising import-time
  execution and implicit parameter/optimizer/RNG resets.
- Pass experiment settings explicitly. Validate configuration once and use it;
  do not duplicate settings in TOML and hidden numeric literals.
- Specify units, ordering, dimensions and bounds where they affect correctness.
  Never silently clip observations, discard incomplete outcomes or substitute defaults
  for corrupt or incompatible state.

## Python style and dependencies

Ruff owns formatting and import ordering; use the existing 100-character line length.
Pyright checks types. Prefer normal imports. Lazy imports are justified for optional
GPU dependencies or enforcing the remote execution boundary; isolate them in the
integration/runtime boundary rather than scattering dynamic imports throughout code.
Avoid `__import__` tricks and broad type/lint suppressions.

Keep dependencies small and purposeful. Local tooling remains separate from remote
neural dependencies. Keep dependency locks/pins and actual runtime metadata sufficient
to reproduce experiments. Never import or evaluate neural checkpoints locally to make
an inspection, test or style cleanup more convenient.

## Comments and docstrings

- Explain a non-obvious reason, invariant, unit, scientific assumption or compatibility
  constraint. Do not narrate obvious statements.
- Do not add comments or docstrings such as `L1 implementation`, `Phase 2`, `v3 fix`,
  `new improved version`, or a running account of edits. Put that history in Git and
  experiment records.
- Document public contracts and surprising behavior. Do not require boilerplate
  docstrings on every trivial private helper.
- Describe measured capabilities accurately. Avoid promotional claims and declarations
  of qualification inside ordinary implementation comments.
- A historical reference is justified when needed to explain an active compatibility
  constraint; identify the constraint and link its evidence rather than repeating history.

For example:

```python
# Poor: Phase 2 / L1 v2 policy update.
# Useful: Update the baseline after the gradient step to exclude the current return.
```

## Errors and validation

Use explicit exceptions for runtime safety, budgets, provenance, file formats and
scientific admission checks. Assertions may disappear under optimization; use them
in tests and, sparingly, for programmer-only invariants, never as the sole runtime gate.
Include enough context to identify the bad value or artifact. Fail closed on unsupported
state. Catch exceptions only to recover deliberately or preserve evidence, then propagate
failure. Do not swallow exceptions, invent a result or fall back to local neural execution.

## Tests and changes

- Test observable behavior, invariants and boundary failures, not private implementation
  details or a copy of the algorithm. Add regression tests for actual bugs.
- Keep local-safe tests explicitly identifiable. Neural execution, checkpoint loads and
  continuation checks run on validated Kaggle or Colab CUDA. Inspect tests before running them locally.
- Keep refactoring separate from changes to objectives, world dynamics, data splits,
  budgets or qualification gates. A structural refactor still requires verification.
- Preserve pinned historical code/artifacts and their results. Never rename a past FAIL
  into a PASS or rewrite sealed outcomes to fit new interfaces.
- Use focused commits with behavior-oriented subjects, such as `fix: reject incomplete
  checkpoint manifests`. Experiment IDs may identify a research-record commit; they
  should not be the only explanation of an implementation change.

Run Ruff check/format, Pyright and the appropriate inspected tests for executable
changes. Documentation-only changes need link/content review, not a GPU experiment.
Report what changed, what was checked, scientific limitations and unexecuted checks.

## Agent instructions and skills

`AGENTS.md` contains mandatory repository rules and links to this guide. It is not an
experiment diary or a duplicate architecture document. Keep detailed protocols and
results in `docs/` and `experiments/`.

Use a skill only for a relevant, repeatable workflow. Do not create a skill for every
milestone or embed skill names, tool invocations or agent identity in product code.
Potential reusable workflows include remote experiment preparation and artifact auditing;
add a skill only after the workflow has a stable contract and clear validation steps.
Repository-specific style belongs here, not in a separate style skill.
