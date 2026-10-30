# Ontogeny research execution rules

Read README.md, docs/RESEARCH_PROTOCOL.md, docs/NEURAL_INTEGRATION.md and the current
experiment record before changing behavior. Inspect Git state and preserve existing work.

LOCAL-SAFE: editing, packaging, lint/type/syntax checks, bounded environment-only
simulation and manually inspected non-neural tests. REMOTE-REQUIRED: every real neural
forward/backward, checkpoint load/evaluation and scientific neural experiment, on Kaggle
CUDA under ../adaptive-mini-gpt/AGENTS.md. No local CPU fallback. State this boundary in reports.

Never modify upstream silently. Pin its exact SHA and verify imported source identity;
read its actual APIs before integration. No copying Transformer implementations.
One persistent LearningState per continuing arm, including optimizer, RNG and external
experience. Episode resets must not reset neural knowledge. Fresh initialization must be
labeled and cannot assert continuity with historical students.

Pass only immutable observations to a policy. Full simulator snapshots are privileged
checkpoint/evaluation data. Heuristics are controls, never hidden neural assistance.
No real-world persistence, survival or resource acquisition. Stop externally at finite budgets.
No structural growth, L.2 or multicellular experiments before independent prior-stage gates.
Preserve failures, preregister revisions/splits and never pick a checkpoint on test results.
Engineering tests cannot promote a learning claim. See docs/PRINCIPLES.md.

# Code and contribution conventions

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for naming, interfaces, comments, errors,
tests and dependency conventions.

- Name reusable code for its behavior, domain or algorithm. Do not introduce milestone,
  phase or cosmetic version names such as `l1.py`, `phase_two` or `PolicyV2`.
- Keep milestone labels and trial history in research records. Preserve real package,
  schema/checkpoint versions, source pins and immutable artifact identities.
- Comments explain reasons, invariants and compatibility constraints; do not narrate
  milestones, phases, skill usage, agent activity or a chronological edit history.
- Keep reusable mechanisms separate from experiment-specific seeds, gates and recovery
  scripts. Scripts call package APIs rather than importing other scripts' internals.
- Use explicit typed interfaces and validated configuration. Contain dynamic imports
  and `Any` at integration boundaries; avoid hidden constants and duplicated settings.
- Use explicit exceptions for runtime admission, budget and provenance checks. Do not
  rely on `assert` for checks that must survive `python -O`.
- Keep cleanup/refactoring separate from scientific behavior changes. Renaming serialized
  classes or modules requires a checkpoint compatibility plan and remote verification.
- Use skills for relevant repeatable workflows, not as milestone scaffolding or a second
  copy of these rules. Add no framework or abstraction without a demonstrated need.

These rules guide future changes; they do not authorize altering historical results,
checkpoint identities, preregistered experiments or the upstream repository.
