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
