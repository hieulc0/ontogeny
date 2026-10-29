# Research protocol

Every experiment declares hypothesis, source revisions, initial checkpoint hashes,
world version/configuration, independent splits, seeds, controls, budget, fixed final
checkpoint, primary metric and stop gates before training. Preserve failed attempts.
No selection of favorable checkpoints after seeing development or sealed outcomes.

The current L.1 protocol is [L1_QUALIFICATION.md](L1_QUALIFICATION.md),
`l1-reinforce-v1`, frozen before any neural execution. It supersedes the unexecuted
immediate-consequence pilot without discarding its historical record (decision 0001,
EXPERIMENTS.md and initialization commit). Exact objectives, controls, larger seed
cohorts, budgets, continuation admission and H1/H2/H3 gates are specified there.
No retrospective threshold or objective changes are permitted.

Evaluation never calls learn. Freeze final checkpoint; isolate evaluation RNG/state and
restore it before later training. Environment seed lists belong to driver, not policy.
Sealed outcomes cannot drive architecture or hyperparameters; a changed design requires
new untouched seeds. Seed names alone do not prove isolation: audit generation and logs.

Stop on leakage, nonfinite updates, budget exhaustion, ownership mismatch, checkpoint
mismatch or a failed admission gate. Keep raw logs, partial checkpoints and tracebacks;
report FAIL/INCONCLUSIVE/NOT RUN per gate. Roll back rejected candidates to immutable
parents. Ablate updates, initialization/history and observation costs before attributing
benefit to development. L.2 additionally needs history-matched reset controls and prior
competence gates; no silent optimizer resets or replacement specialists.
