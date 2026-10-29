# L.0 qualification — CellWorld v1

Verdict: **PASS, engineering only**. Executed 2026-10-29 locally.
Neural work: **NOT RUN**. LOCAL: environment/lint/type/syntax/mock-only.
REMOTE: Kaggle GPU training/neural evaluation required; no kernel submitted.

| Gate | Evidence | Verdict |
|---|---|---|
| Seeded deterministic replay | Seed and snapshot continuation across hazard changes | PASS |
| Energy accounting | Ten bounded seeded ledger episodes; clamping and hazard/rest costs | PASS |
| Valid actions | Enum-only validation before mutation; all movement/interaction kinds | PASS |
| Termination | Exhaustion, horizon and refusal after completion | PASS |
| Observation isolation | Immutable relative field whitelist, radius, no world passed to policy | PASS |
| Reproducible change | Seeded hazard relocation excluding resources and organism | PASS |
| Controls | Random and fixed heuristic reproducible with same seeds | PASS |

31 inspected tests passed. Ruff check/format and Pyright passed. Observation isolation
is an interface property, not a malicious-code sandbox. It does not prove absence of
all future reward exploits. Resource consumption, costly rest and finite episodes were
explicitly checked.

| Control | Episodes | Mean duration | Resources | Hazard exposures | Final energy |
|---|---:|---:|---:|---:|---:|
| Random | 10 | 39.3 | 0.6 | 0.9 | 0.0 |
| Fixed heuristic | 10 | 100.0 | 14.5 | 0.0 | 40.1 |

World seeds 0..9, policy RNG seed world_seed+10000, default config, horizon 100.
Raw rows/config: baselines.json. Heuristic reaches the horizon ceiling on these seeds;
a neural improvement gate may therefore be inconclusive in this initial world.
Do not revise gates after inspecting neural test outcomes.

Reproduce from the containing Git commit (git rev-parse HEAD) with README commands,
using a new output path. Python 3.12.13; dependencies pinned in uv.lock.
Initial setup/static failures and resolution remain in docs/EXPERIMENTS.md.
Next decision: admit remote L.1 persistence and evaluation protocol, not L.2.
