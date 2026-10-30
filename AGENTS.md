# Ontogeny operating contract

Read README.md, docs/ARCHITECTURE.md, docs/RESEARCH_PROTOCOL.md, docs/NEURAL_LEARNING.md
and the current experiment/migration record. Inspect Git state first; preserve other work.

## Packages and code

- mini_gpt is the single canonical neural engine. It must not import ontogeny or an
  external adaptive_mini_gpt runtime. Ontogeny owns worlds, policies/readouts, observations,
  action consequences, organism development and behavioral evaluation.
- Follow CONTRIBUTING.md. Name reusable code for behavior/domain/algorithm, never a
  milestone, phase, date or cosmetic version. Keep real schema versions and immutable
  research identifiers. Comments explain contracts/reasons, not edit/agent history.
- Keep interfaces explicit, typed and small. Contain dynamic types at real serialization
  and optional-runtime boundaries. No unused architecture placeholders or copied engines.
- Historical source/launchers are reproducible at recorded Git revisions. Never alter
  historical results, rename serialized classes casually or silently load replacement state.
- The sibling ../adaptive-mini-gpt is reference/archive only. Do not modify or delete it.
  The selective migration is authorized; provenance/attribution live in docs/MIGRATION.md.

## Execution boundary

LOCAL-SAFE: editing, dependency installation/packaging, lint/type/syntax checks, bounded
non-neural simulations and manually inspected mock/configuration/JSON/hash/import tests.
No local neural construction, forward/backward, real checkpoint load/evaluation or GPU
benchmark. Installing a framework for static analysis is not permission to execute it.

REMOTE-REQUIRED: all real neural/model/checkpoint/learning checks on validated Kaggle or
Colab CUDA. Enforce platform and CUDA admission, exact committed source/import identity,
pinned runtime, finite budgets and complete outputs. Never fall back to CPU. Commit/push
executable source before remote submission; report actual completion and raw failures.
Do not run unknown/full test suites locally. Tests marked remote skip before framework
imports outside an actual remote runtime; select inspected non-neural tests explicitly.

Prefer verified usable Colab T4 capacity for ordinary routing, then quota-admitted Kaggle.
A known assignment may warm up within a finite readiness budget; refusal means no new
allocation retry/polling. Unknown allocation/cleanup blocks duplicate jobs. Do not stop
unrelated sessions. Follow current provider protocols; migration qualification explicitly
uses bounded Kaggle. Never assume fixed free-provider availability or reset schedules.

## State and research integrity

- One persistent LearningState per continuing individual/arm, owning model, optimizer,
  scheduler, RNGs and declared memory/replay/sampler/driver history. Environment resets
  must not reset neural knowledge. Label fresh students and their distinct histories.
- Research students and developmental organisms share code, not mutable state. Independent
  research weights never automatically replace the organism. Consolidation needs a
  separate preregistered experiment and retained-competence evidence.
- Checkpoints are trusted same-code full states, immutable and parent-linked. Preserve
  gradients, optimizer moments, flags and RNGs. Explicit versioned conversion must reject
  unsupported topology, preserve the parent and verify next-update behavior remotely.
- Give a policy only immutable partial observations. Simulator snapshots/seed/oracle data
  are driver/evaluation state. Heuristic controls are never hidden training assistance.
- Preregister hypotheses, source/checkpoint identity, splits, controls, final checkpoints,
  resource limits and stop gates. No test-driven checkpoint selection or tuning on consumed
  sealed worlds. Preserve negative evidence and distinguish FAIL/NOT RUN/INCONCLUSIVE.
- Engineering checks do not establish learning or intelligence. Preserve A.13/B.0 and L.1
  limitations. No L.2, extra cells or structural growth without prior scientific gates.
- No permanent architecture ceilings; each experiment is finite, externally interruptible
  and reversible. No unrestricted real-world persistence or resource-acquisition goals.

Report LOCAL versus REMOTE checks, exact sources/native jobs, state compatibility,
observed outcomes, resource costs and blockers. Do not claim a queued/unrun job passed.
