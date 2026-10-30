# Ontogeny

Ontogeny investigates how one persistent neural learner can acquire knowledge through
experience, retain it and improve its future learning. **Mini-GPT** is its canonical
neural engine; **CellWorld** is the first embodied environment.

Two paths share one implementation:

- Independent neural research: representations, concepts, reasoning, retention and
  learning from finite feedback, without requiring an artificial world.
- Organism learning: partial observation → neural action → environmental consequence →
  persistent update. Research students and the continuing organism own separate states.

mini_gpt owns the dense Transformer, attention/positions, optimizer and LearningState.
ontogeny owns perception encoding, action head/policy objective, world, experience driver
and behavioral evaluation. The sibling repository is no longer a runtime dependency.
See [architecture](docs/ARCHITECTURE.md), [neural learning](docs/NEURAL_LEARNING.md) and
[migration provenance/verification](docs/MIGRATION.md).

**Scientific status is unchanged:** A.13 entity-binding qualification failed. B.0 preserved
one learning state but did not qualify strong initial competence, positive transfer or
benefit from a learned practice policy. Ontogeny L.0 engineering passed; L.1 acquisition/
held-out improvement failed while full continuation passed. Immediate canary passed,
delayed canary failed. No L.2, structural growth or additional cell is admitted.
Engineering consolidation is a separate gate, not a learning claim.

## Development

Python >=3.12, uv, pytest, Ruff and Pyright; Arch Linux is the primary workstation.
The world and configuration contracts require no neural dependencies. For complete static
analysis, install the optional neural dependencies (installation permits no local training):

```bash
UV_CACHE_DIR=/tmp/ontogeny-uv uv sync --locked --extra neural
uv run --extra neural ruff check .
uv run --extra neural ruff format --check .
uv run --extra neural pyright
# Inspected local-safe suites; do not run real model tests locally.
uv run --extra neural pytest -q tests/test_world.py tests/test_actions.py tests/test_observations.py tests/test_determinism.py tests/test_resource_invariants.py tests/test_neural_contracts.py tests/test_runner.py tests/mini_gpt/test_config.py tests/integration/test_unified_contracts.py
uv run ontogeny --seeds 10 --output runs/new-world-smoke.json
```

Both src packages ship in one wheel. Importing their root/configuration modules does
not import Torch. Real neural construction, forward/backward and checkpoint loading
require remote CUDA. No CPU fallback. See [AGENTS](AGENTS.md) and
[conventions](CONTRIBUTING.md).

## Remote execution

Commit/push the exact source, freeze the request/configuration and admit a finite budget.
Active Kaggle/Colab transports clone only Ontogeny, pin Torch2.6.0+cu124/NumPy2.1.3, verify
source and CUDA, and recover immutable hash-verified results. Provider choice stays outside
the policy. Auto-routing is Colab-readiness-first with quota-aware Kaggle fallback;
checkpoint continuation still requires an explicit provider.

```bash
ontogeny-exec qualification-request --source-sha FULL_PUBLISHED_SHA --protocol docs/MIGRATION.md --output runs/request.json
ontogeny-exec run --request runs/request.json --inputs empty-inputs/ --directory runs/new-job --ledger runs/controller --provider auto
# Independent neural learning, inside a validated CUDA worker only:
mini-gpt-train --config model.json --data batches.json --updates 2 --output independent-run/
```

The qualification-request helper is an engineering fixture. Generic scientific studies
need explicit preregistration and independent evaluation; a smoke result cannot promote
L.1. New request schema2 identifies the internal engine. Old schema1 requests remain
readable for artifact audits but execute only at their historical source revision.

Trusted dense historical checkpoints have an explicit one-time remote conversion tool;
unsupported architectures remain loadable with their original repository/source. No
weights, optimizer state or continuing history are reset silently. Exact support and
commands: [migration](docs/MIGRATION.md).

The earlier provider study qualified three origin/destination fixture pairs; reverse
Kaggle→Colab was blocked. Those results belong to their original source/runtime, not an
automatic portability claim for a new package version. [Checkpoint report](docs/CHECKPOINT_QUALIFICATION.md).

Read [vision](docs/VISION.md), [roadmap](docs/DEVELOPMENT_ROADMAP.md),
[research protocol](docs/RESEARCH_PROTOCOL.md), [append-only history](docs/EXPERIMENTS.md),
[L.1 results](docs/L1_QUALIFICATION.md), [diagnosis](docs/LEARNING_DIAGNOSIS.md) and
[defect register](docs/RESEARCH_DEFECTS.md). Historical runners remain in Git at the
revisions recorded in those reports; their scientific artifacts are preserved.
