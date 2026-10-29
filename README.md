# Ontogeny

Developing intelligence through experience: one persistent neural learner encounters a
partially observed world, predicts consequences and learns from its actions.
This repository supplies artificial environments and developmental experiments;
Adaptive Mini-GPT remains the canonical neural foundation.

**Current status:** L.0 engineering qualified on the declared local checks. L.1 adapter
and remote pilot implemented, neural compatibility and learning qualification NOT RUN.
No L.2 or multicellular execution. No inherited intelligence or historical knowledge claimed.

Python 3.12, uv, pytest, Ruff and Pyright. The environment uses only the standard library.

```bash
UV_CACHE_DIR=/tmp/ontogeny-uv uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q tests/test_world.py tests/test_actions.py tests/test_observations.py tests/test_determinism.py tests/test_resource_invariants.py tests/test_neural_contracts.py tests/test_runner.py
uv run ontogeny --seeds 10 --output runs/l0-new.json
```

Output paths are exclusive: use a new name to preserve earlier runs. All listed tests
are local-safe environment or fake/encoding/guard contracts; none imports torch.
Read [the measured qualification](experiments/l0_cellworld/REPORT.md),
[vision](docs/VISION.md), [architecture](docs/ARCHITECTURE.md),
[protocol](docs/RESEARCH_PROTOCOL.md), [roadmap](docs/DEVELOPMENT_ROADMAP.md),
[integration findings](docs/NEURAL_INTEGRATION.md), [world specification](docs/CELLWORLD_SPEC.md)
and [append-only history](docs/EXPERIMENTS.md).

Remote pilot preparation (after admission gates in the integration document):

```bash
# Commit and push the exact executable source first.
uv run python kaggle/prepare.py --revision HEAD --job runs/kaggle-l1/job
kaggle kernels push -p runs/kaggle-l1/job
kaggle kernels status sparklinghydraulics0/ontogeny-l1-pilot
kaggle kernels output sparklinghydraulics0/ontogeny-l1-pilot -p runs/kaggle-l1/output
```

This prepares a private, pinned Kaggle CUDA pilot; preparation never submits it.
Run completion is not scientific promotion. Large logs/checkpoints stay under ignored runs/.
