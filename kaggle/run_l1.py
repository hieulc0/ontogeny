"""Remote-only pilot; engineering admission, not automatic scientific promotion."""

import argparse
import importlib
import json
import os
import random
import sys
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

from ontogeny.neural.adapter import ConsequenceCell, require_remote
from ontogeny.neural.persistence import UPSTREAM_SHA, digest, verify_import, verify_source
from ontogeny.organism.cell import run_episode
from ontogeny.world.environment import CellWorld


def main() -> None:
    require_remote()
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_source(args.upstream)
    sys.path.insert(0, str(args.upstream.resolve() / "src"))
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    ModelConfig = importlib.import_module("adaptive_mini_gpt.model.config").ModelConfig
    model_module = importlib.import_module("adaptive_mini_gpt.model.transformer")
    verify_import(args.upstream, model_module)
    MiniGPT = model_module.MiniGPT
    state_module = importlib.import_module("adaptive_mini_gpt.training.learning_state")
    verify_import(args.upstream, state_module)
    LearningState = state_module.LearningState

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle CUDA required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    spec = tomllib.loads(Path("configs/l1_experiment.toml").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = []
    for seed in spec["model_seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        owner = LearningState(
            MiniGPT(ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0)).cuda(),
            lr=0.001,
            metadata={
                "upstream_sha": UPSTREAM_SHA,
                "initialization": "fresh-no-historical-knowledge",
            },
        )
        initial = args.output / f"{seed}-initial.pt"
        owner.save(initial)
        for learning in (False, True):
            # Matched initial state per arm, never reset between episodes within an arm.
            current = LearningState.load(initial)
            current.metadata["parent_sha256"] = digest(initial)
            cell = ConsequenceCell(
                current, seed=seed, max_updates=spec["max_updates"], learning=learning
            )
            arm = "learning" if learning else "frozen"
            with (args.output / f"{seed}-{arm}-interactions.jsonl").open("x") as stream:

                def record(experience):
                    stream.write(json.dumps(asdict(experience)) + "\n")
                    stream.flush()

                for world_seed in spec["train_seeds"]:
                    if time.monotonic() - started > spec["max_seconds"]:
                        raise RuntimeError("time budget exhausted; partial artifacts preserved")
                    world = CellWorld(seed=world_seed)
                    metrics = run_episode(world, cell, learning=learning, record=record)
                    current.external["world"] = world.snapshot()
                    rows.append(
                        {
                            "model_seed": seed,
                            "arm": arm,
                            "split": "train",
                            "world_seed": world_seed,
                            **asdict(metrics),
                        }
                    )
            final = args.output / f"{seed}-{arm}-final.pt"
            current.save(final)
            # Fixed final checkpoint, no favorable-checkpoint selection.
            for world_seed in spec["dev_seeds"]:
                metrics = run_episode(CellWorld(seed=world_seed), cell, learning=False)
                rows.append(
                    {
                        "model_seed": seed,
                        "arm": arm,
                        "split": "dev",
                        "world_seed": world_seed,
                        **asdict(metrics),
                    }
                )
    report = {
        "status": "PILOT_ONLY_NOT_QUALIFIED",
        "rows": rows,
        "upstream_sha": UPSTREAM_SHA,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "seconds": time.monotonic() - started,
        "remaining_gates": [
            "fresh-process next-update equivalence",
            "independent evaluator and paired controls",
            "sealed variations",
            "resource accounting audit",
        ],
    }
    (args.output / "pilot.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
