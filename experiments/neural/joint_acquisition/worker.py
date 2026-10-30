"""Joint and sequential acquisition with exactly matched per-task feedback streams."""

import argparse
import json
import random
import subprocess
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if revision != args.source:
        raise RuntimeError("source mismatch")
    from mini_gpt.runtime import configure_determinism, verify_import

    torch, np = configure_determinism()
    if (
        np.__version__ != "2.1.3"
        or torch.cuda.device_count() != 1
        or "T4" not in torch.cuda.get_device_name()
    ):
        raise RuntimeError("one T4 and pinned runtime required")
    import mini_gpt.learning.state as state_module
    import mini_gpt.model.transformer as model_module
    from experiments.neural.joint_acquisition import design
    from mini_gpt.checkpoint.identity import file_digest
    from mini_gpt.learning.optimizer import optimizer_membership
    from mini_gpt.learning.trainer import train_tokens
    from mini_gpt.model.config import ModelConfig

    for module in (state_module, model_module):
        verify_import(module, repo / "src/mini_gpt")
    verify_import(design, repo / "experiments/neural/joint_acquisition")
    args.output.mkdir(parents=True, exist_ok=False)
    declared = json.loads((Path(__file__).parent / "data.json").read_text())
    started = time.monotonic()
    costs = {"updates": 0, "training_tokens": 0, "evaluation_tokens": 0}
    parents = []
    records = []
    active = None

    def dataset(task, split):
        rows = design.examples(task, split)
        if design.digest(rows) != declared[f"{task}-{split}"]["sha256"]:
            raise RuntimeError("data identity mismatch")
        return rows

    def budget():
        if time.monotonic() - started > 600:
            raise TimeoutError("worker time exhausted")
        if (
            torch.cuda.max_memory_allocated() > 3 * 1024**3
            or costs["updates"] > 9216
            or costs["evaluation_tokens"] > 350000
        ):
            raise RuntimeError("resource budget exceeded")

    def measure(owner, task, split="development"):
        budget()
        rows = dataset(task, split)
        before = owner.snapshot()
        owner.model.eval()
        try:
            with torch.no_grad():
                tokens = torch.tensor([r[0] for r in rows], device="cuda")
                labels = torch.tensor([r[1] for r in rows], device="cuda")
                logits, _ = owner.model(tokens)
                costs["evaluation_tokens"] += tokens.numel()
                return {
                    "correct": int((logits[:, -1].argmax(-1) == labels).sum()),
                    "count": len(rows),
                }
        finally:
            owner.restore(before)

    def save(owner, name):
        path = args.output / f"{name}.pt"
        owner.save(path)
        item = {"path": path.name, "sha256": file_digest(path)}
        owner.metadata["parent_sha256"] = item["sha256"]
        return item

    def load(item):
        path = args.output / item["path"]
        if file_digest(path) != item["sha256"]:
            raise RuntimeError("parent hash mismatch")
        return state_module.LearningState.load(path)

    def update(owner, rows, identity):
        budget()
        tokens = torch.tensor([r[0] for r in rows], device="cuda")
        targets = torch.full_like(tokens, -100)
        targets[:, -1] = torch.tensor([r[1] for r in rows], device="cuda")
        result = train_tokens(owner, tokens, targets, max_updates=1024)
        costs["updates"] += 1
        costs["training_tokens"] += tokens.numel()
        with (args.output / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps({"identity": identity, **result}) + "\n")
        if result["gradient_norm"] <= 0 or any(
            optimizer_membership(owner.model, owner.optimizer).values()
        ):
            raise RuntimeError("invalid gradient/ownership")
        if any(not torch.isfinite(p).all() for p in owner.model.parameters()):
            raise RuntimeError("nonfinite weights")

    try:
        for seed in design.SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            owner = state_module.LearningState(
                model_module.MiniGPT(
                    ModelConfig(
                        d_model=64, n_layers=2, n_heads=4, d_ff=256, context_length=8, dropout=0.0
                    )
                ).cuda(),
                lr=0.002,
            )
            owner.metadata["study"] = {
                "source": revision,
                "protocol": design.PROTOCOL,
                "seed": seed,
                "history": "fresh paired initialization",
                "data": declared,
            }
            streams = {task: design.feedback_stream(task) for task in ("A", "B")}
            for task, rows in streams.items():
                if design.digest(rows) != declared[f"{task}-stream"]["sha256"]:
                    raise RuntimeError("stream hash mismatch")
            owner.external.update(streams=streams, cursors={"A": 0, "B": 0})
            initial = save(owner, f"{seed}-initial")
            parents.append({"seed": seed, "checkpoint": initial})
            for arm in design.ARMS:
                active = f"{seed}-{arm}"
                owner = load(initial)
                if owner.metadata["steps"] != 0:
                    raise RuntimeError("initial state mismatch")
                owner.metadata["parent_sha256"] = initial["sha256"]
                owner.metadata["arm"] = arm
                curves = [{"step": 0, "A": measure(owner, "A"), "B": measure(owner, "B")}]
                endpoints = {}
                for step in range(1, 1025):
                    batch = []
                    for task, count in zip(("A", "B"), design.allocation(arm, step), strict=True):
                        start = owner.external["cursors"][task]
                        batch.extend(owner.external["streams"][task][start : start + count])
                        owner.external["cursors"][task] += count
                    if len(batch) != 16:
                        raise RuntimeError("feedback stream exhausted")
                    update(owner, batch, active)
                    if step in (512, 1024):
                        endpoints[step] = save(owner, f"{active}-{step}")
                        curves.append(
                            {"step": step, "A": measure(owner, "A"), "B": measure(owner, "B")}
                        )
                if owner.external["cursors"] != {"A": 8192, "B": 8192}:
                    raise RuntimeError("unmatched exposure")
                records.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "initial": initial,
                        "checkpoints": endpoints,
                        "curve": curves,
                        "training": {task: measure(owner, task, "train") for task in ("A", "B")},
                        "cursors": dict(owner.external["cursors"]),
                    }
                )
        # No training is permitted after test outcomes are exposed.
        for parent in parents:
            owner = load(parent["checkpoint"])
            parent["test"] = {task: measure(owner, task, "test") for task in ("A", "B")}
        for record in records:
            owner = load(record["checkpoints"][1024])
            record["test"] = {task: measure(owner, task, "test") for task in ("A", "B")}
            if record["arm"] == "sequential":
                owner = load(record["checkpoints"][512])
                record["before_B"] = {task: measure(owner, task, "test") for task in ("A", "B")}
        budget()
        result = {
            "execution": "COMPLETE",
            "protocol": design.PROTOCOL,
            "source": revision,
            "parents": parents,
            "records": records,
            "costs": costs,
            "wall_seconds": time.monotonic() - started,
            "peak_torch_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": torch.cuda.get_device_name(),
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
    except BaseException:
        (args.output / "failure.txt").write_text(traceback.format_exc())
        (args.output / "partial.json").write_text(
            json.dumps(
                {"active": active, "parents": parents, "records": records, "costs": costs}, indent=2
            )
        )
        raise


if __name__ == "__main__":
    main()
