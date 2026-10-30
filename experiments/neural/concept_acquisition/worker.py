"""Fixed-capacity acquisition with matched operand coverage and fixed update endpoints."""

import argparse
import hashlib
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
        raise RuntimeError("one T4 and pinned NumPy required")
    import mini_gpt.data.comparison as data_module
    import mini_gpt.learning.state as state_module
    import mini_gpt.model.transformer as model_module
    from mini_gpt.checkpoint.identity import file_digest
    from mini_gpt.learning.optimizer import optimizer_membership
    from mini_gpt.learning.trainer import train_tokens
    from mini_gpt.model.config import ModelConfig

    for module in (data_module, state_module, model_module):
        verify_import(module, repo / "src/mini_gpt")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    costs = {"updates": 0, "training_tokens": 0, "evaluation_tokens": 0}
    declared = json.loads((Path(__file__).parent / "coverage-data.json").read_text())

    def dataset(split: str, degree: int):
        rows = data_module.regular_comparisons(
            split, values=tuple(range(32, 64)), degree=degree, max_degree=8, seed=771
        )
        digest = hashlib.sha256(
            json.dumps([(r.left, r.right, r.target) for r in rows]).encode()
        ).hexdigest()
        if digest != declared[f"{split}-{degree}"]["sha256"]:
            raise RuntimeError("data identity mismatch")
        return rows

    def budget() -> None:
        if time.monotonic() - started > 600:
            raise TimeoutError("600 second worker budget exhausted")
        if torch.cuda.max_memory_allocated() > 3 * 1024**3:
            raise RuntimeError("3 GiB allocation budget exceeded")
        if costs["evaluation_tokens"] > 200000 or costs["updates"] > 3072:
            raise RuntimeError("operation budget exceeded")

    def measure(owner, rows):
        budget()
        before = owner.snapshot()
        owner.model.eval()
        try:
            with torch.no_grad():
                tokens = torch.tensor([r.tokens("bytes") for r in rows], device="cuda")
                labels = torch.tensor([r.target for r in rows], device="cuda")
                logits, _ = owner.model(tokens)
                costs["evaluation_tokens"] += tokens.numel()
                return {
                    "correct": int((logits[:, -1].argmax(-1) == labels).sum()),
                    "count": len(rows),
                }
        finally:
            owner.restore(before)

    active = None
    try:
        for seed in (404, 505, 606):
            for degree in (2, 8):
                budget()
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                model = model_module.MiniGPT(
                    ModelConfig(
                        d_model=64, n_layers=2, n_heads=4, d_ff=256, context_length=8, dropout=0.0
                    )
                ).cuda()
                owner = state_module.LearningState(model, lr=0.002)
                identity = f"{seed}-degree-{degree}"
                active = identity
                owner.metadata["acquisition"] = {
                    "source": revision,
                    "protocol": "comparison-coverage-001",
                    "seed": seed,
                    "degree": degree,
                    "history": "fresh research student",
                    "dataset": declared[f"train-{degree}"],
                }
                owner.external["sampler"] = random.Random(444)
                owner.external["operand_counts"] = {v: 0 for v in range(32, 64)}
                initial_path = args.output / f"{identity}-0.pt"
                owner.save(initial_path)
                checkpoints = {0: {"path": initial_path.name, "sha256": file_digest(initial_path)}}
                owner.metadata["parent_sha256"] = checkpoints[0]["sha256"]
                train, development = dataset("train", degree), dataset("development", degree)
                curves = [
                    {
                        "step": 0,
                        "development": measure(owner, development),
                        "training": measure(owner, train),
                    }
                ]
                for step in range(1, 513):
                    budget()
                    rows = owner.external["sampler"].sample(train, 16)
                    for row in rows:
                        owner.external["operand_counts"][row.left] += 1
                        owner.external["operand_counts"][row.right] += 1
                    tokens = torch.tensor([r.tokens("bytes") for r in rows], device="cuda")
                    targets = torch.full_like(tokens, -100)
                    targets[:, -1] = torch.tensor([r.target for r in rows], device="cuda")
                    prior_weight = model.lm_head.weight.detach().clone() if step == 1 else None
                    result = train_tokens(owner, tokens, targets, max_updates=512)
                    costs["updates"] += 1
                    costs["training_tokens"] += tokens.numel()
                    with (args.output / "updates.jsonl").open("a") as stream:
                        stream.write(json.dumps({"identity": identity, **result}) + "\n")
                    if result["gradient_norm"] <= 0 or any(
                        optimizer_membership(owner.model, owner.optimizer).values()
                    ):
                        raise RuntimeError("inactive gradient or disconnected optimizer")
                    if step == 1 and torch.equal(prior_weight, model.lm_head.weight):
                        raise RuntimeError("first update did not change output weights")
                    if any(not torch.isfinite(p).all() for p in model.parameters()):
                        raise RuntimeError("nonfinite parameters")
                    if step in (32, 128, 256, 512):
                        curves.append(
                            {
                                "step": step,
                                "training": measure(owner, train),
                                "development": measure(owner, development),
                            }
                        )
                    if step in (128, 512):
                        path = args.output / f"{identity}-{step}.pt"
                        owner.save(path)
                        checkpoints[step] = {"path": path.name, "sha256": file_digest(path)}
                        owner.metadata["parent_sha256"] = checkpoints[step]["sha256"]
                # Every endpoint is frozen before any held-out test is evaluated.
                tests = {}
                for step, item in checkpoints.items():
                    path = args.output / item["path"]
                    if file_digest(path) != item["sha256"]:
                        raise RuntimeError("checkpoint identity mismatch")
                    restored = state_module.LearningState.load(path)
                    tests[step] = measure(restored, dataset("test", degree))
                    del restored
                record = {
                    "identity": identity,
                    "seed": seed,
                    "degree": degree,
                    "checkpoints": checkpoints,
                    "curve": curves,
                    "tests": tests,
                    "operand_counts": owner.external["operand_counts"],
                }
                records.append(record)
                (args.output / f"{identity}.json").write_text(json.dumps(record, indent=2))
                del owner
        budget()
        result = {
            "execution": "COMPLETE",
            "protocol": "comparison-coverage-001",
            "source": revision,
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
            json.dumps({"active": active, "records": records, "costs": costs}, indent=2)
        )
        raise


if __name__ == "__main__":
    main()
