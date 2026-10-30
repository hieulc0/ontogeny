"""Finite fixed-capacity acquisition diagnosis on independently initialized students."""

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
        raise RuntimeError("one T4 and pinned NumPy required")
    import mini_gpt.learning.state as state_module
    import mini_gpt.model.transformer as model_module
    from mini_gpt.checkpoint.identity import file_digest
    from mini_gpt.data.comparison import comparisons
    from mini_gpt.learning.optimizer import optimizer_membership
    from mini_gpt.learning.trainer import train_tokens
    from mini_gpt.model.config import ModelConfig

    for module in (state_module, model_module):
        verify_import(module, repo / "src/mini_gpt")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    costs = {"updates": 0, "training_tokens": 0, "evaluation_tokens": 0}

    def budget():
        if time.monotonic() - started > 600:
            raise TimeoutError("600 second worker budget exhausted")
        if torch.cuda.max_memory_allocated() > 3 * 1024**3:
            raise RuntimeError("3 GiB allocation budget exceeded")

    def measure(owner, rows, representation):
        budget()
        prior = owner.model.training
        owner.model.eval()
        try:
            with torch.no_grad():
                tokens = torch.tensor([r.tokens(representation) for r in rows], device="cuda")
                labels = torch.tensor([r.target for r in rows], device="cuda")
                logits, _ = owner.model(tokens)
                predictions = logits[:, -1].argmax(-1)
                costs["evaluation_tokens"] += tokens.numel()
                return {"correct": int((predictions == labels).sum()), "count": len(rows)}
        finally:
            owner.model.train(prior)

    try:
        for seed in (101, 202, 303):
            for representation in ("bytes", "symbols"):
                for pairs in (16, 64):
                    budget()
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    config = ModelConfig(
                        d_model=64, n_layers=2, n_heads=4, d_ff=256, context_length=8, dropout=0.0
                    )
                    model = model_module.MiniGPT(config).cuda()
                    owner = state_module.LearningState(model, lr=0.002)
                    identity = f"{seed}-{representation}-{pairs}"
                    owner.metadata["acquisition"] = {
                        "source": revision,
                        "seed": seed,
                        "representation": representation,
                        "pairs": pairs,
                        "history": "fresh research student",
                    }
                    owner.external["sampler"] = random.Random(444)
                    train = comparisons("train", pairs=pairs)
                    development = comparisons("development")
                    initial = owner.snapshot()
                    initial_path = args.output / f"{identity}-initial.pt"
                    owner.save(initial_path)
                    owner.metadata["parent_sha256"] = file_digest(initial_path)
                    before = measure(owner, development, representation)
                    curve = [{"step": 0, "development": before}]
                    logs = []
                    for step in range(1, 129):
                        budget()
                        rows = owner.external["sampler"].sample(train, 16)
                        tokens = torch.tensor(
                            [r.tokens(representation) for r in rows], device="cuda"
                        )
                        targets = torch.full_like(tokens, -100)
                        targets[:, -1] = torch.tensor([r.target for r in rows], device="cuda")
                        prior_weight = model.lm_head.weight.detach().clone() if step == 1 else None
                        result = train_tokens(owner, tokens, targets, max_updates=128)
                        if step == 1 and torch.equal(prior_weight, model.lm_head.weight):
                            raise RuntimeError("first update did not change output weights")
                        if result["gradient_norm"] <= 0 or any(
                            optimizer_membership(owner.model, owner.optimizer).values()
                        ):
                            raise RuntimeError("inactive gradient or disconnected optimizer")
                        costs["updates"] += 1
                        costs["training_tokens"] += tokens.numel()
                        logs.append(result)
                        if step in (32, 128):
                            curve.append(
                                {
                                    "step": step,
                                    "development": measure(owner, development, representation),
                                    "training": measure(owner, train, representation),
                                }
                            )
                    owner.metadata["acquisition"]["costs"] = {"updates": 128, "examples": 2048}
                    checkpoint = args.output / f"{identity}.pt"
                    owner.save(checkpoint)
                    # Fixed final state is saved before independent final examples are materialized.
                    final = measure(owner, comparisons("test"), representation)
                    owner.restore(initial)
                    frozen = measure(owner, comparisons("test"), representation)
                    record = {
                        "identity": identity,
                        "seed": seed,
                        "representation": representation,
                        "pairs": pairs,
                        "curve": curve,
                        "test": final,
                        "frozen_test": frozen,
                        "initial_sha256": file_digest(initial_path),
                        "updates": logs,
                        "checkpoint_sha256": file_digest(checkpoint),
                    }
                    records.append(record)
                    (args.output / f"{identity}.json").write_text(json.dumps(record, indent=2))
                    del owner
        result = {
            "execution": "COMPLETE",
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
            json.dumps({"records": records, "costs": costs}, indent=2)
        )
        raise


if __name__ == "__main__":
    main()
