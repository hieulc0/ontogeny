"""Audit logged policy-gradient arithmetic and complete canary coverage without tensors."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    completion = json.loads((args.input / "completion.json").read_text())
    if completion["status"] != "COMPLETE":
        raise ValueError("execution incomplete")
    probe = json.loads((args.input / "gradient_test.json").read_text())
    if not probe["after"] > probe["before"] or not all(probe["changed"].values()):
        raise ValueError("controlled gradient test failed")
    totals = dict(updates=1, training_interactions=0, evaluation_interactions=0)
    summaries = []
    for task, length in [("immediate", 1), ("delayed", 3)]:
        for seed in (42, 43, 44):
            path = args.input / f"{task}-{seed}-training.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            if len(rows) != 1024:
                raise ValueError("incomplete training records")
            baseline = [0.0] * 3
            max_error = 0.0
            for row in rows:
                if row["reward"] not in (-1, 1) or len(row["predictions"]) != length:
                    raise ValueError("invalid canary trajectory")
                expected_loss = 0.0
                for i, pred in enumerate(row["predictions"]):
                    expected = row["reward"] * 0.97 ** (length - i - 1)
                    if abs(row["returns"][i] - expected) > 1e-12:
                        raise ValueError("return arithmetic mismatch")
                    advantage = expected - baseline[i]
                    if abs(row["advantages"][i] - advantage) > 1e-12:
                        raise ValueError("advantage arithmetic mismatch")
                    probs = pred["probabilities"]
                    if len(probs) != 2 or abs(sum(probs) - 1) > 1e-6:
                        raise ValueError("invalid action distribution")
                    expected_loss -= (
                        math.log(max(probs[pred["action"]], 1e-30)) * advantage / length
                    )
                    baseline[i] = 0.9 * baseline[i] + 0.1 * expected
                max_error = max(max_error, abs(expected_loss - row["loss"]))
                if not math.isfinite(row["gradient_norm"]):
                    raise ValueError("nonfinite gradient")
            if max_error > 1e-4:
                raise ValueError(f"logged probability/loss mismatch: {max_error}")
            evaluation = json.loads((args.input / f"{task}-{seed}-evaluation.json").read_text())
            if len(evaluation) != 1536 or any(
                sum(r["arm"] == arm for r in evaluation) != 512
                for arm in ("random", "frozen", "learning")
            ):
                raise ValueError("incomplete evaluation coverage")
            totals["updates"] += 1024
            totals["training_interactions"] += 1024 * length
            totals["evaluation_interactions"] += 1536 * length
            summaries.append(
                dict(
                    task=task,
                    seed=seed,
                    max_loss_reconstruction_error=max_error,
                    initial_sha256=hashlib.sha256(
                        (args.input / f"{task}-{seed}-initial.pt").read_bytes()
                    ).hexdigest(),
                    final_sha256=hashlib.sha256(
                        (args.input / f"{task}-{seed}-final.pt").read_bytes()
                    ).hexdigest(),
                )
            )
    manifest = {
        p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in args.input.iterdir()
        if p.is_file()
    }
    with args.output.open("x") as stream:
        json.dump(
            dict(status="PASS", counts=totals, checks=summaries, artifacts=manifest),
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
