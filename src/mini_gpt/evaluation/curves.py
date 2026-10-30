"""Paired adaptation curves; inputs are externally collected, not model state."""

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

ARMS = ("continuing", "unrelated", "fresh", "frozen")


def normalized_area(updates: Sequence[int], accuracies: Sequence[float]) -> float:
    """Trapezoidal mean accuracy over the explicitly declared update interval."""
    if len(updates) != len(accuracies) or len(updates) < 2:
        raise ValueError("curve needs matching points")
    if updates[0] != 0 or any(type(x) is not int or x < 0 for x in updates):
        raise ValueError("updates must begin at zero and be nonnegative integers")
    if any(a >= b for a, b in zip(updates, updates[1:], strict=False)):
        raise ValueError("updates must strictly increase")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in accuracies):
        raise ValueError("accuracies must be finite fractions")
    area = sum(
        (b - a) * (x + y) / 2
        for a, b, x, y in zip(updates, updates[1:], accuracies, accuracies[1:], strict=False)
    )
    return area / (updates[-1] - updates[0])


def paired_report(
    rows: Sequence[Mapping],
    updates: Sequence[int],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict:
    """Aggregate complete crossed seed/task pairs and bootstrap both dimensions."""
    if bootstrap_samples < 1:
        raise ValueError("positive bootstrap sample count required")
    curves: dict[tuple[int, str, str], float] = {}
    raw: dict[tuple[int, str, str], tuple[float, ...]] = {}
    for row in rows:
        model_seed, task, arm = row["model_seed"], row["task"], row["arm"]
        if type(model_seed) is not int or not isinstance(task, str) or not task or arm not in ARMS:
            raise ValueError("invalid seed, task, or arm")
        key = (model_seed, task, arm)
        if key in curves:
            raise ValueError("duplicate curve")
        values = tuple(row["accuracy"])
        curves[key] = normalized_area(updates, values)
        raw[key] = values
        if arm == "frozen" and any(value != values[0] for value in values):
            raise ValueError("frozen control changed across updates")
    seeds = sorted({key[0] for key in curves})
    task_ids = sorted({key[1] for key in curves})
    if len(seeds) < 2 or len(task_ids) < 2 or len(curves) != len(seeds) * len(task_ids) * len(ARMS):
        raise ValueError("a complete crossed cohort with multiple seeds and tasks is required")
    if any((s, t, a) not in curves for s in seeds for t in task_ids for a in ARMS):
        raise ValueError("missing paired curves")
    for s in seeds:
        for t in task_ids:
            if raw[(s, t, "continuing")][0] != raw[(s, t, "frozen")][0]:
                raise ValueError("frozen control must match continuing at zero updates")

    def statistic(sample_seeds: Sequence[int], sample_tasks: Sequence[str]) -> tuple[dict, float]:
        n = len(sample_seeds) * len(sample_tasks)
        means = {
            arm: sum(curves[(s, t, arm)] for s in sample_seeds for t in sample_tasks) / n
            for arm in ARMS
        }
        return means, means["continuing"] - max(means["unrelated"], means["fresh"])

    means, difference = statistic(seeds, task_ids)
    rng = random.Random(seed)
    samples = sorted(
        statistic(rng.choices(seeds, k=len(seeds)), rng.choices(task_ids, k=len(task_ids)))[1]
        for _ in range(bootstrap_samples)
    )
    return {
        "seed_count": len(seeds),
        "task_count": len(task_ids),
        "mean_aulc": means,
        "difference_vs_strongest_control": difference,
        "bootstrap_interval_95": [
            samples[int(0.025 * (len(samples) - 1))],
            samples[int(0.975 * (len(samples) - 1))],
        ],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def retention_report(rows: Sequence[Mapping]) -> dict:
    """Summarize separately measured prior-task competence before and after adaptation."""
    if not rows:
        raise ValueError("retention observations required")
    keys = set()
    before, after = [], []
    for row in rows:
        key = (row["model_seed"], row["task"])
        if key in keys:
            raise ValueError("duplicate retention observation")
        keys.add(key)
        for name, target in (("before", before), ("after", after)):
            value = row[name]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("retention accuracy must be a finite fraction")
            target.append(value)
    return {
        "count": len(rows),
        "mean_before": sum(before) / len(before),
        "mean_after": sum(after) / len(after),
        "mean_change": sum(a - b for b, a in zip(before, after, strict=True)) / len(rows),
        "observations": list(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    document = json.loads(args.input.read_text())
    result = paired_report(document["curves"], document["updates"], seed=args.bootstrap_seed)
    if "retention" in document:
        result["retention"] = retention_report(document["retention"])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
