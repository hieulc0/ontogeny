"""Summarize completed GPU measurements without loading neural checkpoints."""

import argparse
import json
import math
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.input / "original_diagnosis.json").read_text())
    summary = []
    for row in original:
        groups = {}
        for label in ("initial", "final"):
            entries = [e for e in row["endpoints"] if e["checkpoint"] == label]
            groups[label] = dict(
                mean_entropy=statistics.mean(e["entropy"] for e in entries),
                effective_actions=math.exp(statistics.mean(e["entropy"] for e in entries)),
                mean_probabilities=[
                    statistics.mean(e["probabilities"][i] for e in entries) for i in range(7)
                ],
                samples=len(entries),
            )
        summary.append(
            dict(
                seed=row["seed"],
                endpoints=groups,
                changed_parameter_tensors=sum(p["change_norm"] > 0 for p in row["parameters"]),
                parameter_tensors=len(row["parameters"]),
                missing_optimizer_states=[
                    p["name"] for p in row["parameters"] if not p["optimizer_state_present"]
                ],
            )
        )
    (args.output / "endpoint_summary.json").write_text(json.dumps(summary, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, task in zip(axes, ("immediate", "delayed"), strict=True):
        found = False
        for seed in (42, 43, 44):
            path = args.input / f"{task}-{seed}-training.jsonl"
            if not path.exists():
                continue
            found = True
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            x = list(range(63, len(rows)))
            axis.plot(
                x,
                [statistics.mean(r["success"] for r in rows[i - 63 : i + 1]) for i in x],
                label=str(seed),
            )
        axis.set(
            title=task if found else f"{task}: NOT ADMITTED",
            xlabel="Episode",
            ylabel="Success (64-episode mean)",
            ylim=(0, 1.05),
        )
        if found:
            axis.legend(title="Initialization")
    fig.savefig(args.output / "canary_learning.svg")
    fig.savefig(args.output / "canary_learning.png", dpi=160)


if __name__ == "__main__":
    main()
