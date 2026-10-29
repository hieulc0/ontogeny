"""Analyze completed artifacts only; never imports neural libraries or checkpoints.

Run: uv run --with matplotlib==3.10.1 python scripts/report_l1.py INPUT OUTPUT
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (args.input / "evaluation.jsonl").read_text().splitlines()]
    train = [json.loads(line) for line in (args.input / "training.jsonl").read_text().splitlines()]
    summary = []
    for variant in ("default", "hazard_shift"):
        for arm in ("rest", "random", "heuristic", "frozen", "learning"):
            group = [
                r
                for r in rows
                if r["split"] == "sealed" and r["variant"] == variant and r["arm"] == arm
            ]
            summary.append(
                {
                    "variant": variant,
                    "arm": arm,
                    "episodes": len(group),
                    **{
                        k: statistics.mean(r[k] for r in group)
                        for k in (
                            "duration",
                            "collected",
                            "hazard_exposures",
                            "resource_efficiency",
                            "observation_tiles",
                        )
                    },
                    "horizon_fraction": statistics.mean(
                        r["termination"] == "horizon" for r in group
                    ),
                }
            )
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2))
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed in (42, 43, 44):
        group = [r for r in train if r["model_seed"] == seed]
        for ax, metric in zip(axes, ("duration", "collected"), strict=True):
            window = 16
            x = list(range(window - 1, len(group)))
            y = [statistics.mean(r[metric] for r in group[i - window + 1 : i + 1]) for i in x]
            ax.plot(x, y, label=f"init {seed}")
            ax.set(xlabel="Training episode", ylabel=f"{metric} (16-episode mean)")
            ax.legend()
    fig.savefig(args.output / "training.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, variant in zip(axes, ("default", "hazard_shift"), strict=True):
        for arm in ("rest", "random", "heuristic", "frozen", "learning"):
            group = [
                r
                for r in rows
                if r["split"] == "sealed" and r["variant"] == variant and r["arm"] == arm
            ]
            # Episode-mean trajectories are conditional on remaining alive at each t.
            means = [
                statistics.mean(
                    r["energy_trajectory"][t] for r in group if len(r["energy_trajectory"]) > t
                )
                for t in range(101)
                if any(len(r["energy_trajectory"]) > t for r in group)
            ]
            ax.plot(range(len(means)), means, label=arm)
        ax.set(title=variant, xlabel="Step", ylabel="Energy (remaining episodes only)")
        ax.legend()
    fig.savefig(args.output / "energy.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
