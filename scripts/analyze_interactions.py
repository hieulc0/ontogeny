"""Analyze recorded training interactions without loading neural checkpoints."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def analyze(path: Path) -> list[dict]:
    episodes, current = [], []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        current.append(item)
        if item["outcome"]["terminated"] or item["outcome"]["truncated"]:
            episodes.append(current)
            current = []
    if current or len(episodes) != 768:
        raise ValueError("incomplete original training record")
    result = []
    for index, seed in enumerate((42, 43, 44)):
        group = episodes[index * 256 : (index + 1) * 256]
        records = []
        for label, selected in [("early", group[:64]), ("late", group[-64:])]:
            entries = [e for episode in selected for e in episode]
            counts = Counter(e["action"] for e in entries)
            opportunities = [
                e
                for e in entries
                if any(
                    x == 0 and y == 0 and k == "resource" for x, y, k in e["observation"]["tiles"]
                )
            ]
            records.append(
                dict(
                    period=label,
                    interactions=len(entries),
                    actions=dict(counts),
                    resources=sum(e["outcome"]["collected"] for e in entries),
                    hazards=sum(e["outcome"]["hazard"] for e in entries),
                    resource_opportunities=len(opportunities),
                    resource_interactions=sum(e["action"] == 5 for e in opportunities),
                    positive_reward_fraction=statistics.mean(
                        e["outcome"]["energy_delta"] > 0 for e in entries
                    ),
                    mean_undiscounted_return=statistics.mean(
                        sum(e["outcome"]["energy_delta"] / 15 for e in ep) for ep in selected
                    ),
                )
            )
        result.append(
            dict(
                seed=seed,
                periods=records,
                exhaustion=sum(ep[-1]["outcome"]["terminated"] for ep in group),
            )
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(analyze(args.input), stream, indent=2)
