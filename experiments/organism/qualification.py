"""Independent non-neural aggregation; no policy selection or training access."""

import math
import statistics


def paired_interval(rows: list[dict], control: str, metric: str, variant: str) -> dict:
    """Student-t interval over paired model-seed means (3 independent inits).

    World seeds are repeated within initializations; treating episodes as independent
    model replicates would understate uncertainty. t(.975,2) is fixed in protocol.
    """
    pairs = {
        (r["model_seed"], r["world_seed"]): r[metric]
        for r in rows
        if r["arm"] == control and r["variant"] == variant
    }
    groups: dict[int, list[float]] = {}
    for r in rows:
        if r["arm"] == "learning" and r["variant"] == variant:
            groups.setdefault(r["model_seed"], []).append(
                r[metric] - pairs[r["model_seed"], r["world_seed"]]
            )
    means = [statistics.mean(v) for v in groups.values()]
    if len(means) != 3:
        raise ValueError("requires exactly three initialization replicates")
    mean = statistics.mean(means)
    half = 4.302652729911275 * statistics.stdev(means) / math.sqrt(3)
    return {
        "mean_difference": mean,
        "lower95": mean - half,
        "upper95": mean + half,
        "initialization_means": means,
    }


def qualification(
    rows: list[dict], training: list[dict], changed: list[bool], continuation: bool
) -> dict:
    comparisons = {}
    h2 = True
    for variant in ("default", "hazard_shift"):
        for control in ("random", "frozen"):
            for metric in ("duration", "collected", "resource_efficiency"):
                key = f"{variant}/{control}/{metric}"
                comparisons[key] = paired_interval(rows, control, metric, variant)
                h2 = h2 and comparisons[key]["lower95"] > 0
    improvements = []
    for seed in (42, 43, 44):
        values = [r for r in training if r["model_seed"] == seed]
        n = max(1, len(values) // 4)
        improvements.append(
            {
                k: statistics.mean(r[k] for r in values[-n:])
                - statistics.mean(r[k] for r in values[:n])
                for k in ("duration", "collected")
            }
        )
    h1 = all(changed) and all(v["duration"] > 0 and v["collected"] > 0 for v in improvements)
    return {
        "H1": "PASS" if h1 else "FAIL",
        "H2": "PASS" if h2 else "FAIL",
        "H3": "PASS" if continuation else "FAIL",
        "recommendation": "PASS" if h1 and h2 and continuation else "FAIL",
        "training_first_last_quarter_differences": improvements,
        "paired_comparisons": comparisons,
    }
