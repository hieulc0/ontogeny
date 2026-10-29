"""Non-neural audit of remote records, coverage, accounting and continuation evidence."""

import argparse
import hashlib
import json
import math
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = tomllib.loads((args.input / "protocol.toml").read_text())
    runtime = json.loads((args.input / "runtime.json").read_text())
    assert runtime["status"] == "COMPLETE"
    assert runtime["torch"] == "2.6.0+cu124" and runtime["numpy"] == "2.1.3"
    assert runtime["upstream_sha"] == "8567836fb2c8c915d442b7c2fbedb4fd9b30396d"
    rows = [json.loads(s) for s in (args.input / "evaluation.jsonl").read_text().splitlines()]
    train = [json.loads(s) for s in (args.input / "training.jsonl").read_text().splitlines()]
    assert len(rows) == 2400 and len(train) == 768
    expected = {
        (seed, arm, split, variant, world)
        for seed in spec["model_seeds"]
        for arm in ("rest", "random", "heuristic", "frozen", "learning")
        for split, start, count in [
            ("dev", spec["dev_seed_start"], spec["dev_worlds"]),
            ("sealed", spec["sealed_seed_start"], spec["sealed_worlds"]),
        ]
        for variant in ("default", "hazard_shift")
        for world in range(start, start + count)
    }
    actual = {(r["model_seed"], r["arm"], r["split"], r["variant"], r["world_seed"]) for r in rows}
    assert actual == expected
    for row in rows + train:
        assert len(row["actions"]) == row["duration"]
        assert len(row["energy_trajectory"]) == row["duration"] + 1
        assert row["energy_trajectory"][0] == 40
        assert all(0 <= e <= 60 for e in row["energy_trajectory"])
        assert all(0 <= a < 7 for a in row["actions"])
        assert row["resource_efficiency"] == row["collected"] / row["duration"]
        if row["termination"] == "exhausted":
            assert row["energy_trajectory"][-1] == 0
        else:
            assert row["duration"] == 100
        expected_tiles = 9 + sum(25 if a == 4 else 9 for a in row["actions"][:-1])
        assert row["observation_tiles"] == expected_tiles
    for seed in spec["model_seeds"]:
        assert {r["episode"] for r in train if r["model_seed"] == seed} == set(range(256))
        gradients = json.loads((args.input / f"{seed}-gradients.json").read_text())
        assert len(gradients) == 256
        for i, grad in enumerate(gradients):
            assert grad["step"] == i + 1
            assert math.isfinite(grad["loss"]) and math.isfinite(grad["gradient_norm"])
            assert grad["backbone_gradient_squared"] > 0 and grad["head_gradient_squared"] > 0
        audit = args.input / f"{seed}-trained-continuation"
        assert json.loads((audit / "admission.json").read_text())["updates"] == 257
        assert all(json.loads((audit / "continuation-fields.json").read_text()).values())
    report = json.loads((args.input / "qualification.json").read_text())
    for c in report["checkpoints"]:
        for stage in ("initial", "final"):
            artifact = args.input / f"{c['seed']}-{stage}.pt"
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == c[stage]
        assert c["updates"] == 256
        assert c["gradient_examples"] == sum(
            r["duration"] for r in train if r["model_seed"] == c["seed"]
        )
    manifest = {
        str(p.relative_to(args.input)): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in args.input.rglob("*")
        if p.is_file()
    }
    result = {
        "status": "PASS",
        "evaluation_episodes": len(rows),
        "training_episodes": len(train),
        "training_interactions": sum(r["duration"] for r in train),
        "evaluation_interactions": sum(r["duration"] for r in rows),
        "evaluation_observation_tiles": sum(r["observation_tiles"] for r in rows),
        "training_observation_tiles": sum(r["observation_tiles"] for r in train),
        "artifact_manifest": manifest,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_manifest"}, indent=2))


if __name__ == "__main__":
    main()
