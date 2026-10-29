import argparse
import json
import tomllib
from dataclasses import asdict
from pathlib import Path

from ontogeny.evaluation.baselines import HeuristicPolicy, RandomPolicy
from ontogeny.organism.cell import run_episode
from ontogeny.world.environment import CellWorld
from ontogeny.world.state import WORLD_VERSION, WorldConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Environment-only bounded baseline comparison")
    parser.add_argument("--config", type=Path, default=Path("configs/cellworld.toml"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runs/l0.json"))
    args = parser.parse_args()
    if not 1 <= args.seeds <= 100:
        parser.error("use 1..100 seeds for local environment checks")
    config = WorldConfig(**tomllib.loads(args.config.read_text()))
    result = {"world_version": WORLD_VERSION, "config": asdict(config), "episodes": []}
    for policy_type in (RandomPolicy, HeuristicPolicy):
        for seed in range(args.seeds):
            metrics = run_episode(CellWorld(config, seed=seed), policy_type(seed + 10000))
            result["episodes"].append(
                {"policy": policy_type.__name__, "seed": seed, **asdict(metrics)}
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
