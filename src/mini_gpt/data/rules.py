"""Seeded arithmetic rule tasks with disjoint feedback and query inputs."""

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuleTask:
    """The rule is evaluator-owned; the student's prompt contains no rule parameters."""

    family: str
    multiplier: int
    offset: int
    modulus: int
    support: tuple[int, ...]
    query: tuple[int, ...]

    @property
    def identity(self) -> str:
        return f"{self.modulus}:{self.multiplier}:{self.offset}"

    def examples(self, split: str) -> list[dict[str, str]]:
        if split not in ("support", "query"):
            raise ValueError("split must be support or query")
        inputs = self.support if split == "support" else self.query
        return [
            {
                "prompt": f"input={value}; output=",
                "target": str((self.multiplier * value + self.offset) % self.modulus),
            }
            for value in inputs
        ]


def tasks(family: str, seed: int) -> tuple[RuleTask, ...]:
    """A is experience, B is development, and C is reserved for future confirmation."""
    settings = {
        "experience": (11, range(1, 5)),
        "development": (11, range(5, 9)),
        "confirmation": (13, range(2, 6)),
    }
    if family not in settings:
        raise ValueError("unknown rule family")
    modulus, multipliers = settings[family]
    result = []
    for multiplier in multipliers:
        for offset in range(modulus):
            identity = f"{modulus}:{multiplier}:{offset}"
            task_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{identity}".encode()).digest(), "big"
            )
            values = list(range(modulus))
            random.Random(task_seed).shuffle(values)
            cutoff = modulus // 2
            result.append(
                RuleTask(
                    family,
                    multiplier,
                    offset,
                    modulus,
                    tuple(values[:cutoff]),
                    tuple(values[cutoff:]),
                )
            )
    return tuple(result)


def generate(output: Path, family: str, seed: int) -> dict:
    """Generate one family at a time, avoiding implicit materialization of confirmation data."""
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for task in tasks(family, seed):
        records.append(
            {
                "identity": task.identity,
                "rule": asdict(task),
                "support": task.examples("support"),
                "query": task.examples("query"),
            }
        )
    path = output / "tasks.jsonl"
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    manifest = {
        "schema": 1,
        "family": family,
        "seed": seed,
        "task_count": len(records),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=("experience", "development", "confirmation"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.family, args.seed), indent=2))


if __name__ == "__main__":
    main()
