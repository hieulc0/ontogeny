"""Generate a small synthetic curriculum with deterministic instance splits."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import TypedDict


class Example(TypedDict):
    id: str
    domain: str
    prompt: str
    target: str
    split: str


def examples() -> list[Example]:
    """Produce idealized examples, not a substitute for a sourced subject corpus."""
    result: list[Example] = []
    for a in range(20):
        for b in range(20):
            candidates = [
                ("math", f"Add {a} and {b}. Answer: ", str(a + b)),
                ("programming", f"Python integer expression {a} * {b} gives: ", str(a * b)),
                (
                    "physics",
                    f"At constant speed {a} m/s for {b} s, distance in meters: ",
                    str(a * b),
                ),
                ("chemistry", f"There are {a} groups of {b} atoms; total atoms: ", str(a * b)),
                (
                    "language",
                    f"Write the numbers {a}, then {b}, separated by a comma: ",
                    f"{a}, {b}",
                ),
            ]
            for domain, prompt, target in candidates:
                identity = hashlib.sha256(f"{domain}:{a}:{b}".encode()).hexdigest()
                bucket = int(identity[:8], 16) % 10
                split = "train" if bucket < 8 else "development" if bucket == 8 else "test"
                result.append(
                    {
                        "id": identity,
                        "domain": domain,
                        "prompt": prompt,
                        "target": target,
                        "split": split,
                    }
                )
    return result


def batch(example: Example) -> dict[str, list[list[int]]]:
    """Train only on the answer and newline; inputs are UTF-8 byte tokens."""
    prompt = example["prompt"].encode("utf-8")
    full = prompt + (example["target"] + "\n").encode("utf-8")
    return {
        "tokens": [list(full[:-1])],
        "targets": [[-100] * (len(prompt) - 1) + list(full[len(prompt) :])],
    }


def generate(output: Path, *, seed: int = 42) -> dict:
    """Write separate splits, training-only batches, and a content-hash manifest."""
    output.mkdir(parents=True, exist_ok=False)
    rows = examples()
    manifest: dict = {
        "schema": 1,
        "seed": seed,
        "encoding": "utf-8-bytes",
        "split_policy": "sha256 of domain and operands, modulo 10",
        "source": "procedurally generated idealized examples",
        "files": {},
        "counts": {},
    }
    for split in ("train", "development", "test"):
        selected = [row for row in rows if row["split"] == split]
        random.Random(seed).shuffle(selected)
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
        manifest["counts"][split] = len(selected)
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if split == "train":
            path = output / "train-batches.json"
            path.write_text(json.dumps([batch(row) for row in selected]) + "\n")
            manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
