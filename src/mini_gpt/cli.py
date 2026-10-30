"""Independent, bounded neural training and explicit trusted checkpoint conversion."""

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

from mini_gpt.checkpoint.identity import file_digest
from mini_gpt.model.config import ModelConfig
from mini_gpt.runtime import configure_determinism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wall-seconds", type=int, default=120)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    if not 0 < args.updates <= 100000 or not 0 < args.wall_seconds <= 3600:
        parser.error("finite positive update/time envelope required")
    config = ModelConfig.from_dict(json.loads(args.config.read_text()))
    batches = json.loads(args.data.read_text())
    if not isinstance(batches, list) or not batches:
        parser.error("data must be a nonempty array of {tokens, targets} batches")
    torch, np = configure_determinism()
    repository = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip():
        raise RuntimeError("independent research requires a clean source revision")
    from mini_gpt.learning.state import LearningState
    from mini_gpt.learning.trainer import train_tokens
    from mini_gpt.model.transformer import MiniGPT

    args.output.mkdir(parents=True, exist_ok=False)
    envelope = {
        "model": config.to_dict(),
        "data_sha256": file_digest(args.data),
        "updates": args.updates,
    }
    if args.resume:
        if file_digest(args.resume) != args.checkpoint_sha256:
            raise ValueError("declared full checkpoint hash required")
        owner = LearningState.load(args.resume)
        if (
            owner.metadata.get("training_envelope") != envelope
            or owner.metadata.get("source_revision") != revision
        ):
            raise ValueError("resume cannot silently change the training envelope")
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        owner = LearningState(
            MiniGPT(config).cuda(),
            lr=0.0005,
            external={"batch_cursor": 0},
            metadata={
                "individual_kind": "research",
                "source_revision": revision,
                "architecture": "dense-causal-transformer",
                "training_envelope": envelope,
            },
        )
    deadline = time.monotonic() + args.wall_seconds
    while owner.metadata["steps"] < args.updates:
        if time.monotonic() >= deadline:
            raise TimeoutError("independent training deadline")
        cursor = owner.external["batch_cursor"]
        batch = batches[cursor % len(batches)]
        row = train_tokens(
            owner,
            torch.tensor(batch["tokens"], device="cuda"),
            torch.tensor(batch["targets"], device="cuda"),
            max_updates=args.updates,
        )
        owner.external["batch_cursor"] = cursor + 1
        owner.save(args.output / f"step-{owner.metadata['steps']}.pt")
        with (args.output / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")


def import_checkpoint() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--trusted", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_determinism()
    from mini_gpt.checkpoint.legacy import import_dense_checkpoint

    owner = import_dense_checkpoint(
        args.source,
        expected_sha256=args.sha256,
        source_revision=args.source_revision,
        trusted=args.trusted,
    )
    owner.save(args.output)


if __name__ == "__main__":
    main()
