"""Independent, bounded neural training using the internal canonical engine."""

import argparse
import json
import random
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
    parser.add_argument("--max-torch-memory-mib", type=int, default=3072)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    if (
        not 0 < args.updates <= 100000
        or not 0 < args.wall_seconds <= 3600
        or args.max_torch_memory_mib <= 0
    ):
        parser.error("finite positive update/time envelope required")
    config = ModelConfig.from_dict(json.loads(args.config.read_text()))
    batches = json.loads(args.data.read_text())
    if not isinstance(batches, list) or not batches:
        parser.error("data must be a nonempty array of {tokens, targets} batches")
    torch, np = configure_determinism()
    torch.cuda.reset_peak_memory_stats()
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
        if owner.metadata.get("training_envelope") != envelope:
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
        peak = torch.cuda.max_memory_allocated()
        if peak > args.max_torch_memory_mib * 1024**2:
            raise RuntimeError("Torch allocation exceeds the experiment envelope")
        row["peak_torch_memory_bytes"] = peak
        owner.external["batch_cursor"] = cursor + 1
        owner.save(args.output / f"step-{owner.metadata['steps']}.pt")
        if torch.cuda.max_memory_allocated() > args.max_torch_memory_mib * 1024**2:
            raise RuntimeError("checkpoint allocation exceeds the experiment envelope")
        with (args.output / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    (args.output / "resources.json").write_text(
        json.dumps(
            {
                "peak_torch_memory_bytes": torch.cuda.max_memory_allocated(),
                "max_torch_memory_mib": args.max_torch_memory_mib,
            }
        )
    )


if __name__ == "__main__":
    main()
