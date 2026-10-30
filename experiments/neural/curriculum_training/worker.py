"""Remote-only, bounded continuation and independent process audit."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PARENT = "8e4ea63d5e3c3719b30364614005b631d202dbf09cb666796f89f0fd2b35df7d"
DATA = "f2841088feefa520c1d545326585d158e61c31c990071937140357191c49708f"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("run", "evaluate", "audit"), default="run")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    from mini_gpt.runtime import configure_determinism

    torch, np = configure_determinism()
    if (
        np.__version__ != "2.1.3"
        or torch.cuda.device_count() != 1
        or "T4" not in torch.cuda.get_device_name()
    ):
        raise RuntimeError("expected one visible T4 and the pinned runtime")
    from mini_gpt.checkpoint.comparison import capture_state, fingerprints
    from mini_gpt.checkpoint.identity import file_digest
    from mini_gpt.data.curriculum import batch, generate
    from mini_gpt.learning.state import LearningState
    from mini_gpt.learning.trainer import train_tokens

    if args.mode in ("evaluate", "audit"):
        owner = LearningState.load(args.checkpoint)
        if args.mode == "audit":
            rows = json.loads(args.data.read_text())
            item = rows[owner.external["batch_cursor"] % len(rows)]
            train_tokens(
                owner,
                torch.tensor(item["tokens"], device="cuda"),
                torch.tensor(item["targets"], device="cuda"),
                max_updates=5,
            )
            owner.external["batch_cursor"] += 1
            if torch.cuda.max_memory_allocated() > 3 * 1024**3:
                raise RuntimeError("audit memory envelope exceeded")
            args.output.write_text(json.dumps(fingerprints(capture_state(owner), torch, np)))
            return
        rows = [json.loads(x) for x in args.data.read_text().splitlines()][:24]
        owner.model.eval()
        sums: dict = {}
        with torch.no_grad():
            for row in rows:
                item = batch(row)
                x = torch.tensor(item["tokens"], device="cuda")
                y = torch.tensor(item["targets"], device="cuda")
                logits, loss = owner.model(x, y)
                mask = y != -100
                n = int(mask.sum().item())
                correct = int(((logits.argmax(dim=-1) == y) & mask).sum().item())
                if not math.isfinite(float(loss)):
                    raise ValueError("nonfinite evaluation loss")
                domain = sums.setdefault(
                    row["domain"], {"examples": 0, "bytes": 0, "correct": 0, "loss_sum": 0.0}
                )
                domain["examples"] += 1
                domain["bytes"] += n
                domain["correct"] += correct
                domain["loss_sum"] += float(loss) * n
        if torch.cuda.max_memory_allocated() > 3 * 1024**3:
            raise RuntimeError("evaluation memory envelope exceeded")
        args.output.write_text(json.dumps(sums, indent=2))
        return
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    if file_digest(args.parent) != PARENT:
        raise ValueError("parent checkpoint hash mismatch")
    manifest = generate(args.output / "data")
    if manifest["files"]["train-batches.json"] != DATA:
        raise ValueError("training data hash mismatch")
    data = args.output / "data"
    env = os.environ.copy()
    script = str(Path(__file__).resolve())

    def child(arguments: list[str], timeout: int = 180) -> None:
        subprocess.run([sys.executable, *arguments], check=True, timeout=timeout, env=env)

    child(
        [
            script,
            "--mode",
            "evaluate",
            "--checkpoint",
            str(args.parent),
            "--data",
            str(data / "development.jsonl"),
            "--output",
            str(args.output / "before.json"),
        ]
    )
    repo = Path(__file__).resolve().parents[3]
    child(
        [
            "-m",
            "mini_gpt.cli",
            "--config",
            str(repo / "experiments/neural/initial_student/model.json"),
            "--data",
            str(data / "train-batches.json"),
            "--output",
            str(args.output / "student"),
            "--resume",
            str(args.parent),
            "--checkpoint-sha256",
            PARENT,
            "--new-segment",
            "starter",
            "--additional-updates",
            "3",
            "--wall-seconds",
            "120",
            "--max-torch-memory-mib",
            "3072",
        ]
    )
    student = args.output / "student"
    child(
        [
            script,
            "--mode",
            "audit",
            "--checkpoint",
            str(student / "segment-start.pt"),
            "--data",
            str(data / "train-batches.json"),
            "--output",
            str(args.output / "audit.json"),
        ]
    )
    actual = json.loads((args.output / "audit.json").read_text())
    expected = fingerprints(capture_state(LearningState.load(student / "step-3.pt")), torch, np)
    if actual != expected:
        raise RuntimeError("fresh-process continuation mismatch")
    child(
        [
            script,
            "--mode",
            "evaluate",
            "--checkpoint",
            str(student / "step-5.pt"),
            "--data",
            str(data / "development.jsonl"),
            "--output",
            str(args.output / "after.json"),
        ]
    )
    if time.monotonic() - start > 200:
        raise RuntimeError("task wall budget exceeded")
    files = {
        str(p.relative_to(args.output)): {"sha256": file_digest(p), "size_bytes": p.stat().st_size}
        for p in args.output.rglob("*")
        if p.is_file()
    }
    if sum(item["size_bytes"] for item in files.values()) > 1024**3:
        raise RuntimeError("artifact budget exceeded")
    result = {
        "execution": "SUCCEEDED",
        "science": "NOT_QUALIFIED",
        "parent_sha256": PARENT,
        "data_sha256": DATA,
        "continuation_exact": True,
        "state_groups": list(expected),
        "task_wall_seconds": time.monotonic() - start,
        "updates": 3,
        "audit_updates": 1,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "files": files,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
