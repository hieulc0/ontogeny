"""Remote-only full-state interrupted-segment comparison."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PARENT = "1b712fd967ef31148925fa79bdfa092cf5eb64a6c3ad5f3e1ee9b492b9008e42"
DATA = "f2841088feefa520c1d545326585d158e61c31c990071937140357191c49708f"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from mini_gpt.runtime import configure_determinism

    torch, np = configure_determinism()
    if (
        np.__version__ != "2.1.3"
        or torch.cuda.device_count() != 1
        or "T4" not in torch.cuda.get_device_name()
    ):
        raise ValueError("one visible T4 and pinned runtime required")
    from mini_gpt.checkpoint.comparison import capture_state, fingerprints
    from mini_gpt.checkpoint.identity import file_digest
    from mini_gpt.data.curriculum import generate
    from mini_gpt.learning.state import LearningState

    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    if file_digest(args.parent) != PARENT:
        raise ValueError("parent hash mismatch")
    manifest = generate(args.output / "data")
    if manifest["files"]["train-batches.json"] != DATA:
        raise ValueError("dataset hash mismatch")
    repo = Path(__file__).resolve().parents[3]
    config = repo / "experiments/neural/initial_student/model.json"
    data = args.output / "data/train-batches.json"
    batches = json.loads(data.read_text())
    parent = LearningState.load(args.parent)
    start_cursor = parent.external["batch_cursor"]
    training_tokens = 2 * sum(
        len(batches[(start_cursor + index) % len(batches)]["tokens"][0]) for index in range(20)
    )
    if training_tokens > 5000:
        raise ValueError("interaction token budget exceeded")
    del parent
    common = [
        sys.executable,
        "-m",
        "mini_gpt.cli",
        "--config",
        str(config),
        "--data",
        str(data),
        "--wall-seconds",
        "180",
        "--max-torch-memory-mib",
        "3072",
        "--checkpoint-every",
        "20",
    ]

    def train(extra: list[str]) -> None:
        subprocess.run(common + extra, check=True, timeout=200, env=os.environ.copy())

    def new(output: str, stop: bool) -> None:
        extra = [
            "--output",
            str(args.output / output),
            "--resume",
            str(args.parent),
            "--checkpoint-sha256",
            PARENT,
            "--new-segment",
            "interruption-audit",
            "--additional-updates",
            "20",
        ]
        if stop:
            extra.extend(["--stop-after-updates", "10"])
        train(extra)

    new("uninterrupted", False)
    new("interrupted", True)
    mid = args.output / "interrupted/step-15.pt"
    train(
        [
            "--output",
            str(args.output / "resumed"),
            "--resume",
            str(mid),
            "--checkpoint-sha256",
            file_digest(mid),
            "--resume-segment",
            "interruption-audit",
        ]
    )
    inputs = batches[start_cursor % len(batches)]["tokens"]
    initial_logits = []
    for path in (args.parent, args.output / "uninterrupted/segment-start.pt"):
        owner = LearningState.load(path)
        owner.model.eval()
        with torch.no_grad():
            prediction, _ = owner.model(torch.tensor(inputs, device="cuda"))
        initial_logits.append(prediction.detach().cpu())
    initial_logits_equal = bool(torch.equal(initial_logits[0], initial_logits[1]))
    if not initial_logits_equal:
        raise RuntimeError("segment start changes parent logits")
    state_groups = []
    logits = []
    for path in (args.output / "uninterrupted/step-25.pt", args.output / "resumed/step-25.pt"):
        owner = LearningState.load(path)
        state_groups.append(fingerprints(capture_state(owner), torch, np))
        owner.model.eval()
        with torch.no_grad():
            prediction, _ = owner.model(torch.tensor(inputs, device="cuda"))
        logits.append(prediction.detach().cpu())
    matches = {key: state_groups[0][key] == state_groups[1][key] for key in state_groups[0]}
    logits_equal = bool(torch.equal(logits[0], logits[1]))
    if not all(matches.values()) or not logits_equal:
        raise RuntimeError("continuation state or logits mismatch")
    if time.monotonic() - started > 300:
        raise TimeoutError("worker time budget exceeded")
    files = {
        str(p.relative_to(args.output)): {"sha256": file_digest(p), "size_bytes": p.stat().st_size}
        for p in args.output.rglob("*")
        if p.is_file()
    }
    if sum(item["size_bytes"] for item in files.values()) > 1024**3:
        raise ValueError("artifact budget exceeded")
    result = {
        "execution": "SUCCEEDED",
        "science": "NOT_RUN",
        "parent_sha256": PARENT,
        "state_groups": matches,
        "logits_equal": logits_equal,
        "initial_logits_equal": initial_logits_equal,
        "training_input_tokens": training_tokens,
        "physical_updates": 40,
        "continuing_updates": 20,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "task_wall_seconds": time.monotonic() - started,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "files": files,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
