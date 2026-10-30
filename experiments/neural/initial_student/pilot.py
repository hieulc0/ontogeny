"""Remote-only finite initialization, token-training and continuation pilot."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-probe", type=Path)
    args = parser.parse_args()
    from mini_gpt.runtime import configure_determinism

    torch, np = configure_determinism()
    if (
        np.__version__ != "2.1.3"
        or torch.cuda.device_count() != 1
        or "T4" not in torch.cuda.get_device_name()
    ):
        raise RuntimeError("the pilot requires the declared one-visible-device T4 profile")
    from mini_gpt.checkpoint.comparison import capture_state, fingerprints
    from mini_gpt.checkpoint.identity import file_digest
    from mini_gpt.learning.state import LearningState
    from mini_gpt.learning.trainer import train_tokens

    source = Path(__file__).resolve().parents[3]
    fixture = source / "experiments/neural/initial_student"
    batches = json.loads((fixture / "batches.json").read_text())
    if args.resume_probe:
        torch.cuda.reset_peak_memory_stats()
        owner = LearningState.load(args.resume_probe)
        item = batches[owner.external["batch_cursor"] % len(batches)]
        x = torch.tensor(item["tokens"], device="cuda")
        y = torch.tensor(item["targets"], device="cuda")
        train_tokens(owner, x, y, max_updates=2)
        owner.external["batch_cursor"] += 1
        args.output.write_text(
            json.dumps(
                {
                    "state": fingerprints(capture_state(owner), torch, np),
                    "peak_torch_memory_bytes": torch.cuda.max_memory_allocated(),
                }
            )
        )
        return
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("model.json", "batches.json", "PROTOCOL.md"):
        shutil.copyfile(fixture / name, args.output / name)
    torch.cuda.reset_peak_memory_stats()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mini_gpt.cli",
            "--config",
            str(fixture / "model.json"),
            "--data",
            str(fixture / "batches.json"),
            "--updates",
            "2",
            "--seed",
            "42",
            "--wall-seconds",
            "220",
            "--output",
            str(args.output / "student"),
        ],
        check=True,
        timeout=240,
    )
    final = LearningState.load(args.output / "student/step-2.pt")
    count = sum(p.numel() for p in final.model.parameters())
    if not 9_500_000 <= count <= 10_500_000 or final.metadata["steps"] != 2:
        raise RuntimeError("unexpected model size or update count")
    expected = fingerprints(capture_state(final), torch, np)
    probe = args.output / "resumed-state.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--resume-probe",
            str(args.output / "student/step-1.pt"),
            "--output",
            str(probe),
        ],
        check=True,
        timeout=120,
    )
    probe_result = json.loads(probe.read_text())
    actual = probe_result["state"]
    if expected != actual:
        raise RuntimeError(
            "fresh-process state differs: " + str([k for k in expected if expected[k] != actual[k]])
        )
    text = b"The river flows past a quiet green hill.\n"
    final.model.eval()
    x = torch.tensor([list(text[:-1])], device="cuda")
    y = torch.tensor([list(text[1:])], device="cuda")
    with torch.no_grad():
        _, loss = final.model(x, y)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite held-out loss")
    training_resources = json.loads((args.output / "student/resources.json").read_text())
    training_rows = [
        json.loads(row) for row in (args.output / "student/updates.jsonl").read_text().splitlines()
    ]
    peak = max(
        torch.cuda.max_memory_allocated(),
        training_resources["peak_torch_memory_bytes"],
        probe_result["peak_torch_memory_bytes"],
        *(row["peak_torch_memory_bytes"] for row in training_rows),
    )
    if peak > 3 * 1024**3 or time.monotonic() - started > 300:
        raise RuntimeError("pilot resource envelope exceeded")
    files = {
        str(p.relative_to(args.output)): {"size_bytes": p.stat().st_size, "sha256": file_digest(p)}
        for p in args.output.rglob("*")
        if p.is_file()
    }
    if sum(item["size_bytes"] for item in files.values()) > 1024**3:
        raise RuntimeError("output size envelope exceeded")
    result = {
        "execution": "SUCCEEDED",
        "science": "NOT_RUN",
        "parameters": count,
        "updates": 2,
        "audit_updates": 1,
        "training_tokens": 128,
        "heldout_tokens": len(text) - 1,
        "heldout_loss": float(loss),
        "continuation_exact": True,
        "state_groups": {k: expected[k] == actual[k] for k in expected},
        "peak_torch_memory_bytes": peak,
        "task_wall_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "files": files,
        "provider": "colab" if os.environ.get("COLAB_RELEASE_TAG") else "kaggle",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "protocol_sha256": file_digest(fixture / "PROTOCOL.md"),
        "python": platform.python_version(),
        "physical_gpus": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines(),
        "package_inventory": sorted(
            (d.metadata["Name"], d.version)
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        ),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
