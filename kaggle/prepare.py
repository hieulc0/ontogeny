"""Prepare an immutable Kaggle job from a clean, pushed Ontogeny commit."""

import argparse
import json
import subprocess
from pathlib import Path

from ontogeny.neural.persistence import UPSTREAM_SHA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "experiment", "recovery"], default="smoke")
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", args.revision], text=True).strip()
    if len(sha) != 40:
        raise RuntimeError("full commit required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit executable code first")
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], text=True
    ).split()[0]
    published = (
        subprocess.run(["git", "merge-base", "--is-ancestor", sha, remote], check=False).returncode
        == 0
    )
    if not published:
        raise RuntimeError("selected revision must be published in origin/main history")
    args.job.mkdir(parents=True, exist_ok=False)
    driver = "kaggle/recover_l1.py" if args.mode == "recovery" else "kaggle/run_l1.py"
    code = f'''import os, subprocess, sys, json, traceback
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    raise RuntimeError("Kaggle required")
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")
def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True, timeout=4200)
for name, revision in [("ontogeny", "{sha}"), ("adaptive-mini-gpt", "{UPSTREAM_SHA}")]:
    root = Path("/tmp") / name
    run(["git", "clone", "https://github.com/hieulc0/" + name + ".git", str(root)])
    run(["git", "checkout", "--detach", revision], root)
run([sys.executable, "-m", "pip", "install", "torch==2.6.0",
     "--index-url", "https://download.pytorch.org/whl/cu124"])
run([sys.executable, "-m", "pip", "install", "numpy==2.1.3", "pyyaml==6.0.2",
     "pytest==8.3.5"])
os.environ["PYTHONPATH"] = "/tmp/ontogeny/src:/tmp/adaptive-mini-gpt/src"
run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", "/tmp/ontogeny",
     "-e", "/tmp/adaptive-mini-gpt"])
run([sys.executable, "-m", "pytest", "-q", "tests"], Path("/tmp/ontogeny"))
run([sys.executable, "{driver}", "--mode", "{args.mode}",
     "--upstream", "/tmp/adaptive-mini-gpt",
     "--output", "/kaggle/working/l1-{args.mode}"], Path("/tmp/ontogeny"))
'''
    (args.job / "launch.py").write_text(code)
    metadata = {
        "id": f"sparklinghydraulics0/ontogeny-l1-{args.mode}",
        "title": f"ontogeny-l1-{args.mode}",
        "code_file": "launch.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
    }
    if args.mode == "recovery":
        metadata["dataset_sources"] = ["sparklinghydraulics0/ontogeny-l1-recovery-input-v1"]
        metadata["id"] = "sparklinghydraulics0/ontogeny-l1-recovery-verified"
        metadata["title"] = "ontogeny-l1-recovery-verified"
    (args.job / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
