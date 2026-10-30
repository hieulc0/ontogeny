"""Render a private pinned diagnostic GPU job after publishing source."""

import json
import subprocess
from pathlib import Path

from ontogeny.neural.persistence import UPSTREAM_SHA


def main() -> None:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit executable code first")
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], text=True
    ).split()[0]
    if sha != remote:
        raise RuntimeError("publish exact execution source first")
    output = Path("runs/learning-diagnosis/job")
    output.mkdir(parents=True, exist_ok=False)
    launcher = f"""import os, subprocess, sys
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    raise RuntimeError('Kaggle required')
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA required')
def run(args,cwd=None):
    subprocess.run(args,cwd=cwd,check=True,timeout=1500)
for name,revision in [('ontogeny','{sha}'),('adaptive-mini-gpt','{UPSTREAM_SHA}')]:
    root=Path('/tmp')/name
    run(['git','clone','https://github.com/hieulc0/'+name+'.git',str(root)])
    run(['git','checkout','--detach',revision],root)
run([sys.executable,'-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124'])
run([sys.executable,'-m','pip','install','numpy==2.1.3','pyyaml==6.0.2'])
os.environ['PYTHONPATH']='/tmp/ontogeny/src:/tmp/adaptive-mini-gpt/src'
run([sys.executable,'kaggle/diagnose_learning.py','--upstream','/tmp/adaptive-mini-gpt',
     '--output','/kaggle/working/learning-diagnosis'],Path('/tmp/ontogeny'))
"""
    (output / "launch.py").write_text(launcher)
    (output / "kernel-metadata.json").write_text(
        json.dumps(
            dict(
                id="sparklinghydraulics0/ontogeny-learning-diagnosis",
                title="ontogeny-learning-diagnosis",
                code_file="launch.py",
                language="python",
                kernel_type="script",
                is_private=True,
                enable_gpu=True,
                enable_internet=True,
                dataset_sources=["sparklinghydraulics0/ontogeny-l1-recovery-input-v1"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
