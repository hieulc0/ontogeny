"""Prepare a finite migration verification job; submission remains explicit."""

import argparse
import json
import subprocess
from pathlib import Path

from mini_gpt.checkpoint.identity import file_digest
from ontogeny.execution.artifacts import write_json

SOURCE_REVISION = json.loads(
    (Path(__file__).resolve().parents[1] / "experiments/integration/reference.json").read_text()
)["source_revision"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", args.revision], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("publish clean source before preparing GPU verification")
    args.job.mkdir(parents=True)
    checkpoint_hash = file_digest(args.checkpoint)
    native = "sparklinghydraulics0/ontogeny-neural-consolidation-" + sha[:10]
    launch = f"""import os,subprocess,sys,json,hashlib,platform,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):raise RuntimeError('Kaggle required')
import torch
if not torch.cuda.is_available():raise RuntimeError('CUDA required; refusing CPU')
def run(args,cwd=None,timeout=600):subprocess.run(args,cwd=cwd,check=True,timeout=timeout)
source=Path('/tmp/ontogeny')
reference=Path('/tmp/neural-reference')
historical=Path('/tmp/organism-reference')
for repository,revision,target in [
 ('ontogeny',{sha!r},source),('adaptive-mini-gpt',{SOURCE_REVISION!r},reference),
 ('ontogeny','a88d7d386eb502e4ffae5bde7dc504441824c76d',historical)]:
 run(['git','clone','https://github.com/hieulc0/'+repository+'.git',str(target)])
 run(['git','checkout','--detach',revision],target)
run([sys.executable,'-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124'])
run([sys.executable,'-m','pip','install','numpy==2.1.3','pytest>=8'])
os.environ['PYTHONPATH']=str(source/'src')+':'+str(reference/'src')
os.environ['REFERENCE_NEURAL_SRC']=str(reference/'src')
os.environ['HISTORICAL_ONTOGENY_SRC']=str(historical/'src')
inputs=[p for p in Path('/kaggle/input').rglob('historical.pt')
        if hashlib.sha256(p.read_bytes()).hexdigest()=={checkpoint_hash!r}]
if len(inputs)!=1:raise RuntimeError('one exact trusted historical checkpoint required')
os.environ['MIGRATION_CHECKPOINT']=str(inputs[0])
os.environ['MIGRATION_CHECKPOINT_SHA256']={checkpoint_hash!r}
output=Path('/kaggle/working/verification')
output.mkdir()
# A new interpreter imports the newly pinned Torch after installation.
run([sys.executable,str(source/'experiments/integration/verify_runtime.py'),
     '--output',str(output),'--native-id',{native!r}],source,timeout=420)
"""
    (args.job / "launch.py").write_text(launch)
    write_json(
        args.job / "kernel-metadata.json",
        {
            "id": native,
            "title": native.split("/")[1],
            "code_file": "launch.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [args.dataset],
        },
    )
    write_json(
        args.job / "qualification.json",
        {
            "source_sha": sha,
            "reference_sha": SOURCE_REVISION,
            "historical_ontogeny_sha": "a88d7d386eb502e4ffae5bde7dc504441824c76d",
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size": args.checkpoint.stat().st_size,
            "dataset": args.dataset,
            "native_id": native,
            "platform_seconds": 720,
            "max_updates": 32,
            "max_interactions": 64,
            "reserve_hours": 2.5,
            "protocol_sha256": file_digest(Path("docs/MIGRATION.md")),
        },
    )
    print(native)


if __name__ == "__main__":
    main()
