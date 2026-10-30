"""Run one bounded remote continuation check with verified checkpoint transport."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from experiments.neural.continuation.worker import PARENT
from ontogeny.execution.providers import colab, colab_transfer, kaggle


def bootstrap(revision: str, root: str, parent: str) -> str:
    return f"""import os,subprocess,sys,zipfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'; os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
root=Path({root!r}); root.mkdir(); repo=root/'source'
def run(args,cwd=None,timeout=900):subprocess.run(args,cwd=cwd,check=True,timeout=timeout)
try:
 run(['git','clone','https://github.com/hieulc0/ontogeny.git',str(repo)])
 run(['git','checkout','--detach',{revision!r}],repo)
 run([sys.executable,'-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124'])
 run([sys.executable,'-m','pip','install','numpy==2.1.3'])
 os.environ['PYTHONPATH']=str(repo/'src')
 run([sys.executable,str(repo/'experiments/neural/continuation/worker.py'),'--parent',{parent!r},'--output',str(root/'result')],repo,timeout=320)
finally:
 with zipfile.ZipFile(root/'result.zip','w',compression=zipfile.ZIP_STORED) as z:
  if (root/'result').exists():
   for p in (root/'result').rglob('*'):
    if p.is_file():z.write(p,p.relative_to(root/'result'))
"""


def verify(directory: Path, revision: str) -> dict:
    result = json.loads((directory / "result.json").read_text())
    if (
        result["execution"] != "SUCCEEDED"
        or result["source_revision"] != revision
        or result["parent_sha256"] != PARENT
        or not result["logits_equal"]
        or not result["initial_logits_equal"]
        or not all(result["state_groups"].values())
    ):
        raise ValueError("continuation contract failed")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(result["files"]) | {"result.json"}:
        raise ValueError("artifact inventory mismatch")
    for name, item in result["files"].items():
        path = directory / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ValueError("artifact hash mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit source before execution")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    published = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], text=True
    ).split()[0]
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, published], check=True)
    with args.parent.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != PARENT:
            raise ValueError("parent mismatch")
    args.directory.mkdir(parents=True, exist_ok=False)
    journal = args.directory / "controller.json"
    record = {"state": "ALLOCATING", "source": revision, "parent": PARENT}
    session = colab.Session(args.directory / "private-colab")
    record["session"] = session.name
    kaggle.journal(journal, record)
    try:
        proof = session.allocate()
        record.update(state="STAGING", readiness=proof)
        kaggle.journal(journal, record)
        parent = "/content/ontogeny-parent-" + session.nonce + ".pt"
        colab_transfer.upload_file(session, args.parent, parent, PARENT)
        root = "/content/continuation-" + session.nonce
        script = session.directory / "bootstrap.py"
        script.write_text(bootstrap(revision, root, parent))
        record["state"] = "RUNNING"
        kaggle.journal(journal, record)
        executed = session.invoke(
            ["exec", "-s", session.name, "-f", str(script), "--timeout", "900"], 950
        )
        if executed.returncode:
            raise RuntimeError("execution failed or transport closed; inspect private log")
        colab_transfer.download_file(
            session, root + "/result.zip", args.directory / "result.zip", max_bytes=1100 * 1024**2
        )
        colab.extract_bundle(args.directory / "result.zip", args.directory / "result", 1024**3)
        result = verify(args.directory / "result", revision)
        session.release()
        record.update(state="COMPLETE", result=result, cleanup="confirmed")
        kaggle.journal(journal, record)
        print(json.dumps(record, indent=2))
    except Exception as error:
        if record["state"] == "RUNNING":
            record.update(state="UNKNOWN", cleanup="not attempted")
        else:
            try:
                session.release()
                record.update(state="FAILED", cleanup="confirmed")
            except Exception:
                record.update(state="UNKNOWN", cleanup="unconfirmed")
        record.update(reason=str(error), error_kind=type(error).__name__)
        kaggle.journal(journal, record)
        raise


if __name__ == "__main__":
    main()
