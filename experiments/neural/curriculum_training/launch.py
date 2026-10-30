"""Single-attempt provider admission and artifact collection for the curriculum run."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from ontogeny.execution.providers import colab, kaggle

PARENT = "8e4ea63d5e3c3719b30364614005b631d202dbf09cb666796f89f0fd2b35df7d"


def bootstrap(revision: str, root: str, parent: str) -> str:
    return f"""import os,subprocess,sys,zipfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
root=Path({root!r});root.mkdir(parents=True,exist_ok=True)
repo=root/'source'
def run(args,cwd=None,timeout=600):subprocess.run(args,cwd=cwd,check=True,timeout=timeout)
try:
 run(['git','clone','https://github.com/hieulc0/ontogeny.git',str(repo)])
 run(['git','checkout','--detach',{revision!r}],repo)
 run([sys.executable,'-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124'])
 run([sys.executable,'-m','pip','install','numpy==2.1.3'])
 os.environ['PYTHONPATH']=str(repo/'src')
 run([sys.executable,str(repo/'experiments/neural/curriculum_training/worker.py'),'--parent',{parent!r},'--output',str(root/'result')],repo,timeout=320)
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
        or not result["continuation_exact"]
        or result["source_revision"] != revision
        or result["parent_sha256"] != PARENT
    ):
        raise ValueError("run contract did not pass")
    paths = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if paths != set(result["files"]) | {"result.json"}:
        raise ValueError("result artifact inventory mismatch")
    for name, item in result["files"].items():
        path = directory / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ValueError("result artifact mismatch: " + name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit all experiment code before submission")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    published = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], text=True
    ).split()[0]
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, published], check=True)
    with args.parent.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != PARENT:
            raise ValueError("parent hash mismatch")
    args.directory.mkdir(parents=True, exist_ok=False)
    journal = args.directory / "controller.json"
    record = {"source": revision, "state": "ALLOCATING", "parent": PARENT}
    kaggle.journal(journal, record)
    session = colab.Session(args.directory / "private-colab")
    record["session"] = session.name
    kaggle.journal(journal, record)
    try:
        try:
            proof = session.allocate()
        except colab.ColabError as error:
            if not error.fallback_safe:
                record.update(state="UNKNOWN", reason=str(error))
                kaggle.journal(journal, record)
                raise
            session.release()
            record["colab_reason"] = str(error)
        else:
            record.update(state="STAGING", provider="colab", readiness=proof)
            kaggle.journal(journal, record)
            parent = "/content/ontogeny-parent-" + session.nonce + ".pt"
            upload = session.invoke(["upload", "-s", session.name, str(args.parent), parent], 360)
            if upload.returncode:
                raise RuntimeError("checkpoint upload failed; see private command log")
            root = "/content/curriculum-" + session.nonce
            script = session.directory / "bootstrap.py"
            script.write_text(bootstrap(revision, root, parent))
            record["state"] = "RUNNING"
            kaggle.journal(journal, record)
            execution = session.invoke(
                ["exec", "-s", session.name, "-f", str(script), "--timeout", "600"], 650
            )
            if execution.returncode:
                raise RuntimeError(
                    "remote execution failed or transport closed; see private command log"
                )
            download = session.invoke(
                [
                    "download",
                    "-s",
                    session.name,
                    root + "/result.zip",
                    str(args.directory / "result.zip"),
                ],
                600,
            )
            if download.returncode:
                raise RuntimeError("remote artifact download failed; see private command log")
            colab.extract_bundle(args.directory / "result.zip", args.directory / "result", 1024**3)
            result = verify(args.directory / "result", revision)
            session.release()
            record.update(state="COMPLETE", result=result, cleanup="confirmed")
            kaggle.journal(journal, record)
            print(json.dumps(record, indent=2))
            return
        quota = kaggle.quota_hours(kaggle.command(["kaggle", "quota", "--format", "json"]))
        if quota < 2.5 + 600 / 3600 + 0.01:
            raise RuntimeError("Kaggle quota does not leave declared reserve")
        dataset_dir = args.directory / "dataset"
        dataset_dir.mkdir()
        dataset = "sparklinghydraulics0/ontogeny-curriculum-" + revision[:12]
        shutil.copyfile(args.parent, dataset_dir / "parent.pt")
        (dataset_dir / "dataset-metadata.json").write_text(
            json.dumps(
                {"id": dataset, "title": dataset.split("/")[1], "licenses": [{"name": "other"}]}
            )
        )
        record.update(state="UPLOADING", quota_before=quota, dataset=dataset)
        kaggle.journal(journal, record)
        response = kaggle.command(
            ["kaggle", "datasets", "create", "-p", str(dataset_dir), "--keep-tabular"], timeout=360
        )
        if dataset not in response or "being created" not in response:
            raise RuntimeError("private checkpoint upload unconfirmed")
        job = args.directory / "job"
        job.mkdir()
        native = "sparklinghydraulics0/ontogeny-curriculum-run-" + revision[:10]
        (job / "launch.py").write_text(
            bootstrap(
                revision,
                "/kaggle/working/curriculum",
                "/kaggle/input/" + dataset.split("/")[1] + "/parent.pt",
            )
        )
        (job / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": native,
                    "title": native.split("/")[1],
                    "code_file": "launch.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": True,
                    "enable_gpu": True,
                    "enable_internet": True,
                    "dataset_sources": [dataset],
                }
            )
        )
        record.update(state="SUBMITTING", native_id=native, upload=response)
        kaggle.journal(journal, record)
        response = kaggle.command(
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(job),
                "--accelerator",
                "NvidiaTeslaT4",
                "-t",
                "600",
            ]
        )
        if not re.search(r"Kernel version \d+ successfully pushed", response):
            raise RuntimeError("Kaggle submission unconfirmed")
        record.update(state="SUBMITTED", submission=response)
        kaggle.journal(journal, record)
        print(json.dumps(record, indent=2))
    except Exception as error:
        if record.get("state") in ("STAGING", "RUNNING", "ALLOCATING"):
            # Once execution starts, leave the session intact for reconciliation and export.
            if record.get("state") == "RUNNING":
                record.update(state="UNKNOWN", cleanup="not attempted")
            else:
                try:
                    session.release()
                    record.update(state="FAILED", cleanup="confirmed")
                except Exception:
                    record.update(state="UNKNOWN", cleanup="unconfirmed")
        elif record.get("state") in ("SUBMITTING", "UPLOADING"):
            record["state"] = "UNKNOWN"
        elif record.get("state") != "UNKNOWN":
            record["state"] = "FAILED"
        record.update(reason=str(error), error_kind=type(error).__name__)
        kaggle.journal(journal, record)
        raise


if __name__ == "__main__":
    main()
