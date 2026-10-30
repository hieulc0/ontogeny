"""One Colab assignment with pinned source, bounded work and verified artifact export."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from ontogeny.execution.providers import colab, colab_transfer, kaggle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    published = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], text=True
    ).split()[0]
    if revision != published:
        raise RuntimeError("source must be published at origin/main")
    args.directory.mkdir(parents=True, exist_ok=False)
    session = colab.Session(args.directory / "private-colab")
    journal = args.directory / "controller.json"
    record: dict = {"source": revision, "state": "ALLOCATING", "session": session.name}
    kaggle.journal(journal, record)
    root = "/content/acquisition-" + session.nonce
    script = session.directory / "bootstrap.py"
    script.write_text(f"""import os, subprocess, sys, json, hashlib, zipfile, traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
root=Path({root!r}); root.mkdir(parents=True)
repo=root/'source'; result=root/'result'
def run(args, cwd=None, timeout=500):
 subprocess.run(args,cwd=cwd,check=True,timeout=timeout)
try:
 run(['git','clone','https://github.com/hieulc0/ontogeny.git',str(repo)])
 run(['git','checkout','--detach',{revision!r}],repo)
 run([sys.executable,'-m','pip','install','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124'])
 run([sys.executable,'-m','pip','install','numpy==2.1.3'])
 os.environ['PYTHONPATH']=str(repo/'src')
 run([sys.executable,str(repo/'experiments/neural/concept_acquisition/worker.py'),'--source',{revision!r},'--output',str(result)],repo,620)
except BaseException:
 result.mkdir(exist_ok=True)
 (result/'bootstrap-failure.txt').write_text(traceback.format_exc())
 raise
finally:
 result.mkdir(exist_ok=True)
 files={{p.name:hashlib.sha256(p.read_bytes()).hexdigest() 
        for p in result.iterdir() if p.is_file()}}
 (result/'manifest.json').write_text(json.dumps(files))
 with zipfile.ZipFile(root/'result.zip','w',compression=zipfile.ZIP_STORED) as archive:
  for p in result.iterdir():
   if p.is_file():archive.write(p,p.name)
""")
    try:
        record["readiness"] = session.allocate()
        record["state"] = "RUNNING"
        kaggle.journal(journal, record)
        response = session.invoke(
            ["exec", "-s", session.name, "-f", str(script), "--timeout", "1200"], 1250
        )
        record["returncode"] = response.returncode
        colab_transfer.download_file(
            session, root + "/result.zip", args.directory / "result.zip", max_bytes=256 * 1024**2
        )
        destination = args.directory / "result"
        colab.extract_bundle(args.directory / "result.zip", destination, 256 * 1024**2)
        manifest = json.loads((destination / "manifest.json").read_text())
        if {p.name for p in destination.iterdir()} != set(manifest) | {"manifest.json"}:
            raise RuntimeError("artifact inventory mismatch")
        for name, digest in manifest.items():
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("artifact hash mismatch")
        result = json.loads((destination / "result.json").read_text())
        if (
            response.returncode
            or result["source"] != revision
            or result["execution"] != "COMPLETE"
            or len(result["records"]) != 12
        ):
            raise RuntimeError("incomplete cohort; inspect recovered evidence")
        session.release()
        record.update(state="COMPLETE", cleanup="confirmed")
        kaggle.journal(journal, record)
    except BaseException as error:
        record["reason"] = str(error)
        if isinstance(error, colab.ColabError) and not error.fallback_safe:
            record.update(state="UNKNOWN", cleanup="unconfirmed; reconcile owned assignment")
        elif record["state"] == "RUNNING":
            record.update(state="UNKNOWN", cleanup="not attempted; reconcile worker before release")
        else:
            try:
                session.release()
                record.update(state="FAILED", cleanup="confirmed")
            except BaseException:
                record.update(state="UNKNOWN", cleanup="unconfirmed")
        kaggle.journal(journal, record)
        raise


if __name__ == "__main__":
    main()
