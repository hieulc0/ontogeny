"""Run the bounded student pilot on one verified provider; never restart a started trial."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from experiments.neural.initial_student.job import bootstrap, verify
from ontogeny.execution.providers import colab, kaggle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit the pilot and protocol before allocation")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    published = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], text=True
    ).split()[0]
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, published], check=True)
    args.directory.mkdir(parents=True, exist_ok=False)
    record = {"source": revision, "state": "ALLOCATING", "preference": "colab-first"}
    journal = args.directory / "controller.json"
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
            record.update(state="RUNNING", provider="colab", session=session.name, readiness=proof)
            kaggle.journal(journal, record)
            remote = "/content/student-" + session.nonce
            script = session.directory / "pilot_bootstrap.py"
            script.write_text(bootstrap(revision, remote))
            session.invoke(
                ["exec", "-s", session.name, "-f", str(script), "--timeout", "1000"], 1050
            )
            archive = args.directory / "result.zip"
            session.invoke(
                ["download", "-s", session.name, remote + "/result.zip", str(archive)], 300
            )
            colab.extract_bundle(archive, args.directory / "result", 1024**3)
            result = verify(args.directory / "result")
            session.release()
            record.update(state="COMPLETE", result=result, cleanup="confirmed")
            kaggle.journal(journal, record)
            print(json.dumps(record, indent=2))
            return
        quota = kaggle.quota_hours(kaggle.command(["kaggle", "quota", "--format", "json"]))
        if quota < 2.5 + 480 / 3600 + 0.01:
            raise RuntimeError("Kaggle quota does not leave the declared reserve")
        job = args.directory / "kaggle-job"
        job.mkdir()
        native = "sparklinghydraulics0/ontogeny-student-" + revision[:10]
        (job / "launch.py").write_text(bootstrap(revision, "/kaggle/working/student-pilot"))
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
                    "dataset_sources": [],
                }
            )
        )
        record.update(state="SUBMITTING", provider="kaggle", native_id=native, quota_before=quota)
        kaggle.journal(journal, record)
        submitted = kaggle.command(
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(job),
                "--accelerator",
                "NvidiaTeslaT4",
                "-t",
                "480",
            ]
        )
        if not re.search(r"Kernel version \d+ successfully pushed", submitted):
            raise RuntimeError("Kaggle submission unconfirmed")
        record.update(state="SUBMITTED", submission=submitted)
        kaggle.journal(journal, record)
        print(json.dumps(record, indent=2))
    except Exception as error:
        if record.get("state") in ("RUNNING", "ALLOCATING"):
            try:
                session.release()
            except Exception:
                record.update(state="UNKNOWN", cleanup="unconfirmed")
            else:
                record.update(state="FAILED", cleanup="confirmed")
        elif record.get("state") == "SUBMITTING":
            record["state"] = "UNKNOWN"
        elif record.get("state") != "UNKNOWN":
            record["state"] = "FAILED"
        record.update(reason=str(error), error_kind=type(error).__name__)
        kaggle.journal(journal, record)
        raise


if __name__ == "__main__":
    main()
