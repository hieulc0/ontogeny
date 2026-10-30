"""Submit one explicitly fresh, bounded pilot after a recorded transport failure."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from experiments.neural.initial_student.job import bootstrap
from ontogeny.execution.providers import kaggle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("publish the amended pilot protocol first")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    published = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], text=True
    ).split()[0]
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, published], check=True)
    quota_text = kaggle.command(["kaggle", "quota", "--format", "json"])
    quota = kaggle.quota_hours(quota_text)
    if quota < 2.5 + 480 / 3600 + 0.01:
        raise RuntimeError("insufficient Kaggle quota after reserve")
    args.directory.mkdir(parents=True, exist_ok=False)
    job = args.directory / "job"
    job.mkdir()
    native = "sparklinghydraulics0/ontogeny-student-fresh-" + revision[:10]
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
    record = {
        "state": "SUBMITTING",
        "native_id": native,
        "source": revision,
        "fresh_lineage": True,
        "quota_before": quota,
        "platform_seconds": 480,
        "reserve_hours": 2.5,
    }
    journal = args.directory / "controller.json"
    kaggle.journal(journal, record)
    try:
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
                "480",
            ]
        )
        if not re.search(r"Kernel version \d+ successfully pushed", response):
            raise RuntimeError("submission unconfirmed")
    except Exception:
        record["state"] = "UNKNOWN"
        kaggle.journal(journal, record)
        raise
    record.update(state="SUBMITTED", submission=response)
    kaggle.journal(journal, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
