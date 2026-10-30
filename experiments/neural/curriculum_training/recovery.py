"""Submit the preregistered Kaggle recovery after a confirmed pre-training failure."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from experiments.neural.curriculum_training.launch import PARENT, bootstrap
from ontogeny.execution.providers import kaggle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--colab-record", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.colab_record.read_text())
    if (
        prior.get("state") != "FAILED"
        or prior.get("cleanup") != "confirmed"
        or prior.get("reason") != "checkpoint upload failed; see private command log"
    ):
        raise ValueError("recovery requires the recorded pre-training Colab upload failure")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit the amended protocol before submission")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    published = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], text=True
    ).split()[0]
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, published], check=True)
    with args.parent.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != PARENT:
            raise ValueError("parent checkpoint hash mismatch")
    quota = kaggle.quota_hours(kaggle.command(["kaggle", "quota", "--format", "json"]))
    if quota < 2.5 + 340 / 3600 + 0.01:
        raise RuntimeError("Kaggle quota does not leave the declared reserve")
    args.directory.mkdir(parents=True, exist_ok=False)
    journal = args.directory / "controller.json"
    record = {
        "state": "UPLOADING",
        "source": revision,
        "parent": PARENT,
        "prior_colab": str(args.colab_record),
        "quota_before": quota,
        "platform_seconds": 340,
    }
    kaggle.journal(journal, record)
    try:
        dataset = "sparklinghydraulics0/ontogeny-curriculum-" + revision[:12]
        directory = args.directory / "dataset"
        directory.mkdir()
        shutil.copyfile(args.parent, directory / "parent.pt")
        (directory / "dataset-metadata.json").write_text(
            json.dumps(
                {"id": dataset, "title": dataset.split("/")[1], "licenses": [{"name": "other"}]}
            )
        )
        result = kaggle.command(
            ["kaggle", "datasets", "create", "-p", str(directory), "--keep-tabular"], timeout=360
        )
        if dataset not in result or "being created" not in result:
            raise RuntimeError("dataset upload unconfirmed")
        record.update(dataset=dataset, upload=result)
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
        # Recheck quota immediately before submission; do not assume a stale balance.
        quota = kaggle.quota_hours(kaggle.command(["kaggle", "quota", "--format", "json"]))
        if quota < 2.5 + 340 / 3600 + 0.01:
            raise RuntimeError("Kaggle quota changed before submission")
        record.update(state="SUBMITTING", native_id=native, quota_at_submission=quota)
        kaggle.journal(journal, record)
        result = kaggle.command(
            [
                "kaggle",
                "kernels",
                "push",
                "-p",
                str(job),
                "--accelerator",
                "NvidiaTeslaT4",
                "-t",
                "340",
            ]
        )
        if not re.search(r"Kernel version \d+ successfully pushed", result):
            raise RuntimeError("job submission unconfirmed")
        record.update(state="SUBMITTED", submission=result)
        kaggle.journal(journal, record)
        print(json.dumps(record, indent=2))
    except Exception as error:
        record.update(
            state="UNKNOWN" if record["state"] in ("UPLOADING", "SUBMITTING") else "FAILED",
            reason=str(error),
        )
        kaggle.journal(journal, record)
        raise


if __name__ == "__main__":
    main()
