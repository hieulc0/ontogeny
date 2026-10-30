"""Low-level execution lifecycle without provider-specific learner arguments."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from ontogeny.execution.artifacts import verify_bundle, write_json
from ontogeny.execution.contracts import Request, fingerprint
from ontogeny.execution.providers import kaggle
from ontogeny.neural.persistence import digest
from ontogeny.world.state import WorldConfig


def qualification_request(source_sha: str, protocol_sha: str) -> Request:
    return Request.parse(
        {
            "schema_version": 2,
            "run_id": "execution-foundation",
            "trial_id": "complete-state",
            "arm_id": "unchanged-policy",
            "protocol_sha256": protocol_sha,
            "split": "diagnostic",
            "split_manifest_sha256": fingerprint(
                {"world_seed": 91823, "episodes": 1, "kind": "continuation_fixture"}
            ),
            "operation": "continuation_check",
            "source": {"ontogeny_sha": source_sha, "engine": "mini_gpt"},
            "runtime": {
                "torch": "2.6.0+cu124",
                "cuda": "12.4",
                "gpu": "Tesla T4",
                "visible_cuda_devices": 1,
            },
            "initialization": {"kind": "fresh", "seed": 42, "checkpoint": None},
            "world": {
                "seed": 91823,
                "episodes": 1,
                "config": asdict(WorldConfig(initial_energy=100, max_energy=100, max_steps=10)),
            },
            "objective": {
                "algorithm": "reinforce",
                "learning_rate": 0.0005,
                "gamma": 0.97,
                "baseline_decay": 0.9,
            },
            "targets": {"kind": "engineering", "criterion": "state_integrity"},
            "limits": {
                "updates": 3,
                "interactions": 22,
                "wall_seconds": 120,
                "output_bytes": 64 * 1024 * 1024,
                "torch_memory_bytes": 128 * 1024 * 1024,
            },
            "inputs": [],
            "required_outputs": [
                "runtime.json",
                "execution.json",
                "metrics.json",
                "predictions.jsonl",
                "learning.jsonl",
                "checkpoint_manifest.json",
            ],
        }
    )


def checkpoint_request(parent: Request, checkpoint: Path, operation: str) -> Request:
    """Declare an explicit full-state input without changing its continuing lineage."""
    if operation not in ("train", "evaluate"):
        raise ValueError("checkpoint operation must be train or evaluate")
    value = parent.to_dict()
    value["operation"] = operation
    value["split"] = "train" if operation == "train" else "development"
    value["initialization"] = {"kind": "resume", "seed": None, "checkpoint": "parent"}
    value["inputs"] = [
        {
            "name": "parent",
            "path": "checkpoints/parent.pt",
            "sha256": digest(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "role": "checkpoint",
            "trusted": True,
        }
    ]
    if operation == "evaluate":
        value["world"]["seed"] = 92823
        value["world"]["episodes"] = 2
        value["split_manifest_sha256"] = fingerprint(value["world"])
    return Request.parse(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("qualification-request")
    example.add_argument(
        "--source-sha", help="optional source revision; defaults to the current checkout"
    )
    example.add_argument("--protocol", type=Path, required=True)
    example.add_argument("--output", type=Path, required=True)
    checkpoint = commands.add_parser("checkpoint-request")
    checkpoint.add_argument("--parent-request", type=Path, required=True)
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument("--operation", choices=["train", "evaluate"], required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    upload = commands.add_parser("upload-inputs")
    upload.add_argument("--request", type=Path, required=True)
    upload.add_argument("--inputs", type=Path, required=True)
    upload.add_argument("--directory", type=Path, required=True)
    for name in ("validate", "prepare", "status", "collect", "verify"):
        p = commands.add_parser(name)
        p.add_argument("--request", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--job", type=Path, required=True)
            p.add_argument("--dataset", action="append", default=[])
            p.add_argument("--platform-seconds", type=int, default=1200)
        if name in ("status", "collect"):
            p.add_argument("--ledger", type=Path, required=True)
        if name in ("collect", "verify"):
            p.add_argument("--directory", type=Path, required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--job", type=Path, required=True)
    submit.add_argument("--ledger", type=Path, required=True)
    submit.add_argument("--reserve-hours", type=float, default=2.5)
    route = commands.add_parser("run")
    route.add_argument("--request", type=Path, required=True)
    route.add_argument("--directory", type=Path, required=True)
    route.add_argument("--ledger", type=Path, required=True)
    route.add_argument("--inputs", type=Path, required=True)
    route.add_argument("--provider", choices=["auto", "colab", "kaggle"], default="auto")
    route.add_argument("--reserve-hours", type=float, default=2.5)
    reconciliation = commands.add_parser("reconcile")
    reconciliation.add_argument("--request", type=Path, required=True)
    reconciliation.add_argument("--directory", type=Path, required=True)
    reconciliation.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "checkpoint-request":
        request = checkpoint_request(
            Request.read(args.parent_request), args.checkpoint, args.operation
        )
        write_json(args.output, request.to_dict())
        result = {"request_sha256": request.sha256}
    elif args.command == "upload-inputs":
        result = {
            "dataset": kaggle.upload_inputs(Request.read(args.request), args.inputs, args.directory)
        }
    elif args.command == "reconcile":
        from ontogeny.execution.routing import reconcile

        result = reconcile(Request.read(args.request), args.directory, args.ledger)
    elif args.command == "run":
        from ontogeny.execution.routing import run

        result = run(
            Request.read(args.request),
            args.directory,
            args.ledger,
            args.inputs,
            provider=args.provider,
            reserve_hours=args.reserve_hours,
        )
    elif args.command == "qualification-request":
        source = (
            args.source_sha
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
        request = qualification_request(
            source, hashlib.sha256(args.protocol.read_bytes()).hexdigest()
        )
        write_json(args.output, request.to_dict())
        result = {"request_sha256": request.sha256}
    elif args.command == "submit":
        result = kaggle.submit(args.job, args.ledger, reserve_hours=args.reserve_hours)
    else:
        request = Request.read(args.request)
        if args.command == "validate":
            result = {"request_sha256": request.sha256, "valid": True}
        elif args.command == "prepare":
            result = vars(
                kaggle.prepare(
                    request,
                    args.job,
                    dataset_sources=tuple(args.dataset),
                    platform_seconds=args.platform_seconds,
                )
            )
        elif args.command == "status":
            result = kaggle.status(request, args.ledger)
        elif args.command == "collect":
            result = kaggle.collect(request, args.ledger, args.directory)
        else:
            result = verify_bundle(args.directory, request)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
