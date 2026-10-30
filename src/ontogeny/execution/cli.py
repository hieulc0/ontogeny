"""Low-level execution lifecycle without provider-specific learner arguments."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ontogeny.execution.artifacts import verify_bundle, write_json
from ontogeny.execution.contracts import Request, fingerprint
from ontogeny.execution.providers import kaggle
from ontogeny.neural.persistence import UPSTREAM_SHA
from ontogeny.world.state import WorldConfig


def qualification_request(source_sha: str, protocol_sha: str) -> Request:
    return Request.parse(
        {
            "schema_version": 1,
            "run_id": "execution-foundation",
            "trial_id": "complete-state",
            "arm_id": "unchanged-policy",
            "protocol_sha256": protocol_sha,
            "split": "diagnostic",
            "split_manifest_sha256": fingerprint(
                {"world_seed": 91823, "episodes": 1, "kind": "continuation_fixture"}
            ),
            "operation": "continuation_check",
            "source": {"ontogeny_sha": source_sha, "upstream_sha": UPSTREAM_SHA},
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


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("qualification-request")
    example.add_argument("--source-sha", required=True)
    example.add_argument("--protocol", type=Path, required=True)
    example.add_argument("--output", type=Path, required=True)
    for name in ("validate", "prepare", "status", "collect", "verify"):
        p = commands.add_parser(name)
        p.add_argument("--request", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--job", type=Path, required=True)
            p.add_argument("--dataset", action="append", default=[])
        if name in ("status", "collect"):
            p.add_argument("--ledger", type=Path, required=True)
        if name in ("collect", "verify"):
            p.add_argument("--directory", type=Path, required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--job", type=Path, required=True)
    submit.add_argument("--ledger", type=Path, required=True)
    submit.add_argument("--reserve-hours", type=float, default=2.5)
    args = parser.parse_args()
    if args.command == "qualification-request":
        request = qualification_request(
            args.source_sha, hashlib.sha256(args.protocol.read_bytes()).hexdigest()
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
            result = vars(kaggle.prepare(request, args.job, dataset_sources=tuple(args.dataset)))
        elif args.command == "status":
            result = kaggle.status(request, args.ledger)
        elif args.command == "collect":
            result = kaggle.collect(request, args.ledger, args.directory)
        else:
            result = verify_bundle(args.directory, request)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
