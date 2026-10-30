"""Confined, immutable artifact publication and non-neural result verification."""

import json
import os
import tempfile
from pathlib import Path

from ontogeny.execution.contracts import ContractError, Request, fingerprint, relative_path
from ontogeny.neural.persistence import digest


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def confined(root: Path, name: str) -> Path:
    path = root / relative_path(name)
    if not path.resolve().is_relative_to(root.resolve()) or any(
        p.is_symlink() for p in [path, *path.parents] if p != root.parent
    ):
        raise ContractError(f"artifact escapes trusted root: {name}")
    return path


def verify_inputs(request: Request, root: Path) -> None:
    for item in request.inputs:
        path = confined(root, item.path)
        if (
            not path.is_file()
            or path.stat().st_size != item.size_bytes
            or digest(path) != item.sha256
        ):
            raise ContractError(f"input hash/size mismatch: {item.name}")


def publish_index(root: Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError("symlink in output bundle")
        if path.is_file() and path.name != "artifacts.json":
            name = path.relative_to(root).as_posix()
            files[name] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
    write_json(root / "artifacts.json", {"schema_version": 1, "files": files})


def verify_bundle(root: Path, request: Request) -> dict:
    index = json.loads((root / "artifacts.json").read_text())
    if set(index) != {"schema_version", "files"} or index["schema_version"] != 1:
        raise ContractError("unsupported artifact manifest")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "artifacts.json"
    }
    if actual != set(index["files"]) or any(p.is_symlink() for p in root.rglob("*")):
        raise ContractError("missing, unindexed or linked output files")
    for name, entry in index["files"].items():
        path = confined(root, name)
        if (
            set(entry) != {"sha256", "size_bytes"}
            or path.stat().st_size != entry["size_bytes"]
            or digest(path) != entry["sha256"]
        ):
            raise ContractError(f"output integrity mismatch: {name}")
    returned = Request.read(root / "request.json")
    if returned.sha256 != request.sha256:
        raise ContractError("result belongs to a different scientific request")
    result = json.loads((root / "result.json").read_text())
    if (
        set(result) != {"schema_version", "request_sha256", "execution", "science", "reason"}
        or result["schema_version"] != 1
        or result["request_sha256"] != request.sha256
        or result["execution"] not in ("SUCCEEDED", "FAILED", "CANCELLED")
        or result["science"] != "NOT_RUN"
    ):
        raise ContractError("invalid result contract")
    if result["execution"] != "SUCCEEDED":
        raise ContractError(f"remote execution {result['execution']}: {result['reason']}")
    if not set(request.required_outputs).issubset(actual):
        raise ContractError("incomplete required outputs")
    checkpoints = json.loads((root / "checkpoint_manifest.json").read_text())
    if set(checkpoints) != {"schema_version", "checkpoints"} or not checkpoints["checkpoints"]:
        raise ContractError("missing checkpoint lineage")
    for checkpoint in checkpoints["checkpoints"]:
        path = confined(root, checkpoint["path"])
        if digest(path) != checkpoint["sha256"]:
            raise ContractError("checkpoint lineage hash mismatch")
    if sum(e["size_bytes"] for e in index["files"].values()) > request.limits.output_bytes:
        raise ContractError("output budget exceeded")
    return {
        "request_sha256": request.sha256,
        "artifacts_sha256": fingerprint(index),
        "execution": result["execution"],
        "science": result["science"],
        "verified_files": len(actual),
    }


def stage_inputs(request: Request, source: Path, destination: Path) -> None:
    """Map flat provider-mounted files to declared logical paths by hash, never by seed."""
    import shutil

    destination.mkdir(parents=True, exist_ok=False)
    for item in request.inputs:
        candidates = [
            p
            for p in source.rglob(Path(item.path).name)
            if p.is_file()
            and not p.is_symlink()
            and p.stat().st_size == item.size_bytes
            and digest(p) == item.sha256
        ]
        if len(candidates) != 1:
            raise ContractError(f"expected one exact mounted input for {item.name}")
        target = confined(destination, item.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with candidates[0].open("rb") as original, target.open("xb") as output:
            shutil.copyfileobj(original, output)
    verify_inputs(request, destination)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage_inputs(Request.read(args.request), args.source, args.output)


if __name__ == "__main__":
    main()
