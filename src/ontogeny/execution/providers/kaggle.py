"""Kaggle CLI transport, quota admission and conservative single-attempt journaling."""

import fcntl
import json
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from ontogeny.execution.artifacts import confined, verify_bundle, verify_inputs, write_json
from ontogeny.execution.contracts import (
    ExecutionContext,
    Request,
    canonical,
    fingerprint,
    hexadecimal,
)
from ontogeny.execution.state_identity import STATE_GROUPS


class ProviderError(RuntimeError):
    pass


def command(args: list[str], *, timeout: int = 60) -> str:
    completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    # Some Kaggle CLI application failures return zero. Submission confirmation is
    # checked separately and never inferred from this process exit code alone.
    if completed.returncode:
        raise ProviderError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout + completed.stderr


def quota_hours(text: str) -> float:
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ProviderError("unexpected Kaggle quota response")
    candidates = [r for r in rows if r.get("resource") == "GPU"]
    if len(candidates) != 1:
        raise ProviderError("GPU quota unavailable")
    remaining = candidates[0].get("remaining")
    if not isinstance(remaining, str) or not re.fullmatch(r"\d+(?:\.\d+)?h", remaining):
        raise ProviderError("unrecognized quota units")
    return float(remaining[:-1])


def journal(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def locked(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def upload_inputs(
    request: Request, inputs: Path, directory: Path, *, owner: str = "sparklinghydraulics0"
) -> str:
    """Create a private allowlisted dataset; never update a prior dataset version."""
    import shutil

    if not re.fullmatch("[a-z0-9_-]+", owner) or len(request.inputs) != 1:
        raise ProviderError("one declared checkpoint and a valid owner are required")
    verify_inputs(request, inputs)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    item = request.inputs[0]
    # The request digest prevents unrelated task uploads from sharing mutable identity.
    native_id = f"{owner}/ontogeny-input-{request.sha256[:20]}"
    shutil.copyfile(confined(inputs, item.path), directory / Path(item.path).name)
    write_json(
        directory / "dataset-metadata.json",
        {
            "id": native_id,
            "title": native_id.split("/")[1],
            "licenses": [{"name": "other"}],
        },
    )
    response = command(
        ["kaggle", "datasets", "create", "-p", str(directory), "--keep-tabular"], timeout=180
    )
    if native_id not in response or "being created" not in response:
        raise ProviderError(
            "private checkpoint dataset creation unconfirmed; reconcile before retry"
        )
    write_json(
        directory / "upload_receipt.json",
        {
            "dataset": native_id,
            "checkpoint_sha256": item.sha256,
            "request_sha256": request.sha256,
            "response": response,
        },
    )
    return native_id


def prepare(
    request: Request,
    directory: Path,
    *,
    owner: str = "sparklinghydraulics0",
    platform_seconds: int = 1200,
    dataset_sources: tuple[str, ...] = (),
    batch: tuple[tuple[Request, dict[str, str]], ...] = (),
) -> ExecutionContext:
    if request.schema_version != 2:
        raise ProviderError("historical requests require their original pinned worker")
    if not re.fullmatch("[a-z0-9_-]+", owner):
        raise ProviderError("invalid Kaggle owner")
    if not request.limits.wall_seconds + 120 <= platform_seconds <= 3600:
        raise ProviderError("platform budget must leave setup/export margin and be <= 3600 s")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ProviderError("commit executable changes before preparation")
    published = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], text=True, timeout=30
    ).split()[0]
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", request.source["ontogeny_sha"], published],
        check=False,
    ).returncode:
        raise ProviderError("execution source must be published")
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", item) for item in dataset_sources):
        raise ProviderError("invalid explicit Kaggle dataset source")
    if bool(request.inputs) != bool(dataset_sources):
        raise ProviderError(
            "checkpoint input requires explicit dataset sources, fresh jobs require none"
        )
    if batch:
        for _, check in batch:
            if set(check) != set(STATE_GROUPS):
                raise ProviderError("batch gates require all complete-state groups")
            for value in check.values():
                hexadecimal(value, 64)
    entries = batch or ((request, {}),)
    if entries[0][0].sha256 != request.sha256:
        raise ProviderError("batch principal request mismatch")
    if len({r.sha256 for r, _ in entries}) != len(entries) or any(
        r.source != request.source or r.runtime != request.runtime for r, _ in entries
    ):
        raise ProviderError("batch requests must be unique with identical source/runtime")
    if sum(r.limits.wall_seconds for r, _ in entries) + 120 > platform_seconds:
        raise ProviderError("batch worker budgets leave no setup/export margin")
    batch_identity = [{"request": r.to_dict(), "expected_state": check} for r, check in entries]
    identity = fingerprint(batch_identity) if batch else request.sha256
    native_id = f"{owner}/ontogeny-exec-{identity[:16]}"
    context = ExecutionContext("kaggle", native_id, "attempt-1")
    directory.mkdir(parents=True, exist_ok=False)
    encoded_context = canonical(vars(context)).decode()
    encoded_batch = canonical(batch_identity).decode()
    visible_devices = ",".join(str(i) for i in range(request.runtime["visible_cuda_devices"]))
    launcher = f"""import os, subprocess, sys
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = {visible_devices!r}
if not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    raise RuntimeError('Kaggle runtime required')
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA required; refusing CPU')
def run(args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True, timeout={platform_seconds})
root = Path('/tmp/ontogeny')
run(['git','clone','https://github.com/hieulc0/ontogeny.git',str(root)])
run(['git','checkout','--detach',{request.source["ontogeny_sha"]!r}],root)
run([sys.executable, '-m', 'pip', 'install', 'torch==2.6.0',
     '--index-url', 'https://download.pytorch.org/whl/cu124'])
run([sys.executable, '-m', 'pip', 'install', 'numpy==2.1.3'])
os.environ['PYTHONPATH'] = '/tmp/ontogeny/src'
import json
entries = json.loads({encoded_batch!r})
for index, entry in enumerate(entries):
    spec = entry['request']
    from hashlib import sha256
    identity = sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    request = Path('/kaggle/working') / ('request-' + identity + '.json')
    request.write_text(json.dumps(spec))
    inputs = Path('/kaggle/working/inputs') / identity
    output = (Path('/kaggle/working/result') if index == 0
              else Path('/kaggle/working/followups') / identity)
    run([sys.executable, '-m', 'ontogeny.execution.artifacts', '--request', str(request),
         '--source', '/kaggle/input', '--output', str(inputs)])
    run([sys.executable, '-m', 'ontogeny.execution.worker', '--request', str(request),
         '--context', {encoded_context!r},
         '--inputs', str(inputs), '--output', str(output)], Path('/tmp/ontogeny'))
    if entry['expected_state']:
        actual = json.loads((output / 'final_state.json').read_text())
        if actual != entry['expected_state']:
            raise RuntimeError('batch complete-state gate failed: ' + identity)

"""
    (directory / "launch.py").write_text(launcher)
    write_json(
        directory / "kernel-metadata.json",
        {
            "id": native_id,
            "title": native_id.split("/")[1],
            "code_file": "launch.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": list(dataset_sources),
        },
    )
    write_json(directory / "request.json", request.to_dict())
    write_json(
        directory / "submission.json",
        {
            "context": vars(context),
            "transport_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "platform_seconds": platform_seconds,
            "batch": batch_identity if batch else [],
        },
    )
    return context


def submit(directory: Path, ledger: Path, *, reserve_hours: float = 2.5) -> dict:
    if not 0 <= reserve_hours <= 30:
        raise ProviderError("invalid explicit reserve")
    request = Request.read(directory / "request.json")
    submission = json.loads((directory / "submission.json").read_text())
    native_id = submission["context"]["native_id"]
    path = ledger / (request.sha256 + ".json")
    with locked(ledger):
        if path.exists():
            previous = json.loads(path.read_text())
            # Refuse blind resubmission even after errors/timeouts. Reconciliation
            # is read-only; another push could create a duplicate version/run.
            return previous
        outstanding = [json.loads(p.read_text()) for p in ledger.glob("*.json")]
        if any(v["state"] not in ("COMPLETE", "ERROR", "CANCELLED") for v in outstanding):
            raise ProviderError("single-job controller has an unresolved/active reservation")
        quota = command(["kaggle", "quota", "--format", "json"])
        available = quota_hours(quota)
        required = reserve_hours + submission["platform_seconds"] / 3600 + 0.01
        if available < required:
            raise ProviderError(f"quota admission denied: {available:.2f} h < {required:.2f} h")
        record = {
            "request_sha256": request.sha256,
            "native_id": native_id,
            "state": "SUBMITTING",
            "reserved_seconds": submission["platform_seconds"],
            "reserve_hours": reserve_hours,
            "quota_before": json.loads(quota),
            "submitted_at": time.time(),
            "batch": submission.get("batch", []),
            "transport_sha": submission.get("transport_sha"),
        }
        journal(path, record)
        try:
            output = command(
                [
                    "kaggle",
                    "kernels",
                    "push",
                    "-p",
                    str(directory),
                    "--accelerator",
                    "NvidiaTeslaT4",
                    "-t",
                    str(submission["platform_seconds"]),
                ]
            )
            if not re.search(r"Kernel version \d+ successfully pushed", output):
                raise ProviderError("submission unconfirmed: " + output)
            version = re.search(r"Kernel version (\d+) successfully pushed", output)
            record.update(
                state="SUBMITTED",
                submission_output=output,
                version=int(version.group(1)) if version else None,
            )
        except Exception as error:
            record.update(state="UNKNOWN", submission_error=str(error))
            journal(path, record)
            raise ProviderError("submission uncertain; reconcile before any retry") from error
        journal(path, record)
        return record


def status(request: Request, ledger: Path) -> dict:
    with locked(ledger):
        path = ledger / (request.sha256 + ".json")
        record = json.loads(path.read_text())
        output = command(
            [
                "kaggle",
                "kernels",
                "status",
                record["native_id"] + (f"/{record['version']}" if record.get("version") else ""),
            ]
        )
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
        if not match:
            raise ProviderError("unrecognized native status; reservation remains held")
        native = match.group(1)
        # Duration-cap cancellation is a confirmed stopped job, not a successful result.
        state = (
            "CANCELLED"
            if (native == "CANCEL_ACKNOWLEDGED" and "was stopped because it exceeded" in output)
            else native
        )
        record.update(
            native_state=native, state=state, status_output=output, observed_at=time.time()
        )
        journal(path, record)
        return record


def collect(request: Request, ledger: Path, directory: Path) -> dict:
    record = status(request, ledger)
    if record["state"] not in ("COMPLETE", "ERROR", "CANCELLED"):
        raise ProviderError("remote job has not terminated")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        command(
            ["kaggle", "kernels", "output", record["native_id"], "-p", str(directory), "-q"],
            timeout=180,
        )
        root = directory / "result"
        receipt = verify_bundle(root, request)
        if record.get("batch"):
            receipts = {request.sha256: dict(receipt)}
            for entry in record["batch"]:
                child = Request.parse(entry["request"])
                child_root = (
                    root
                    if child.sha256 == request.sha256
                    else directory / "followups" / child.sha256
                )
                receipts[child.sha256] = verify_bundle(child_root, child)
                if entry["expected_state"] != json.loads(
                    (child_root / "final_state.json").read_text()
                ):
                    raise ProviderError("batch expected complete-state mismatch")
            receipt["batch_receipts"] = receipts
        receipt["quota_after"] = json.loads(command(["kaggle", "quota", "--format", "json"]))
        if record["state"] != "COMPLETE":
            raise ProviderError("provider terminal status is not COMPLETE")
        write_json(directory / "receipt.json", receipt)
        return receipt
    except Exception as error:
        write_json(
            directory / "collection_failure.json",
            {"kind": type(error).__name__, "message": str(error)},
        )
        raise
