"""Provider-neutral orchestration with a currently Kaggle-qualified remote boundary."""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ontogeny.execution.artifacts import (
    confined,
    publish_index,
    verify_bundle,
    verify_inputs,
    write_json,
)
from ontogeny.execution.contracts import ContractError, ExecutionContext, Request
from ontogeny.neural.persistence import digest


def remote_runtime(request: Request, context: ExecutionContext, upstream: Path) -> dict[str, Any]:
    if context.provider != "kaggle" or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ContractError("this release admits only real Kaggle CUDA runtimes; no local fallback")
    source = Path(__file__).resolve().parents[3]
    sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True, timeout=30
    )
    if sha != request.source["ontogeny_sha"] or dirty.strip():
        raise ContractError("Ontogeny source identity mismatch or dirty checkout")
    from ontogeny.neural.runtime import initialize

    torch, _, _ = initialize(upstream)
    actual = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "visible_cuda_devices": torch.cuda.device_count(),
    }
    if actual != request.runtime:
        raise ContractError(f"runtime incompatibility: {actual}")
    hardware = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        .strip()
        .splitlines()
    )
    return {
        **actual,
        "python": platform.python_version(),
        "source": request.source,
        "packages": sorted(
            (d.metadata["Name"], d.version)
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allocated_gpu_count": len(hardware),
        "physical_gpu_inventory": hardware,
        "kernel_time_seconds": None,
    }


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def execute(
    request: Request,
    context: ExecutionContext,
    upstream: Path,
    inputs: Path,
    output: Path,
    cancel_file: Path | None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "ontogeny.execution.tasks",
        "--request",
        str(output / "request.json"),
        "--upstream",
        str(upstream),
        "--inputs",
        str(inputs),
        "--output",
        str(output),
        "--context",
        json.dumps(asdict(context)),
    ]
    deadline = time.monotonic() + request.limits.wall_seconds
    with (output / "task.log").open("x") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                if cancel_file is not None and cancel_file.exists():
                    raise InterruptedError("external cancellation requested")
                if time.monotonic() >= deadline:
                    raise TimeoutError("hard worker wall-time budget exhausted")
                if (
                    sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                    > request.limits.output_bytes
                ):
                    raise ContractError("monitored output budget exceeded")
                time.sleep(0.1)
            if process.returncode != 0:
                raise RuntimeError(f"task exited {process.returncode}; see task.log")
        finally:
            stop(process)


def run(
    request_path: Path,
    context: ExecutionContext,
    upstream: Path,
    inputs: Path,
    output: Path,
    cancel_file: Path | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    request = None
    execution, reason = "FAILED", ""
    try:
        request = Request.read(request_path)
        write_json(output / "request.json", request.to_dict())
        verify_inputs(request, inputs)
        if cancel_file is not None and cancel_file.exists():
            raise InterruptedError("cancelled before remote admission")
        runtime = remote_runtime(request, context, upstream)
        write_json(output / "runtime.json", runtime)
        execute(request, context, upstream, inputs, output, cancel_file)
        if not all(
            (output / name).is_file()
            for name in request.required_outputs
            if name != "execution.json"
        ):
            raise ContractError("task omitted required output")
        outcome = json.loads((output / "metrics.json").read_text())
        if (
            outcome["updates"] > request.limits.updates
            or outcome["interactions"] > request.limits.interactions
            or outcome["peak_torch_memory_bytes"] > request.limits.torch_memory_bytes
        ):
            raise ContractError("task exceeded declared counters or Torch memory budget")
        checkpoint_index = json.loads((output / "checkpoint_manifest.json").read_text())
        if (
            set(checkpoint_index) != {"schema_version", "checkpoints"}
            or checkpoint_index["schema_version"] != 1
            or not checkpoint_index["checkpoints"]
        ):
            raise ContractError("incomplete checkpoint publication")
        for record in checkpoint_index["checkpoints"]:
            if digest(confined(output, record["path"])) != record["sha256"]:
                raise ContractError("checkpoint publication hash mismatch")
        if any(p.is_symlink() for p in output.rglob("*")):
            raise ContractError("task published a symlink")
        if (
            sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) + 16384
            > request.limits.output_bytes
        ):
            raise ContractError("output budget lacks publication headroom")
        execution = "SUCCEEDED"
    except Exception as error:
        reason = str(error)
        if isinstance(error, InterruptedError):
            execution = "CANCELLED"
        write_json(
            output / "failure.json",
            {"kind": type(error).__name__, "message": reason, "traceback": traceback.format_exc()},
        )
    write_json(
        output / "execution.json",
        {
            "schema_version": 1,
            **asdict(context),
            "execution": execution,
            "worker_wall_seconds": time.monotonic() - started,
        },
    )
    result = {
        "schema_version": 1,
        "request_sha256": request.sha256 if request else None,
        "execution": execution,
        "science": "NOT_RUN",
        "reason": reason,
    }
    write_json(output / "result.json", result)
    publish_index(output)
    if execution == "SUCCEEDED" and request:
        try:
            verify_bundle(output, request)
        except Exception:
            # Never turn a malformed successful bundle into an accepted result.
            # Preserve the immutable output for diagnosis and fail the worker exit.
            raise ContractError("published bundle failed verification") from None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()
    context = ExecutionContext(**json.loads(args.context))
    result = run(args.request, context, args.upstream, args.inputs, args.output, args.cancel_file)
    raise SystemExit(0 if result["execution"] == "SUCCEEDED" else 1)


if __name__ == "__main__":
    main()
