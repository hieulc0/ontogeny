"""Local-safe contract, transport and worker failure tests; no neural imports/execution."""

import copy
import json
import subprocess
import sys

import pytest

from ontogeny.execution import worker
from ontogeny.execution.artifacts import (
    stage_inputs,
    verify_bundle,
    verify_inputs,
    write_json,
)
from ontogeny.execution.cli import qualification_request
from ontogeny.execution.contracts import ContractError, ExecutionContext, Request
from ontogeny.execution.providers import kaggle
from ontogeny.neural.persistence import digest


@pytest.fixture
def specification():
    return qualification_request("a" * 40, "b" * 64).to_dict()


@pytest.fixture
def request_file(tmp_path, specification):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(specification))
    return path


def test_canonical_digest_and_duplicate_json_rejection(request_file, specification):
    original = Request.read(request_file)
    request_file.write_text(json.dumps(dict(reversed(list(specification.items())))))
    assert Request.read(request_file).sha256 == original.sha256
    request_file.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ContractError, match="duplicate"):
        Request.read(request_file)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"unknown": 1},
        {"operation": "arbitrary.shell"},
        {"run_id": "../../escape"},
        {"protocol_sha256": "missing"},
        {"initialization": {"kind": "fresh", "seed": 0, "checkpoint": "old"}},
        {"targets": {"kind": "science", "criterion": "PASS"}},
    ],
)
def test_bad_requests_reject(specification, change):
    specification.update(change)
    with pytest.raises(ValueError):
        Request.parse(specification)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("limits", "interactions", True),
        ("limits", "updates", 0),
        ("limits", "wall_seconds", 3601),
        ("limits", "interactions", 9),
        ("world", "episodes", 3),
        ("runtime", "visible_cuda_devices", False),
        ("runtime", "gpu", "CPU"),
        ("source", "ontogeny_sha", "short"),
        ("objective", "gamma", 0.9),
    ],
)
def test_nested_contract_failures(specification, section, key, value):
    specification[section][key] = value
    with pytest.raises(ValueError):
        Request.parse(specification)


def checkpoint_request(specification, path):
    value = copy.deepcopy(specification)
    value["operation"] = "train"
    value["split"] = "train"
    value["initialization"] = {"kind": "resume", "seed": None, "checkpoint": "parent"}
    value["inputs"] = [
        {
            "name": "parent",
            "path": "logical/parent.pt",
            "sha256": digest(path),
            "size_bytes": path.stat().st_size,
            "role": "checkpoint",
            "trusted": True,
        }
    ]
    return Request.parse(value)


def test_input_transport_hashes_and_confinement(tmp_path, specification):
    source = tmp_path / "source"
    source.mkdir()
    original = source / "parent.pt"
    original.write_bytes(b"fake-full-state")
    req = checkpoint_request(specification, original)
    target = tmp_path / "staged"
    stage_inputs(req, source, target)
    verify_inputs(req, target)
    (target / "logical/parent.pt").write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="hash"):
        verify_inputs(req, target)
    data = req.to_dict()
    data["inputs"][0]["path"] = "../escape"
    with pytest.raises(ContractError):
        Request.parse(data)
    (source / "duplicate").mkdir()
    (source / "duplicate/parent.pt").write_bytes(original.read_bytes())
    with pytest.raises(ContractError, match="one exact"):
        stage_inputs(req, source, tmp_path / "ambiguous")


def test_symlink_input_cannot_escape(tmp_path, specification):
    original = tmp_path / "parent.pt"
    original.write_bytes(b"trusted")
    req = checkpoint_request(specification, original)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "logical").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ContractError, match="escapes"):
        verify_inputs(req, inputs)


def fake_task(request, context, upstream, inputs, output, cancel_file):
    (output / "predictions.jsonl").write_text("{}\n")
    (output / "learning.jsonl").write_text("{}\n")
    (output / "checkpoint.pt").write_bytes(b"opaque-fake-checkpoint")
    write_json(
        output / "checkpoint_manifest.json",
        {
            "schema_version": 1,
            "checkpoints": [{"path": "checkpoint.pt", "sha256": digest(output / "checkpoint.pt")}],
        },
    )
    write_json(
        output / "metrics.json", {"updates": 1, "interactions": 10, "peak_torch_memory_bytes": 1}
    )


def test_worker_publishes_verifiable_bundle_with_science_not_run(
    tmp_path, request_file, monkeypatch
):
    monkeypatch.setattr(worker, "remote_runtime", lambda *args: {"mock": True})
    monkeypatch.setattr(worker, "execute", fake_task)
    req = Request.read(request_file)
    output = tmp_path / "result"
    result = worker.run(
        request_file,
        ExecutionContext("kaggle", "owner/id", "attempt-1"),
        tmp_path,
        tmp_path,
        output,
    )
    assert result["execution"] == "SUCCEEDED" and result["science"] == "NOT_RUN"
    assert verify_bundle(output, req)["execution"] == "SUCCEEDED"
    (output / "checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(ContractError):
        verify_bundle(output, req)


@pytest.mark.parametrize("failure", [TimeoutError, InterruptedError, RuntimeError])
def test_worker_preserves_failure_and_partial_checkpoint(
    tmp_path, request_file, monkeypatch, failure
):
    monkeypatch.setattr(worker, "remote_runtime", lambda *args: {"mock": True})

    def failed(*args):
        (args[4] / "pending.pt").write_bytes(b"opaque-pending")
        raise failure("injected failure")

    monkeypatch.setattr(worker, "execute", failed)
    output = tmp_path / "result"
    result = worker.run(
        request_file,
        ExecutionContext("kaggle", "owner/id", "attempt-1"),
        tmp_path,
        tmp_path,
        output,
    )
    assert result["execution"] == ("CANCELLED" if failure is InterruptedError else "FAILED")
    assert (output / "pending.pt").read_bytes() == b"opaque-pending"
    assert json.loads((output / "failure.json").read_text())["kind"] == failure.__name__
    with pytest.raises(ContractError):
        verify_bundle(output, Request.read(request_file))


def test_missing_outputs_never_become_success(tmp_path, request_file, monkeypatch):
    monkeypatch.setattr(worker, "remote_runtime", lambda *args: {"mock": True})
    monkeypatch.setattr(worker, "execute", lambda *args: None)
    result = worker.run(
        request_file,
        ExecutionContext("kaggle", "owner/id", "attempt-1"),
        tmp_path,
        tmp_path,
        tmp_path / "result",
    )
    assert result["execution"] == "FAILED"


def test_local_refusal_before_torch_import(tmp_path, request_file, monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    result = worker.run(
        request_file,
        ExecutionContext("kaggle", "owner/id", "attempt-1"),
        tmp_path,
        tmp_path,
        tmp_path / "result",
    )
    assert result["execution"] == "FAILED" and "no local fallback" in result["reason"]
    assert "torch" not in sys.modules


def test_unindexed_artifact_rejected(tmp_path, request_file, monkeypatch):
    monkeypatch.setattr(worker, "remote_runtime", lambda *args: {"mock": True})
    monkeypatch.setattr(worker, "execute", fake_task)
    output = tmp_path / "result"
    worker.run(
        request_file,
        ExecutionContext("kaggle", "owner/id", "attempt-1"),
        tmp_path,
        tmp_path,
        output,
    )
    (output / "unindexed").write_text("extra")
    with pytest.raises(ContractError, match="unindexed"):
        verify_bundle(output, Request.read(request_file))


def prepared(tmp_path, request_file):
    directory = tmp_path / "job"
    directory.mkdir()
    request = Request.read(request_file)
    write_json(directory / "request.json", request.to_dict())
    write_json(
        directory / "submission.json",
        {"context": {"native_id": "owner/id"}, "platform_seconds": 1200},
    )
    return request, directory, tmp_path / "ledger"


def test_quota_unknown_and_insufficient_reject_without_submit(tmp_path, request_file, monkeypatch):
    req, directory, ledger = prepared(tmp_path, request_file)
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        return '[{"resource":"GPU","remaining":"2.70h"}]'

    monkeypatch.setattr(kaggle, "command", execute)
    with pytest.raises(kaggle.ProviderError, match="quota admission"):
        kaggle.submit(directory, ledger)
    assert len(calls) == 1
    for text in ["[]", "{}", '[{"resource":"GPU","remaining":"unknown"}]']:
        with pytest.raises(kaggle.ProviderError):
            kaggle.quota_hours(text)


def test_submission_idempotent_across_calls(tmp_path, request_file, monkeypatch):
    req, directory, ledger = prepared(tmp_path, request_file)
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        return (
            '[{"resource":"GPU","remaining":"3.06h"}]'
            if "quota" in args
            else "Kernel version 1 successfully pushed."
        )

    monkeypatch.setattr(kaggle, "command", execute)
    first = kaggle.submit(directory, ledger)
    assert kaggle.submit(directory, ledger) == first
    assert sum("push" in c for c in calls) == 1


def test_ambiguous_timeout_never_resubmits(tmp_path, request_file, monkeypatch):
    req, directory, ledger = prepared(tmp_path, request_file)
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        if "quota" in args:
            return '[{"resource":"GPU","remaining":"3.06h"}]'
        raise subprocess.TimeoutExpired(args, 60)

    monkeypatch.setattr(kaggle, "command", execute)
    with pytest.raises(kaggle.ProviderError, match="uncertain"):
        kaggle.submit(directory, ledger)
    assert kaggle.submit(directory, ledger)["state"] == "UNKNOWN"
    assert sum("push" in c for c in calls) == 1


def test_collection_does_not_accept_exit_zero_without_artifacts(
    tmp_path, request_file, monkeypatch
):
    req, directory, ledger = prepared(tmp_path, request_file)
    ledger.mkdir()
    write_json(ledger / (req.sha256 + ".json"), {"native_id": "owner/id", "state": "SUBMITTED"})
    monkeypatch.setattr(
        kaggle,
        "command",
        lambda args, **kwargs: "KernelWorkerStatus.COMPLETE" if "status" in args else "",
    )
    destination = tmp_path / "download"
    with pytest.raises(FileNotFoundError):
        kaggle.collect(req, ledger, destination)
    assert (destination / "collection_failure.json").exists()


def test_validation_survives_python_optimization(tmp_path, request_file):
    data = json.loads(request_file.read_text())
    data["limits"]["updates"] = 0
    request_file.write_text(json.dumps(data))
    code = (
        "import sys\nfrom pathlib import Path\n"
        "from ontogeny.execution.contracts import Request\nRequest.read(Path(sys.argv[1]))"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code, str(request_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "ContractError" in result.stderr


def test_training_cannot_access_final_split(specification):
    specification["operation"] = "train"
    specification["split"] = "test"
    with pytest.raises(ContractError, match="split"):
        Request.parse(specification)


def test_runtime_gate_cannot_be_bypassed_by_colab_context(tmp_path, request_file, monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    result = worker.run(
        request_file,
        ExecutionContext("colab", "session", "attempt-1"),
        tmp_path,
        tmp_path,
        tmp_path / "result",
    )
    assert result["execution"] == "FAILED" and "unverified remote" in result["reason"]
    assert "torch" not in sys.modules


def test_hard_timeout_reaps_task_process(tmp_path, specification, monkeypatch):
    original = subprocess.Popen
    children = []

    def sleeper(*args, **kwargs):
        process = original([sys.executable, "-c", "import time; time.sleep(20)"], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(worker.subprocess, "Popen", sleeper)
    readings = iter((0.0, 200.0))
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(readings))
    output = tmp_path / "task"
    output.mkdir()
    with pytest.raises(TimeoutError):
        worker.execute(
            Request.parse(specification),
            ExecutionContext("kaggle", "id", "attempt-1"),
            tmp_path,
            tmp_path,
            output,
            None,
        )
    assert children and children[0].poll() is not None


def test_counter_overrun_fails_before_success_publication(tmp_path, request_file, monkeypatch):
    monkeypatch.setattr(worker, "remote_runtime", lambda *args: {"mock": True})

    def overrun(*args):
        fake_task(*args)
        path = args[4] / "metrics.json"
        path.unlink()
        write_json(path, {"updates": 999, "interactions": 10, "peak_torch_memory_bytes": 1})

    monkeypatch.setattr(worker, "execute", overrun)
    result = worker.run(
        request_file,
        ExecutionContext("kaggle", "id", "attempt-1"),
        tmp_path,
        tmp_path,
        tmp_path / "result",
    )
    assert result["execution"] == "FAILED" and "counters" in result["reason"]


def test_audit_branches_count_toward_sensory_cost(tmp_path):
    from ontogeny.execution.tasks import physical_observation_tiles

    (tmp_path / "resumed").mkdir()
    (tmp_path / "resumed/predictions.jsonl").write_text('{"observation_tiles":25}\n')
    (tmp_path / "reference_experience.jsonl").write_text(
        json.dumps({"observation": {"tiles": [[0, 0, "empty"]] * 9}}) + "\n"
    )
    assert physical_observation_tiles(tmp_path, 100) == 134
