"""Local-safe silent-allocation and readiness tests with a fake CLI, never Torch."""

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from ontogeny.execution import routing
from ontogeny.execution.cli import qualification_request
from ontogeny.execution.providers import colab


@pytest.mark.parametrize(
    "exit_code,text", [(0, ""), (1, ""), (1, "Allocation refused"), (0, "503 Service Unavailable")]
)
def test_missing_assignment_never_becomes_ready(tmp_path, monkeypatch, exit_code, text):
    session = colab.Session(tmp_path / "private")
    calls = []

    def invoke(args, timeout):
        calls.append(args)
        return subprocess.CompletedProcess(args, exit_code, text, "")

    monkeypatch.setattr(session, "invoke", invoke)
    with pytest.raises(colab.ColabError) as error:
        session.allocate()
    assert error.value.fallback_safe
    assert len(calls) == 1 and calls[0][0] == "new"


def test_real_probe_required_despite_success_message(tmp_path, monkeypatch):
    session = colab.Session(tmp_path / "private")
    calls = []

    def invoke(args, timeout):
        calls.append(args)
        if args[0] == "new":
            session.config.write_text(json.dumps({session.name: {"endpoint": "owned"}}))
        if args[0] == "stop":
            session.config.write_text("{}")
        return subprocess.CompletedProcess(args, 0, "Session READY", "")

    monkeypatch.setattr(session, "invoke", invoke)
    with pytest.raises(colab.ColabError, match="readiness_failed"):
        session.allocate()
    assert sum(c[0] == "new" for c in calls) == 1
    assert any(c[0] == "stop" for c in calls)


def test_nonce_bound_t4_proof(tmp_path, monkeypatch):
    session = colab.Session(tmp_path / "private")

    def invoke(args, timeout):
        if args[0] == "new":
            session.config.write_text(json.dumps({session.name: {"endpoint": "owned"}}))
        if args[0] == "download":
            Path(args[-1]).write_text(
                json.dumps({"nonce": session.nonce, "cuda": True, "gpu": "Tesla T4", "colab": True})
            )
        if args[0] == "stop":
            session.config.write_text("{}")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(session, "invoke", invoke)
    assert session.allocate()["cuda"]
    session.release()
    assert not session.assigned()


def test_allocation_timeout_is_not_safe_fallback(tmp_path, monkeypatch):
    session = colab.Session(tmp_path / "private")

    def timeout(*args):
        raise subprocess.TimeoutExpired("colab", 90)

    monkeypatch.setattr(session, "invoke", timeout)
    with pytest.raises(colab.ColabError, match="allocation_unknown") as error:
        session.allocate()
    assert not error.value.fallback_safe


def test_archive_cannot_escape_or_exceed_budget(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escaped", "data")
    with pytest.raises(ValueError):
        colab.extract_bundle(archive, tmp_path / "out", 100)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("large", "x" * 101)
    with pytest.raises(colab.ColabError, match="budget"):
        colab.extract_bundle(archive, tmp_path / "out2", 100)
    assert not (tmp_path / "escaped").exists()


def mock_publication(monkeypatch):
    monkeypatch.setattr(
        routing.subprocess,
        "check_output",
        lambda args, **kw: "" if "status" in args else "a" * 40 + "\trefs/heads/main",
    )
    monkeypatch.setattr(
        routing.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], 0)
    )


def test_unavailable_routes_once_and_is_idempotent(tmp_path, monkeypatch):
    mock_publication(monkeypatch)

    def denied(self):
        raise colab.ColabError("allocation_unavailable", fallback_safe=True)

    monkeypatch.setattr(colab.Session, "allocate", denied)
    monkeypatch.setattr(colab.Session, "release", lambda self: None)
    calls = []
    monkeypatch.setattr(routing.kaggle, "prepare", lambda *a, **kw: calls.append("prepare"))
    monkeypatch.setattr(routing.kaggle, "submit", lambda *a, **kw: {"state": "SUBMITTED"})
    request = qualification_request("a" * 40, "b" * 64)
    first = routing.run(request, tmp_path / "job", tmp_path / "ledger", tmp_path)
    assert first["selected_provider"] == "kaggle"
    assert routing.run(request, tmp_path / "job", tmp_path / "ledger", tmp_path) == first
    assert calls == ["prepare"]


def test_started_trial_failure_never_provider_hops(tmp_path, monkeypatch):
    mock_publication(monkeypatch)
    monkeypatch.setattr(colab.Session, "allocate", lambda self: {"cuda": True})
    monkeypatch.setattr(colab.Session, "release", lambda self: None)

    def broken(*args):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(colab, "execute", broken)
    monkeypatch.setattr(
        routing.kaggle, "prepare", lambda *a, **kw: pytest.fail("unsafe provider hopping")
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        routing.run(
            qualification_request("a" * 40, "b" * 64),
            tmp_path / "job",
            tmp_path / "ledger",
            tmp_path,
        )


def test_other_submitted_provider_job_blocks_new_allocation(tmp_path, monkeypatch):
    mock_publication(monkeypatch)
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "other.json").write_text('{"state":"SUBMITTED"}')
    monkeypatch.setattr(colab.Session, "allocate", lambda self: pytest.fail("duplicate allocation"))
    with pytest.raises(ValueError, match="another routing"):
        routing.run(qualification_request("a" * 40, "b" * 64), tmp_path / "job", ledger, tmp_path)


def test_wrong_gpu_proof_is_rejected_and_released(tmp_path, monkeypatch):
    session = colab.Session(tmp_path / "private")

    def invoke(args, timeout):
        if args[0] == "new":
            session.config.write_text(json.dumps({session.name: {"endpoint": "owned"}}))
        if args[0] == "download":
            Path(args[-1]).write_text(
                json.dumps(
                    {"nonce": session.nonce, "cuda": True, "gpu": "wrong GPU", "colab": True}
                )
            )
        if args[0] == "stop":
            session.config.write_text("{}")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(session, "invoke", invoke)
    with pytest.raises(colab.ColabError, match="readiness_failed"):
        session.allocate()
    assert not session.assigned()


def test_cleanup_reconciliation_never_restarts_trial(tmp_path, monkeypatch):
    request = qualification_request("a" * 40, "b" * 64)
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    record = {"selected_provider": "colab", "state": "UNKNOWN", "session": "owned"}
    (ledger / (request.sha256 + ".json")).write_text(json.dumps(record))
    private = tmp_path / "job/private-colab"
    private.mkdir(parents=True)
    (private / "ownership.json").write_text('{"session":"owned","nonce":"probe"}')
    calls = []
    monkeypatch.setattr(colab.Session, "release", lambda self: calls.append("release"))
    monkeypatch.setattr(
        colab.Session, "invoke", lambda *a, **kw: subprocess.CompletedProcess([], 0)
    )
    monkeypatch.setattr(routing, "verify_bundle", lambda *a: {"execution": "SUCCEEDED"})
    result = routing.reconcile(request, tmp_path / "job", ledger)
    assert result["state"] == "COMPLETE" and result["cleanup"] == "confirmed"
    assert calls == ["release"]


def test_cleanup_retry_accepts_verified_work_without_new_allocation(tmp_path, monkeypatch):
    mock_publication(monkeypatch)
    monkeypatch.setattr(colab.Session, "allocate", lambda self: {"cuda": True})
    releases = []

    def release(self):
        releases.append(1)
        if len(releases) == 1:
            raise colab.ColabError("cleanup_unconfirmed")

    monkeypatch.setattr(colab.Session, "release", release)

    def completed(session, request, inputs, directory):
        (directory / "result").mkdir(parents=True)
        return {"execution": "SUCCEEDED"}

    monkeypatch.setattr(colab, "execute", completed)
    monkeypatch.setattr(routing, "verify_bundle", lambda *a: {"execution": "SUCCEEDED"})
    request = qualification_request("a" * 40, "b" * 64)
    result = routing.run(request, tmp_path / "job", tmp_path / "ledger", tmp_path)
    assert result["state"] == "COMPLETE" and result["cleanup_retry"]
    assert len(releases) == 2
