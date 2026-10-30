"""Single-job routing: readiness, not allocation wording, determines provider choice."""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from ontogeny.execution.artifacts import verify_bundle, verify_inputs
from ontogeny.execution.contracts import Request
from ontogeny.execution.providers import colab, kaggle


def run(
    request: Request,
    directory: Path,
    ledger: Path,
    inputs: Path,
    *,
    provider: str = "auto",
    reserve_hours: float = 2.5,
) -> dict:
    if provider not in ("auto", "colab", "kaggle"):
        raise ValueError("unsupported provider preference")
    verify_inputs(request, inputs)
    if request.inputs and provider != "kaggle":
        raise ValueError("Colab checkpoint input transport must be qualified before auto resume")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("publish clean executable source before allocation")
    published = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/main"], text=True, timeout=30
    ).split()[0]
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", request.source["ontogeny_sha"], published],
        check=False,
    ).returncode:
        raise ValueError("execution source is unpublished")
    directory = directory.resolve()
    ledger = ledger.resolve()
    with kaggle.locked(ledger):
        path = ledger / (request.sha256 + ".json")
        if path.exists():
            return json.loads(path.read_text())
        if any(
            json.loads(p.read_text()).get("state")
            in ("ALLOCATING", "RUNNING", "UNKNOWN", "SUBMITTED")
            for p in ledger.glob("*.json")
            if p.name != "colab_cooldown.json"
        ):
            raise ValueError("another routing request requires reconciliation")
        directory.mkdir(parents=True, exist_ok=False)
        record: dict[str, Any] = {
            "request_sha256": request.sha256,
            "state": "ALLOCATING",
            "preference": provider,
        }
        kaggle.journal(path, record)
        cooldown = ledger / "colab_cooldown.json"
        unavailable = json.loads(cooldown.read_text()) if cooldown.exists() else {}
        eligible = time.time() >= unavailable.get("eligible_after", 0)
        session = None
        try:
            if provider != "kaggle" and eligible:
                session = colab.Session(directory / "private-colab")
                record["session"] = session.name
                kaggle.journal(path, record)
                try:
                    proof = session.allocate()
                except colab.ColabError as error:
                    record["colab_reason"] = str(error)
                    if not error.fallback_safe:
                        raise
                    kaggle.journal(
                        cooldown,
                        {
                            "eligible_after": time.time() + 1800,
                            "reason": str(error),
                            "observed_at": time.time(),
                        },
                    )
                    # Confirm release before another provider gets any workload.
                    session.release()
                    session = None
                    if provider == "colab":
                        raise
                else:
                    record.update(state="RUNNING", selected_provider="colab", readiness=proof)
                    kaggle.journal(path, record)
                    receipt = colab.execute(session, request, inputs, directory / "download")
                    # Export/hash verification happens before releasing the owned VM.
                    session.release()
                    session = None
                    record.update(state="COMPLETE", receipt=receipt)
                    kaggle.journal(path, record)
                    return record
            elif provider == "colab":
                raise colab.ColabError("colab_cooldown_active")
            if session is not None:
                raise RuntimeError("unresolved Colab lease prevents fallback")
            record["selected_provider"] = "kaggle"
            if not eligible:
                record["colab_reason"] = "cached_unavailable"
            kaggle.prepare(request, directory / "kaggle-job")
            submitted = kaggle.submit(
                directory / "kaggle-job", ledger / "kaggle", reserve_hours=reserve_hours
            )
            record.update(state="SUBMITTED", kaggle=submitted)
            kaggle.journal(path, record)
            return record
        except Exception as error:
            record.update(
                state="UNKNOWN"
                if isinstance(error, subprocess.TimeoutExpired)
                or str(error) in ("allocation_unknown", "cleanup_unconfirmed")
                else "FAILED",
                error_kind=type(error).__name__,
                reason=str(error),
            )
            if session is not None:
                try:
                    session.release()
                except Exception:
                    record.update(state="UNKNOWN", cleanup="unconfirmed")
                else:
                    bundle = directory / "download" / "result"
                    if record.get("selected_provider") == "colab" and bundle.exists():
                        try:
                            receipt = verify_bundle(bundle, request)
                        except Exception:
                            pass
                        else:
                            record.update(
                                state="COMPLETE",
                                cleanup="confirmed",
                                receipt=receipt,
                                cleanup_retry=True,
                            )
                            kaggle.journal(path, record)
                            return record
            kaggle.journal(path, record)
            raise


def reconcile(request: Request, directory: Path, ledger: Path) -> dict:
    """Recover completed work after a cleanup/transport error, never create another VM."""
    with kaggle.locked(ledger):
        path = ledger / (request.sha256 + ".json")
        record = json.loads(path.read_text())
        if record.get("selected_provider") != "colab":
            raise ValueError("use the Kaggle status/collect commands for its native journal")
        session = colab.Session.reopen(directory / "private-colab", record["session"])
        session.release()
        status = session.invoke(["status", "-s", session.name], 30)
        if status.returncode != 0 or session.assigned():
            raise colab.ColabError("cleanup_unconfirmed")
        receipt = verify_bundle(directory / "download" / "result", request)
        record.update(
            state="COMPLETE", cleanup="confirmed", receipt=receipt, reconciled_at=time.time()
        )
        kaggle.journal(path, record)
        return record
