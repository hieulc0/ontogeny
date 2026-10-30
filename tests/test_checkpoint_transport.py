"""Local-safe fake artifact tests; never import or deserialize a neural checkpoint."""

import json
import random
from types import SimpleNamespace

import pytest

from ontogeny.execution.cli import checkpoint_request, qualification_request
from ontogeny.execution.providers import colab, kaggle
from ontogeny.execution.state_identity import fingerprints


def parent_input(tmp_path):
    checkpoint = tmp_path / "opaque.pt"
    checkpoint.write_bytes(b"trusted-fake-full-state")
    request = checkpoint_request(qualification_request("a" * 40, "b" * 64), checkpoint, "train")
    inputs = tmp_path / "inputs/checkpoints"
    inputs.mkdir(parents=True)
    (inputs / "parent.pt").write_bytes(checkpoint.read_bytes())
    return request, inputs.parent


def test_resume_request_preserves_global_lineage(tmp_path):
    request, _ = parent_input(tmp_path)
    assert request.initialization == {"kind": "resume", "seed": None, "checkpoint": "parent"}
    assert request.world["episodes"] == 1 and request.limits.interactions == 22
    evaluated = checkpoint_request(request, tmp_path / "opaque.pt", "evaluate")
    assert evaluated.split == "development" and evaluated.world["episodes"] == 2
    assert evaluated.world["seed"] == 92823 and evaluated.source == request.source


def test_private_dataset_contains_only_declared_input(tmp_path, monkeypatch):
    request, inputs = parent_input(tmp_path)
    (inputs / "credential.env").write_text("must-not-upload")
    directory = tmp_path / "dataset"
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        assert set(p.name for p in directory.iterdir()) == {"dataset-metadata.json", "parent.pt"}
        metadata = json.loads((directory / "dataset-metadata.json").read_text())
        return f"Dataset is being created: https://www.kaggle.com/datasets/{metadata['id']}"

    monkeypatch.setattr(kaggle, "command", command)
    dataset = kaggle.upload_inputs(request, inputs, directory)
    assert dataset.startswith("sparklinghydraulics0/ontogeny-input-")
    assert "--public" not in calls[0] and "-u" not in calls[0]
    assert (directory / "upload_receipt.json").is_file()
    with pytest.raises(FileExistsError):
        kaggle.upload_inputs(request, inputs, directory)


def test_corrupt_upload_stops_before_provider_call(tmp_path, monkeypatch):
    request, inputs = parent_input(tmp_path)
    (inputs / "checkpoints/parent.pt").write_bytes(b"corrupt")
    monkeypatch.setattr(kaggle, "command", lambda *a, **kw: pytest.fail("uploaded corrupt data"))
    with pytest.raises(ValueError, match="hash/size"):
        kaggle.upload_inputs(request, inputs, tmp_path / "dataset")


def test_colab_upload_does_not_trust_silent_failure(tmp_path, monkeypatch):
    import subprocess

    request, inputs = parent_input(tmp_path)
    session = colab.Session(tmp_path / "private")
    calls = []

    def invoke(args, timeout):
        calls.append(args)
        if args[0] == "upload":
            return subprocess.CompletedProcess(args, 1, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(session, "invoke", invoke)
    with pytest.raises(colab.ColabError, match="upload_failed"):
        colab.execute(session, request, inputs, tmp_path / "download")
    assert sum(c[0] == "upload" for c in calls) == 1
    assert not any(c[0] == "download" for c in calls)
    bootstrap = (session.directory / "worker_bootstrap.py").read_text()
    assert "ontogeny.execution.artifacts" in bootstrap
    assert "staged-inputs" in bootstrap


def test_state_identity_tracks_rng_and_container_types():
    # Fake tensor/array types keep this test completely non-neural.
    torch = SimpleNamespace(Tensor=type("Tensor", (), {}))
    np = SimpleNamespace(ndarray=type("Array", (), {}))
    rng = random.Random(42)
    values = (
        {"weight": 1},
        {},
        None,
        {"rng": rng, "locations": {(2, 3), (1, 4)}},
        {},
        {},
        (),
        True,
        [True],
    )
    before = fingerprints(values, torch, np)
    assert fingerprints(values, torch, np) == before
    rng.random()
    after = fingerprints(values, torch, np)
    assert before["external"] != after["external"]
    assert before["model"] == after["model"]
    with pytest.raises(TypeError):
        fingerprints((object(), *values[1:]), torch, np)
