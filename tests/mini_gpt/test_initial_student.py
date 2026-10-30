"""Local-safe configuration and provider admission tests; no neural operations."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.neural.initial_student import launch
from mini_gpt.model.config import ModelConfig
from ontogeny.execution.providers import colab


def test_initial_configuration_is_about_ten_million_parameters():
    root = Path(__file__).resolve().parents[2] / "experiments/neural/initial_student"
    cfg = ModelConfig.from_dict(json.loads((root / "model.json").read_text()))
    d, ff, vocab = cfg.d_model, cfg.d_ff, cfg.vocab_size
    block = 4 * d * d + 4 * d + 2 * d * ff + ff + d + 4 * d
    total = cfg.n_layers * block + 2 * vocab * d + 2 * d
    assert total == 10_054_016
    batches = json.loads((root / "batches.json").read_text())
    assert len(batches) == 2 and all(len(x["tokens"][0]) == 64 for x in batches)
    assert "adaptive-mini-gpt" not in launch.bootstrap("a" * 40, "/content/job")


def test_unavailable_colab_uses_kaggle_only_after_quota_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(
        launch.subprocess,
        "check_output",
        lambda args, **kw: "" if "status" in args else "a" * 40 + "\trefs/heads/main",
    )
    monkeypatch.setattr(
        launch.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], 0)
    )
    calls = []

    def unavailable(self):
        calls.append("colab")
        raise colab.ColabError("allocation_unavailable", fallback_safe=True)

    monkeypatch.setattr(colab.Session, "allocate", unavailable)
    monkeypatch.setattr(colab.Session, "release", lambda self: None)

    def command(args, **kwargs):
        calls.append("quota" if "quota" in args else "push")
        return '[{"resource":"GPU","remaining":"2.50h"}]'

    monkeypatch.setattr(launch.kaggle, "command", command)
    monkeypatch.setattr(sys, "argv", ["launch", "--directory", str(tmp_path / "run")])
    with pytest.raises(RuntimeError, match="quota"):
        launch.main()
    assert calls == ["colab", "quota"]
    record = json.loads((tmp_path / "run/controller.json").read_text())
    assert record["state"] == "FAILED" and record["colab_reason"] == "allocation_unavailable"
