"""Local-only mock of ambiguous provider allocation; never invokes a provider or Torch."""

import json
import sys
from pathlib import Path

import pytest

from experiments.neural.concept_acquisition import launch


def test_ambiguous_allocation_remains_unknown(tmp_path, monkeypatch):
    class UnknownSession:
        name = "owned-session"
        nonce = "nonce"

        def __init__(self, directory):
            self.directory = Path(directory)
            self.directory.mkdir(parents=True)

        def allocate(self):
            raise launch.colab.ColabError("allocation_unknown")

        def release(self):
            pytest.fail("no local assignment does not prove remote cleanup")

    directory = tmp_path / "attempt"
    monkeypatch.setattr(sys, "argv", ["launch", "--directory", str(directory)])
    monkeypatch.setattr(launch.colab, "Session", UnknownSession)
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *args, **kwargs: "abc")
    with pytest.raises(launch.colab.ColabError, match="allocation_unknown"):
        launch.main()
    record = json.loads((directory / "controller.json").read_text())
    assert record["state"] == "UNKNOWN"
    assert record["source"] == "abc"
