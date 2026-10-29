import sys
from pathlib import Path

import pytest

from ontogeny.neural.adapter import require_remote
from ontogeny.neural.encoding import encode
from ontogeny.neural.persistence import UPSTREAM_SHA, digest
from ontogeny.world.actions import Action
from ontogeny.world.environment import CellWorld


def test_remote_guard(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    with pytest.raises(RuntimeError, match="Kaggle"):
        require_remote()


def test_encoding_is_bounded_observable_only():
    observation = CellWorld().observe()
    a, b = encode(observation, Action.NORTH), encode(observation, Action.REST)
    assert a[:-1] == b[:-1] and a[-1] != b[-1]
    assert len(a) == 13 and all(0 <= x < 256 for x in a)
    assert len(UPSTREAM_SHA) == 40


def test_hash(tmp_path: Path):
    p = tmp_path / "checkpoint"
    p.write_bytes(b"abc")
    assert digest(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_import_did_not_load_torch():
    assert "torch" not in sys.modules


def test_source_revision_and_dirty_tree_reject(monkeypatch, tmp_path):
    import ontogeny.neural.persistence as persistence

    responses = iter([UPSTREAM_SHA + "\n", " M modified.py\n"])
    monkeypatch.setattr(persistence.subprocess, "check_output", lambda *a, **k: next(responses))
    with pytest.raises(RuntimeError, match="clean"):
        persistence.verify_source(tmp_path)
    responses = iter(["0" * 40, ""])
    with pytest.raises(RuntimeError, match="revision"):
        persistence.verify_source(tmp_path)


def test_import_path_rejects_wrong_module(tmp_path):
    from types import ModuleType

    from ontogeny.neural.persistence import verify_import

    module = ModuleType("fake")
    module.__file__ = str(tmp_path / "wrong" / "model.py")
    with pytest.raises(RuntimeError, match="outside"):
        verify_import(tmp_path, module)
    module.__file__ = str(tmp_path / "src" / "adaptive_mini_gpt" / "model.py")
    verify_import(tmp_path, module)
