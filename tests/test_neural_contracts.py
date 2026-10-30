import sys
from pathlib import Path

import pytest

from mini_gpt.provenance import SOURCE_REVISION
from ontogeny.neural.adapter import require_remote
from ontogeny.neural.encoding import encode_policy
from ontogeny.neural.persistence import digest
from ontogeny.world.environment import CellWorld


def test_remote_guard(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    with pytest.raises(RuntimeError, match="Kaggle"):
        require_remote()


def test_encoding_is_bounded_observable_only():
    observation = CellWorld().observe()
    encoded = encode_policy(observation)
    assert len(encoded) == 29 and all(0 <= x < 256 for x in encoded)
    assert len(SOURCE_REVISION) == 40


def test_hash(tmp_path: Path):
    p = tmp_path / "checkpoint"
    p.write_bytes(b"abc")
    assert digest(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_import_did_not_load_torch():
    assert "torch" not in sys.modules
