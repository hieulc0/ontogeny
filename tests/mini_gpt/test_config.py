"""Local-safe configuration and packaging contracts; never import tensor modules."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from mini_gpt.model.config import ModelConfig


@pytest.mark.parametrize(
    "field,value",
    [
        ("d_model", 0),
        ("n_heads", True),
        ("n_layers", -1),
        ("dropout", 1.0),
        ("pos_encoding", "unknown"),
        ("bias", 1),
    ],
)
def test_invalid_config_fails(field, value):
    with pytest.raises(ValueError):
        ModelConfig(**{field: value})


def test_config_round_trip_and_no_permanent_width_ceiling():
    config = ModelConfig(d_model=512, n_heads=8, n_layers=12, d_ff=2048, pos_encoding="rope")
    assert ModelConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="even"):
        ModelConfig(d_model=12, n_heads=4, pos_encoding="rope")
    with pytest.raises(TypeError):
        ModelConfig.from_dict({"adaptive_enabled": True})


def test_public_package_imports_do_not_load_neural_frameworks():
    script = (
        "import sys,mini_gpt,ontogeny\n"
        "from mini_gpt.model.config import ModelConfig\n"
        "from mini_gpt.checkpoint.identity import file_digest\n"
        "assert 'torch' not in sys.modules\n"
        "assert not any(n.startswith('adaptive_mini_gpt') for n in sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_core_has_no_organism_or_external_repository_imports():
    root = Path(__file__).resolve().parents[2] / ".." / "src/mini_gpt"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = (
                [node.module]
                if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert all(
                not name or not name.startswith(("ontogeny", "adaptive_mini_gpt")) for name in names
            )


def test_core_has_no_external_revision_or_import_compatibility_layer():
    root = Path(__file__).resolve().parents[2] / ".." / "src/mini_gpt"
    assert not (root / "provenance.py").exists()
    assert not (root / "checkpoint/legacy.py").exists()
    for path in root.rglob("*.py"):
        text = path.read_text()
        assert "adaptive_mini_gpt" not in text
        assert "adaptive-mini-gpt" not in text
