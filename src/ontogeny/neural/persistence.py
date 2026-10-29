import hashlib
import subprocess
from pathlib import Path
from types import ModuleType

UPSTREAM_SHA = "8567836fb2c8c915d442b7c2fbedb4fd9b30396d"


def verify_source(root: Path) -> None:
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True)
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if sha.strip() != UPSTREAM_SHA or dirty.strip():
        raise RuntimeError("requires exact clean upstream revision")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_import(root: Path, module: ModuleType) -> None:
    if module.__file__ is None or not Path(module.__file__).resolve().is_relative_to(
        (root / "src" / "adaptive_mini_gpt").resolve()
    ):
        raise RuntimeError("imported neural module is outside verified source")
