"""Source ancestry, not an external runtime dependency."""

from pathlib import Path
from types import ModuleType

SOURCE_REPOSITORY = "https://github.com/hieulc0/adaptive-mini-gpt"
SOURCE_REVISION = "87e8928c440fecc4f3a6f6c5ea3c869a6a8537e6"
HISTORICAL_REVISION = "8567836fb2c8c915d442b7c2fbedb4fd9b30396d"
ARCHITECTURE = "dense-causal-transformer"
CHECKPOINT_VERSION = 2


def verify_import(module: ModuleType, package_root: Path) -> None:
    if module.__file__ is None or not Path(module.__file__).resolve().is_relative_to(
        package_root.resolve()
    ):
        raise RuntimeError("neural module imported outside the verified internal package")
