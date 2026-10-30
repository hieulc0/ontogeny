"""Explicit remote CUDA admission, independent of any organism or world."""

import importlib
import os
from pathlib import Path
from typing import Any


def require_remote() -> str:
    for platform, marker, root in (
        ("kaggle", "KAGGLE_KERNEL_RUN_TYPE", "/kaggle/working"),
        ("colab", "COLAB_RELEASE_TAG", "/content"),
    ):
        if os.environ.get(marker) and Path(root).is_dir():
            return platform
    raise RuntimeError("Kaggle or Colab CUDA required; local neural execution refused")


def require_remote_cuda() -> Any:
    require_remote()
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    return torch


def configure_determinism() -> tuple[Any, Any]:
    torch = require_remote_cuda()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python startup")
    if torch.__version__ != "2.6.0+cu124":
        raise RuntimeError("qualified Torch 2.6.0+cu124 required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, importlib.import_module("numpy")
