"""Verified remote dependency boundary for diagnostic experiments."""

import importlib
import os
import random
import sys
from pathlib import Path
from typing import Any

from ontogeny.neural.adapter import require_remote
from ontogeny.neural.persistence import verify_import, verify_source


def initialize(root: Path) -> tuple[Any, Any, Any]:
    require_remote()
    verify_source(root)
    sys.path.insert(0, str(root.resolve() / "src"))
    torch, np = importlib.import_module("torch"), importlib.import_module("numpy")
    if not torch.cuda.is_available() or torch.__version__ != "2.6.0+cu124":
        raise RuntimeError("pinned Kaggle CUDA runtime required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("numerical policy must precede Python startup")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name in (
        "model.config",
        "model.transformer",
        "training.learning_state",
        "training.optimizer_contract",
    ):
        verify_import(root, importlib.import_module("adaptive_mini_gpt." + name))
    Owner = importlib.import_module("adaptive_mini_gpt.training.learning_state").LearningState
    return torch, np, Owner


def fresh_owner(seed: int, torch: Any, np: Any, Owner: Any) -> Any:
    from ontogeny.neural.policy import PolicyNetwork

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = importlib.import_module("adaptive_mini_gpt.model.config").ModelConfig
    model = importlib.import_module("adaptive_mini_gpt.model.transformer").MiniGPT
    network = PolicyNetwork(
        model(config(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0))
    ).cuda()
    return Owner(
        network,
        lr=0.0005,
        external={"policy_rng": random.Random(seed)},
        metadata={"initialization": "fresh-no-historical-knowledge", "seed": seed},
    )
