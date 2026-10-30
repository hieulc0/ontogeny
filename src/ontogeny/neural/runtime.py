"""Thin organism factory over the internal canonical neural engine."""

import importlib
import random
from pathlib import Path
from typing import Any

from mini_gpt.runtime import configure_determinism


def initialize() -> tuple[Any, Any, Any]:
    torch, np = configure_determinism()
    from mini_gpt.learning.state import LearningState
    from mini_gpt.runtime import verify_import

    package_root = Path(__file__).resolve().parents[2] / "mini_gpt"
    for name in (
        "mini_gpt.model.transformer",
        "mini_gpt.model.attention",
        "mini_gpt.learning.state",
    ):
        verify_import(importlib.import_module(name), package_root)

    return torch, np, LearningState


def fresh_owner(seed: int, torch: Any, np: Any, Owner: Any) -> Any:
    from mini_gpt.model.config import ModelConfig
    from mini_gpt.model.transformer import MiniGPT
    from ontogeny.neural.policy import PolicyNetwork

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    network = PolicyNetwork(
        MiniGPT(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0))
    ).cuda()
    return Owner(
        network,
        lr=0.0005,
        external={"policy_rng": random.Random(seed)},
        metadata={
            "initialization": "fresh-no-historical-knowledge",
            "seed": seed,
            "individual_kind": "developmental_organism",
        },
    )
