"""Explicit trusted import of the historical dense-only full-object state format.

Class remapping is confined to this conversion operation. No old namespace is installed
or aliased for normal loading. Unsupported neural structures stay with their old code.
"""

import json
import pickle
from pathlib import Path
from types import ModuleType
from typing import Any

from mini_gpt.checkpoint.identity import file_digest
from mini_gpt.model.config import ModelConfig
from mini_gpt.runtime import require_remote_cuda

REFERENCE = json.loads(Path(__file__).with_name("reference.json").read_text())
HISTORICAL_REVISION = REFERENCE["historical_revision"]
SOURCE_REVISION = REFERENCE["source_revision"]


class DenseCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        from mini_gpt.model.attention import CausalSelfAttention
        from mini_gpt.model.transformer import FeedForward, MiniGPT, TransformerBlock

        mapping = {
            ("adaptive_mini_gpt.model.config", "ModelConfig"): ModelConfig,
            ("adaptive_mini_gpt.model.attention", "CausalSelfAttention"): CausalSelfAttention,
            ("adaptive_mini_gpt.model.transformer", "MiniGPT"): MiniGPT,
            ("adaptive_mini_gpt.model.transformer", "TransformerBlock"): TransformerBlock,
            ("adaptive_mini_gpt.model.transformer", "FeedForward"): FeedForward,
        }
        if (module, name) in mapping:
            return mapping[module, name]
        if module.startswith("adaptive_mini_gpt"):
            raise ValueError(f"unsupported historical class: {module}.{name}")
        return super().find_class(module, name)


def import_dense_checkpoint(
    path: Path, *, expected_sha256: str, source_revision: str, trusted: bool
) -> Any:
    torch = require_remote_cuda()
    if trusted is not True or source_revision not in (HISTORICAL_REVISION, SOURCE_REVISION):
        raise ValueError("explicit trust and an audited source revision are required")
    if file_digest(path) != expected_sha256:
        raise ValueError("historical checkpoint hash mismatch")
    from mini_gpt.learning.state import LearningState
    from mini_gpt.model.transformer import MiniGPT

    reader: Any = ModuleType("dense_checkpoint_reader")
    reader.Unpickler = DenseCheckpointUnpickler
    reader.load = pickle.load
    snapshot = torch.load(path, weights_only=False, pickle_module=reader)
    if snapshot.get("version") != 1:
        raise ValueError("unsupported historical checkpoint format")
    model, optimizer, _, _, metadata = snapshot["roots"]
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("only audited AdamW state is supported")
    brains = [module for module in model.modules() if isinstance(module, MiniGPT)]
    if len(brains) != 1:
        raise ValueError("conversion requires exactly one dense MiniGPT backbone")
    brain = brains[0]
    config = brain.config
    if (
        getattr(config, "adaptive_enabled", False)
        or getattr(config, "mod_enabled", False)
        or getattr(config, "trainable_token_ids", None)
        or getattr(config, "domain_trigger_token_ids", None)
        or getattr(brain, "moe_head", None) is not None
        or len(getattr(brain, "skip_connections", {}))
    ):
        raise ValueError("adaptive/gated/hook-bearing topology requires its original source")
    normalized = ModelConfig(
        **{name: getattr(config, name) for name in ModelConfig.__dataclass_fields__}
    )
    config.__dict__.clear()
    config.__dict__.update(vars(normalized))
    for block in brain.blocks:
        if (
            getattr(block, "moe", None) is not None
            or getattr(block, "mod_router", None) is not None
        ):
            raise ValueError("historical block contains an unsupported neural extension")
        for name in ("moe", "mod_router"):
            if hasattr(block, name):
                delattr(block, name)
    for name in (
        "moe_head",
        "skip_connections",
        "last_aux_info",
        "_embedding_grad_mask",
        "_wte_hook_registered",
    ):
        if hasattr(brain, name):
            delattr(brain, name)
    metadata.setdefault("checkpoint_imports", []).append(
        {
            "parent_sha256": expected_sha256,
            "source_revision": source_revision,
            "format_from": 1,
            "format_to": LearningState.VERSION,
            "method": "dense_class_mapping_without_parameter_reinitialization",
        }
    )
    snapshot["version"] = LearningState.VERSION
    # Restoring the joint graph preserves optimizer/model/external aliases and all RNGs.
    owner = LearningState(brain if model is brain else model)
    owner.restore(snapshot)
    owner.snapshot()  # Reject unsupported hooks before permitting publication.
    return owner


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--trusted", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from mini_gpt.runtime import configure_determinism

    configure_determinism()
    owner = import_dense_checkpoint(
        args.source,
        expected_sha256=args.sha256,
        source_revision=args.source_revision,
        trusted=args.trusted,
    )
    owner.save(args.output)


if __name__ == "__main__":
    main()
