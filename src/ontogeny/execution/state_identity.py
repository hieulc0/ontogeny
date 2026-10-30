"""Exact value identities independent of pickle container bytes and temporary filenames."""

import hashlib
import random
from dataclasses import fields, is_dataclass
from typing import Any

from ontogeny.execution.contracts import canonical

STATE_GROUPS = (
    "model",
    "optimizer",
    "scheduler",
    "external",
    "metadata",
    "gradients",
    "rng",
    "training_mode",
    "requires_grad",
)


def value_identity(value: Any, torch: Any, np: Any) -> Any:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        return [
            "tensor",
            str(value.dtype),
            list(value.shape),
            hashlib.sha256(array.tobytes()).hexdigest(),
        ]
    if isinstance(value, np.ndarray):
        return [
            "array",
            str(value.dtype),
            list(value.shape),
            hashlib.sha256(value.tobytes()).hexdigest(),
        ]
    if isinstance(value, random.Random):
        return ["random", value_identity(value.getstate(), torch, np)]
    if is_dataclass(value) and not isinstance(value, type):
        return [
            "dataclass",
            type(value).__module__ + "." + type(value).__qualname__,
            [[f.name, value_identity(getattr(value, f.name), torch, np)] for f in fields(value)],
        ]
    if isinstance(value, dict):
        pairs = [
            [value_identity(k, torch, np), value_identity(v, torch, np)] for k, v in value.items()
        ]
        return ["dict", sorted(pairs, key=lambda pair: canonical(pair[0]))]
    if isinstance(value, (set, frozenset)):
        values = [value_identity(v, torch, np) for v in value]
        return [type(value).__name__, sorted(values, key=canonical)]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [value_identity(v, torch, np) for v in value]]
    if isinstance(value, float):
        return ["float", value.hex()]
    if value is None or type(value) in (str, int, bool):
        return [type(value).__name__, value]
    # IntEnum must retain its type, including Action and Tile values in experience.
    from enum import Enum

    if isinstance(value, Enum):
        return ["enum", type(value).__module__ + "." + type(value).__qualname__, value.name]
    raise TypeError(f"unsupported checkpoint value identity: {type(value)}")


def fingerprints(values: tuple, torch: Any, np: Any) -> dict[str, str]:
    return {
        name: hashlib.sha256(canonical(value_identity(value, torch, np))).hexdigest()
        for name, value in zip(STATE_GROUPS, values, strict=True)
    }
