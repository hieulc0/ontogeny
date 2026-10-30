"""Exact value identities independent of pickle container bytes and temporary filenames."""

import hashlib
import json
import random
from dataclasses import fields, is_dataclass
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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


def equal(left: Any, right: Any, torch: Any, np: Any) -> bool:
    if isinstance(left, random.Random):
        return isinstance(right, random.Random) and left.getstate() == right.getstate()
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.device == right.device
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray):
        return (
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equal(left[k], right[k], torch, np) for k in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            equal(a, b, torch, np) for a, b in zip(left, right, strict=True)
        )
    return left == right


def capture_state(owner: Any) -> tuple:
    snapshot = owner.snapshot()
    model, optimizer, scheduler, external, metadata = snapshot["roots"]
    return (
        model.state_dict(),
        optimizer.state_dict(),
        None
        if scheduler is None
        else {
            "type": type(scheduler).__module__ + "." + type(scheduler).__qualname__,
            "state": scheduler.state_dict(),
        },
        external,
        metadata,
        snapshot["gradients"],
        snapshot["rng"],
        model.training,
        [p.requires_grad for p in model.parameters()],
    )
