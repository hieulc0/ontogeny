"""Strict, non-neural interchange contracts for bounded remote work."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from ontogeny.world.state import WorldConfig


class ContractError(ValueError):
    pass


def fields(value: Any, names: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ContractError(f"expected exactly these fields: {names}")
    return value


def integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise ContractError("invalid logical identifier")
    return value


def hexadecimal(value: Any, length: int) -> str:
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise ContractError(f"expected {length}-character lowercase hexadecimal value")
    return value


def relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError("invalid relative artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ContractError("artifact path must be normalized and confined")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class InputArtifact:
    name: str
    path: str
    sha256: str
    size_bytes: int
    role: Literal["checkpoint"]
    trusted: bool

    @classmethod
    def parse(cls, value: Any) -> "InputArtifact":
        data = fields(value, "name path sha256 size_bytes role trusted")
        if data["role"] != "checkpoint" or data["trusted"] is not True:
            raise ContractError("only explicitly trusted full checkpoints are supported")
        return cls(
            identifier(data["name"]),
            relative_path(data["path"]),
            hexadecimal(data["sha256"], 64),
            integer(data["size_bytes"], "size"),
            "checkpoint",
            True,
        )


@dataclass(frozen=True)
class Limits:
    updates: int
    interactions: int
    wall_seconds: int
    output_bytes: int
    torch_memory_bytes: int

    @classmethod
    def parse(cls, value: Any) -> "Limits":
        data = fields(value, "updates interactions wall_seconds output_bytes torch_memory_bytes")
        result = cls(**{k: integer(v, k) for k, v in data.items()})
        if result.wall_seconds > 3600:
            raise ContractError("worker segments must be <= 3600 seconds")
        return result


@dataclass(frozen=True)
class Request:
    schema_version: int
    run_id: str
    trial_id: str
    arm_id: str
    protocol_sha256: str
    split: str
    split_manifest_sha256: str
    operation: Literal["train", "evaluate", "continuation_check"]
    source: dict[str, str]
    runtime: dict[str, Any]
    initialization: dict[str, Any]
    world: dict[str, Any]
    objective: dict[str, Any]
    targets: dict[str, str]
    limits: Limits
    inputs: tuple[InputArtifact, ...]
    required_outputs: tuple[str, ...]

    @classmethod
    def parse(cls, value: Any) -> "Request":
        data = fields(
            value,
            "schema_version run_id trial_id arm_id protocol_sha256 "
            "split split_manifest_sha256 operation "
            "source runtime initialization world objective targets limits inputs required_outputs",
        )
        if type(data["schema_version"]) is not int or data["schema_version"] not in (1, 2):
            raise ContractError("unsupported request schema")
        operation = data["operation"]
        if operation not in ("train", "evaluate", "continuation_check"):
            raise ContractError("unsupported operation; arbitrary entry points are forbidden")
        allowed_splits = {
            "train": ("train",),
            "evaluate": ("development", "test"),
            "continuation_check": ("diagnostic",),
        }
        if data["split"] not in allowed_splits[operation]:
            raise ContractError("operation cannot access this split")
        hexadecimal(data["split_manifest_sha256"], 64)
        if data["schema_version"] == 1:
            source = fields(data["source"], "ontogeny_sha upstream_sha")
            for value in source.values():
                hexadecimal(value, 40)
        else:
            source = fields(data["source"], "ontogeny_sha engine")
            hexadecimal(source["ontogeny_sha"], 40)
            if source["engine"] != "mini_gpt":
                raise ContractError("unified requests require the canonical internal engine")
        runtime = fields(data["runtime"], "torch cuda gpu visible_cuda_devices")
        if (runtime["torch"], runtime["cuda"], runtime["gpu"]) != (
            "2.6.0+cu124",
            "12.4",
            "Tesla T4",
        ):
            raise ContractError("unsupported runtime profile")
        integer(runtime["visible_cuda_devices"], "visible_cuda_devices")
        if runtime["visible_cuda_devices"] not in (1, 2):
            raise ContractError("unqualified visible CUDA device count")
        initialization = fields(data["initialization"], "kind seed checkpoint")
        if initialization["kind"] == "fresh":
            integer(initialization["seed"], "seed", 0)
            if initialization["checkpoint"] is not None or operation == "evaluate":
                raise ContractError(
                    "fresh initialization requires null checkpoint and cannot evaluate"
                )
        elif initialization["kind"] == "resume":
            identifier(initialization["checkpoint"])
            if initialization["seed"] is not None:
                raise ContractError("resume must not reseed")
        else:
            raise ContractError("initialization must be fresh or full-state resume")
        world = fields(data["world"], "seed episodes config")
        integer(world["seed"], "world seed", 0)
        integer(world["episodes"], "episodes")
        config = fields(world["config"], " ".join(WorldConfig.__dataclass_fields__))
        settings = WorldConfig(**config)
        if settings.max_energy > 127 or settings.max_steps > 100:
            raise ContractError("policy encoding/buffer envelope exceeded")
        objective = fields(data["objective"], "algorithm learning_rate gamma baseline_decay")
        if (
            objective["algorithm"],
            objective["learning_rate"],
            objective["gamma"],
            objective["baseline_decay"],
        ) != ("reinforce", 0.0005, 0.97, 0.9):
            raise ContractError("only the unchanged historical policy objective is qualified")
        targets = fields(data["targets"], "kind criterion")
        if targets != {"kind": "engineering", "criterion": "state_integrity"}:
            raise ContractError("worker cannot award scientific learning qualification")
        limits = Limits.parse(data["limits"])
        if settings.max_steps * world["episodes"] > limits.interactions:
            raise ContractError("interaction envelope smaller than requested episode envelope")
        if world["episodes"] > limits.updates and operation != "evaluate":
            raise ContractError("update budget smaller than requested episodes")
        if not isinstance(data["inputs"], list) or not isinstance(data["required_outputs"], list):
            raise ContractError("inputs and required_outputs must be arrays")
        inputs = tuple(InputArtifact.parse(v) for v in data["inputs"])
        if len({v.name for v in inputs}) != len(inputs) or len({v.path for v in inputs}) != len(
            inputs
        ):
            raise ContractError("duplicate input identity/path")
        if initialization["kind"] == "fresh" and inputs:
            raise ContractError("fresh jobs may not silently load a checkpoint")
        if initialization["kind"] == "resume" and (
            len(inputs) != 1 or inputs[0].name != initialization["checkpoint"]
        ):
            raise ContractError("resume requires exactly its declared full checkpoint")
        outputs = tuple(relative_path(v) for v in data["required_outputs"])
        mandatory = {
            "runtime.json",
            "execution.json",
            "metrics.json",
            "predictions.jsonl",
            "learning.jsonl",
            "checkpoint_manifest.json",
        }
        if len(set(outputs)) != len(outputs) or set(outputs) != mandatory:
            raise ContractError("required output contract must match supported worker outputs")
        if operation == "continuation_check" and (
            initialization["kind"] != "fresh"
            or world["episodes"] != 1
            or settings.max_steps != 10
            or settings.initial_energy
            <= 10 * (max(settings.action_cost, settings.inspect_cost) + settings.hazard_cost)
        ):
            raise ContractError("continuation fixture must guarantee ten live steps")
        return cls(
            data["schema_version"],
            identifier(data["run_id"]),
            identifier(data["trial_id"]),
            identifier(data["arm_id"]),
            hexadecimal(data["protocol_sha256"], 64),
            data["split"],
            hexadecimal(data["split_manifest_sha256"], 64),
            operation,
            dict(source),
            dict(runtime),
            dict(initialization),
            dict(world),
            dict(objective),
            dict(targets),
            limits,
            inputs,
            outputs,
        )

    @classmethod
    def read(cls, path: Path) -> "Request":
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ContractError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value):
            raise ContractError(f"nonfinite JSON value: {value}")

        return cls.parse(
            json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=reject_constant)
        )

    def to_dict(self) -> dict:
        # Canonical JSON represents tuples as arrays.
        return json.loads(canonical(asdict(self)))

    @property
    def sha256(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class ExecutionContext:
    provider: str
    native_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        if self.provider not in ("kaggle", "colab") or not self.native_id:
            raise ContractError("invalid provider context")
        identifier(self.attempt_id)
