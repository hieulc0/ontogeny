"""Full learning ownership with alias-preserving rollback and deterministic continuation.

Only explicitly trusted same-code state is deserialized. No implicit model replacement.
"""

import copy
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from mini_gpt.learning.optimizer import adamw_groups, optimizer_membership
from mini_gpt.runtime import require_remote_cuda


class LearningState:
    """One model, optimizer, scheduler and all declared mutable experience/state."""

    VERSION = 2

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 0.002,
        weight_decay: float = 0.0,
        external: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        require_remote_cuda()
        self.model = model
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            adamw_groups(model, weight_decay, include_frozen=True), lr=lr
        )
        self.scheduler = None
        self.external = (
            external if external is not None else {"replay": [], "memory": {}, "controller": {}}
        )
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("steps", 0)
        self._in_candidate = False

    def snapshot(self) -> dict[str, Any]:
        for module in self.model.modules():
            if module._forward_hooks or module._backward_hooks:
                raise NotImplementedError(
                    "Hook-bearing model needs an explicit serializable hook contract"
                )
        if any(p._backward_hooks for p in self.model.parameters()):
            raise NotImplementedError("Parameter hooks need an explicit serializable hook contract")
        return {
            "version": self.VERSION,
            "roots": copy.deepcopy(
                (self.model, self.optimizer, self.scheduler, self.external, self.metadata)
            ),
            "gradients": {
                n: p.grad.detach().clone()
                for n, p in self.model.named_parameters()
                if p.grad is not None
            },
            "rng": (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            ),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot["version"] != self.VERSION:
            raise ValueError("Unsupported learning-state version")
        model, optimizer, scheduler, external, metadata = snapshot["roots"]
        if type(model) is not type(self.model) or type(optimizer) is not type(self.optimizer):
            raise ValueError("Restore requires matching root types")
        # Root handles stay live; a shared memo also rebinds controller/self aliases.
        memo = {id(model): self.model, id(optimizer): self.optimizer, id(external): self.external}
        model_dict = copy.deepcopy(model.__dict__, memo)
        optimizer_dict = copy.deepcopy(optimizer.__dict__, memo)
        scheduler = copy.deepcopy(scheduler, memo)
        external_dict = copy.deepcopy(dict(external), memo)
        metadata = copy.deepcopy(metadata, memo)
        self.model.__dict__.clear()
        self.model.__dict__.update(model_dict)
        self.optimizer.__dict__.clear()
        self.optimizer.__dict__.update(optimizer_dict)
        self.scheduler = scheduler
        self.external.clear()
        self.external.update(external_dict)
        self.metadata = metadata
        for name, p in self.model.named_parameters():
            p.grad = snapshot["gradients"].get(name)
            if p.grad is not None:
                p.grad = p.grad.clone()
        py, np_rng, cpu, cuda = snapshot["rng"]
        random.setstate(py)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu.cpu())
        if cuda is not None:
            if len(cuda) != torch.cuda.device_count():
                raise RuntimeError("Exact resumed learning requires matching CUDA device count")
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
        if any(optimizer_membership(self.model, self.optimizer).values()):
            raise RuntimeError("Restored optimizer ownership is invalid")

    def candidate(self, modify: Any, validate: Any) -> bool:
        if self._in_candidate:
            raise RuntimeError("Nested candidates are unsupported")
        before = self.snapshot()
        roots = (self.model, self.optimizer, self.external)

        def rollback():
            self.model, self.optimizer, self.external = roots
            self.restore(before)

        self._in_candidate = True
        try:
            modify(self)
            if any(
                current is not original
                for current, original in zip(
                    (self.model, self.optimizer, self.external), roots, strict=True
                )
            ):
                raise RuntimeError("Candidate cannot replace owned root handles")
            if any(optimizer_membership(self.model, self.optimizer).values()):
                raise RuntimeError("Candidate optimizer ownership violation")
            accepted = bool(validate(self))
            if not accepted:
                rollback()
            return accepted
        except BaseException:
            rollback()
            raise
        finally:
            self._in_candidate = False

    def save(self, path: Path | str) -> None:
        if self._in_candidate:
            raise RuntimeError("Candidate checkpoints cannot be published")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(self.snapshot(), f)
                f.flush()
                os.fsync(f.fileno())
            # Same-filesystem hard-link publication is atomic and cannot clobber
            # an existing immutable checkpoint, including a concurrent publisher.
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: Path | str) -> "LearningState":
        require_remote_cuda()
        # Only same-code trusted artifacts produced by this owner are supported.
        try:
            snapshot = torch.load(path, weights_only=False)
        except ModuleNotFoundError as error:
            raise ValueError(
                "historical class paths require explicit trusted conversion"
            ) from error
        if snapshot.get("version") != cls.VERSION:
            raise ValueError("checkpoint format requires an explicit versioned conversion")
        model = copy.deepcopy(snapshot["roots"][0])
        owner = cls(model)
        owner.restore(snapshot)
        return owner

    def backward(self, loss: Tensor) -> None:
        require_remote_cuda()
        if not torch.isfinite(loss).all():
            raise ValueError("nonfinite loss")
        if any(optimizer_membership(self.model, self.optimizer).values()):
            raise RuntimeError("optimizer ownership violation")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()
        ):
            raise ValueError("nonfinite gradient")

    def step(self, *, clip_norm: float = 1.0) -> float:
        require_remote_cuda()
        if not clip_norm > 0:
            raise ValueError("gradient clipping norm must be positive")
        if any(optimizer_membership(self.model, self.optimizer).values()):
            raise RuntimeError("optimizer ownership violation")
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite gradient norm")
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.metadata["steps"] += 1
        return float(norm)
