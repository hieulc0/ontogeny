"""Balanced integer comparison with unordered-pair-disjoint example splits."""

import hashlib
import random
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Comparison:
    left: int
    right: int

    @property
    def target(self) -> int:
        return ord(str(int(self.left < self.right)))

    def tokens(self, representation: Literal["bytes", "symbols"]) -> list[int]:
        if representation == "bytes":
            return list(f"{self.left:02d}<{self.right:02d}=".encode())
        if representation == "symbols":
            return [128 + self.left, 128 + self.right, ord("=")]
        raise ValueError("unknown representation")


def comparisons(split: str, *, pairs: int | None = None) -> tuple[Comparison, ...]:
    """Both orders stay together, so reversing a training pair cannot leak into a query."""
    buckets = {"train": (0, 1, 2), "development": (3,), "test": (4,)}
    if split not in buckets:
        raise ValueError("unknown split")
    selected = []
    for left in range(32):
        for right in range(left + 1, 32):
            digest = hashlib.sha256(f"comparison:{left}:{right}".encode()).digest()
            if int.from_bytes(digest, "big") % 5 in buckets[split]:
                selected.append((left, right))
    random.Random(1729).shuffle(selected)
    if pairs is not None:
        if type(pairs) is not int or not 1 <= pairs <= len(selected):
            raise ValueError("invalid pair count")
        selected = selected[:pairs]
    return tuple(
        Comparison(a, b) for left, right in selected for a, b in ((left, right), (right, left))
    )
