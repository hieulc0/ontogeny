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


def regular_comparisons(
    split: str, *, values: tuple[int, ...], degree: int, max_degree: int, seed: int
) -> tuple[Comparison, ...]:
    """Nested regular training graphs with equal operand frequency and shared queries.

    Degree counts partners per operand. Each edge contributes both orderings. Queries
    exclude the entire maximum-degree graph, including unused edges in smaller arms.
    """
    if split not in ("train", "development", "test"):
        raise ValueError("unknown split")
    if (
        len(values) < 4
        or len(set(values)) != len(values)
        or any(type(v) is not int or not 0 <= v <= 99 for v in values)
        or type(degree) is not int
        or type(max_degree) is not int
        or not 2 <= degree <= max_degree < len(values) - 1
        or degree % 2
        or max_degree % 2
        or type(seed) is not int
    ):
        raise ValueError("invalid regular comparison graph")
    vertices = list(values)
    random.Random(seed).shuffle(vertices)

    def edges(d: int) -> set[tuple[int, int]]:
        return {
            (min(a, b), max(a, b))
            for i, a in enumerate(vertices)
            for offset in range(1, d // 2 + 1)
            for b in (vertices[(i + offset) % len(vertices)],)
        }

    if split == "train":
        selected = sorted(edges(degree))
    else:
        largest = edges(max_degree)
        selected = sorted((a, b) for a in values for b in values if a < b and (a, b) not in largest)
        random.Random(seed + 1).shuffle(selected)
        halfway = len(selected) // 2
        selected = selected[:halfway] if split == "development" else selected[halfway:]
    return tuple(
        Comparison(a, b) for left, right in selected for a, b in ((left, right), (right, left))
    )
