"""Matched per-task feedback streams with different temporal schedules."""

import hashlib
import json
import random
from typing import Literal

from mini_gpt.data.comparison import regular_comparisons

Example = tuple[tuple[int, ...], int]
PROTOCOL = "ordering-joint-001"
SEEDS = (1010, 1111, 1212)
ARMS = ("joint", "alternating", "sequential")


def examples(task: Literal["A", "B"], split: str) -> tuple[Example, ...]:
    if task not in ("A", "B"):
        raise ValueError("unknown task")
    rows = regular_comparisons(split, values=tuple(range(32)), degree=8, max_degree=8, seed=1223)
    relation = "<" if task == "A" else ">"
    return tuple(
        (
            tuple(f"{r.left + 100:03d}{relation}{r.right + 100:03d}=".encode()),
            ord(str(int(r.left < r.right if task == "A" else r.left > r.right))),
        )
        for r in rows
    )


def digest(rows: tuple[Example, ...]) -> str:
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def feedback_stream(task: Literal["A", "B"]) -> tuple[Example, ...]:
    rng = random.Random(444 if task == "A" else 555)
    rows = examples(task, "train")
    return tuple(row for _ in range(512) for row in rng.sample(rows, 16))


def allocation(arm: str, step: int) -> tuple[int, int]:
    """Number of A/B records consumed at a one-based update; no reordering within a task."""
    if type(step) is not int or not 1 <= step <= 1024:
        raise ValueError("invalid step")
    if arm == "joint":
        return 8, 8
    if arm == "alternating":
        return (16, 0) if step % 2 else (0, 16)
    if arm == "sequential":
        return (16, 0) if step <= 512 else (0, 16)
    raise ValueError("unknown arm")
