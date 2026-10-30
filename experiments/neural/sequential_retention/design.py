"""Frozen task and feedback allocation for the ordering-retention experiment."""

import hashlib
import json
from typing import Literal

from mini_gpt.data.comparison import regular_comparisons

Example = tuple[tuple[int, ...], int]
ARMS = ("ordinary", "replay", "matched_new")
SEEDS = (707, 808, 909)
PROTOCOL = "ordering-retention-001"


def examples(task: Literal["A", "B"], split: str) -> tuple[Example, ...]:
    if task not in ("A", "B"):
        raise ValueError("unknown task")
    result = []
    for row in regular_comparisons(
        split, values=tuple(range(64, 96)), degree=8, max_degree=8, seed=991
    ):
        relation = "<" if task == "A" else ">"
        tokens = tuple(f"{row.left:02d}{relation}{row.right:02d}=".encode())
        target = int(row.left < row.right) if task == "A" else int(row.left > row.right)
        result.append((tokens, ord(str(target))))
    return tuple(result)


def digest(rows: tuple[Example, ...]) -> str:
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def practice_batch(arm: str, current: list[Example], replay: list[Example]) -> list[Example]:
    """Match total tokens; matched_new also matches replay's distinct B presentations."""
    if len(current) != 16 or len(replay) != 8:
        raise ValueError("expected16 current and8 remembered examples")
    if arm == "ordinary":
        return current
    if arm == "replay":
        return current[:8] + replay
    if arm == "matched_new":
        return current[:8] * 2
    raise ValueError("unknown arm")
