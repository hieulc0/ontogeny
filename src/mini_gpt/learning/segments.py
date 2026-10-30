"""Explicit lineage and dataset cursors for continuing a research student."""

import re
from typing import Any


def checkpoint_due(step: int, invocation_start: int, stop_at: int, interval: int) -> bool:
    """Publish at the stop boundary even if it is not on the periodic schedule."""
    if interval <= 0 or not invocation_start < step <= stop_at:
        raise ValueError("invalid checkpoint schedule")
    return step == stop_at or (step - invocation_start) % interval == 0


def begin_segment(
    metadata: dict[str, Any],
    external: dict[str, Any],
    *,
    name: str,
    parent_sha256: str,
    data_sha256: str,
    additional_updates: int,
    model_config: dict[str, Any],
) -> int:
    """Record a new finite segment, retaining the existing learner and sampler history."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
        raise ValueError("invalid segment name")
    if not all(re.fullmatch(r"[0-9a-f]{64}", x) for x in (parent_sha256, data_sha256)):
        raise ValueError("invalid checkpoint or data digest")
    if type(additional_updates) is not int or not 0 < additional_updates <= 100000:
        raise ValueError("invalid segment update budget")
    envelope = metadata.get("training_envelope")
    if not isinstance(envelope, dict) or envelope.get("model") != model_config:
        raise ValueError("segment requires the original model configuration")
    steps = metadata.get("steps")
    if type(steps) is not int or steps < 0:
        raise ValueError("invalid learner progress")
    records = metadata.get("segments", [])
    if not isinstance(records, list) or any(r.get("name") == name for r in records):
        raise ValueError("invalid or duplicate segment name")
    cursors = external.get("dataset_cursors", {})
    if not isinstance(cursors, dict):
        raise ValueError("invalid dataset cursors")
    cursors = dict(cursors)
    previous_digest = external.get("active_data_sha256", envelope.get("data_sha256"))
    previous_cursor = external.get("batch_cursor")
    if (
        not isinstance(previous_digest, str)
        or type(previous_cursor) is not int
        or previous_cursor < 0
    ):
        raise ValueError("missing prior dataset cursor")
    cursors[previous_digest] = previous_cursor
    next_cursor = cursors.get(data_sha256, 0)
    if type(next_cursor) is not int or next_cursor < 0:
        raise ValueError("invalid next dataset cursor")
    limit = steps + additional_updates
    metadata["segments"] = [
        *records,
        {
            "name": name,
            "parent_checkpoint_sha256": parent_sha256,
            "data_sha256": data_sha256,
            "start_step": steps,
            "end_step": limit,
            "start_cursor": next_cursor,
        },
    ]
    external["dataset_cursors"] = cursors
    external["active_data_sha256"] = data_sha256
    external["batch_cursor"] = next_cursor
    return limit
