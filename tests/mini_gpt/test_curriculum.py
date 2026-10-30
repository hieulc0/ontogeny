"""Non-neural curriculum and learner-lineage contracts."""

import copy
import hashlib
import json

import pytest

from mini_gpt.data.curriculum import batch, examples, generate
from mini_gpt.learning.segments import begin_segment


def test_curriculum_splits_and_targets(tmp_path):
    rows = examples()
    assert len(rows) == 2000
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["prompt"] for r in rows}) == len(rows)
    for row in rows:
        sample = batch(row)
        tokens, targets = sample["tokens"][0], sample["targets"][0]
        assert len(tokens) == len(targets) <= 512
        assert bytes(x for x in targets if x != -100).decode() == row["target"] + "\n"
        assert all(0 <= x < 256 for x in tokens)
    first = generate(tmp_path / "one")
    second = generate(tmp_path / "two")
    assert first == second
    assert sum(first["counts"].values()) == 2000
    for name, digest in first["files"].items():
        assert hashlib.sha256((tmp_path / "one" / name).read_bytes()).hexdigest() == digest
    train = json.loads((tmp_path / "one" / "train-batches.json").read_text())
    assert len(train) == first["counts"]["train"]


def test_segments_keep_progress_and_dataset_cursors():
    model = {"width": 10}
    original, other, parent = "a" * 64, "b" * 64, "c" * 64
    metadata = {"steps": 2, "training_envelope": {"model": model, "data_sha256": original}}
    external = {"batch_cursor": 2, "replay": ["preserved"]}
    assert (
        begin_segment(
            metadata,
            external,
            name="math",
            parent_sha256=parent,
            data_sha256=other,
            additional_updates=3,
            model_config=model,
        )
        == 5
    )
    assert external["batch_cursor"] == 0 and external["replay"] == ["preserved"]
    assert external["dataset_cursors"][original] == 2
    metadata["steps"] = 5
    external["batch_cursor"] = 3
    assert (
        begin_segment(
            metadata,
            external,
            name="return",
            parent_sha256=parent,
            data_sha256=original,
            additional_updates=1,
            model_config=model,
        )
        == 6
    )
    assert external["batch_cursor"] == 2 and external["dataset_cursors"][other] == 3
    assert metadata["segments"][0]["parent_checkpoint_sha256"] == parent


def test_invalid_segment_cannot_mutate_lineage():
    metadata = {"steps": 2, "training_envelope": {"model": {}, "data_sha256": "a" * 64}}
    external = {"batch_cursor": 2}
    before = copy.deepcopy((metadata, external))
    with pytest.raises(ValueError):
        begin_segment(
            metadata,
            external,
            name="bad name",
            parent_sha256="b" * 64,
            data_sha256="c" * 64,
            additional_updates=1,
            model_config={},
        )
    assert (metadata, external) == before
