"""Non-neural tests of temporal matching and held-out identities."""

import pytest

from experiments.neural.joint_acquisition.design import ARMS, allocation, examples, feedback_stream


def test_exact_per_task_streams_and_budgets():
    streams = {task: feedback_stream(task) for task in ("A", "B")}
    for arm in ARMS:
        cursors = {"A": 0, "B": 0}
        observed = {"A": [], "B": []}
        for step in range(1, 1025):
            counts = allocation(arm, step)
            assert sum(counts) == 16
            for task, count in zip(("A", "B"), counts, strict=True):
                begin = cursors[task]
                observed[task].extend(streams[task][begin : begin + count])
                cursors[task] += count
        assert cursors == {"A": 8192, "B": 8192}
        assert all(tuple(observed[t]) == streams[t] for t in streams)
    assert allocation("sequential", 512) == (16, 0)
    assert allocation("sequential", 513) == (0, 16)
    for step in (0, 1025, True):
        with pytest.raises(ValueError):
            allocation("joint", step)


def test_labels_coverage_and_disjoint_pairs():
    groups = []
    for split in ("train", "development", "test"):
        a, b = examples("A", split), examples("B", split)
        pairs = set()
        for (tokens, label), (other, reverse) in zip(a, b, strict=True):
            text = bytes(tokens).decode()
            left, right = int(text[:3]), int(text[4:7])
            assert 100 <= left <= 131 and 100 <= right <= 131 and len(tokens) == 8
            assert label == 48 + int(left < right) and reverse == 48 + int(left > right)
            assert tokens[:3] + tokens[4:] == other[:3] + other[4:]
            pairs.add(tuple(sorted((left, right))))
        assert sum(label == 49 for _, label in a) == len(a) // 2
        groups.append(pairs)
    assert [len(g) for g in groups] == [128, 184, 184]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
