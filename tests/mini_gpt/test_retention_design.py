"""Local-safe split and feedback-allocation audit, without neural imports."""

import pytest

from experiments.neural.sequential_retention.design import examples, practice_batch


def test_order_tasks_have_visible_cues_and_no_split_leakage():
    sets = [examples("A", s) for s in ("train", "development", "test")]
    assert [len(s) for s in sets] == [256, 368, 368]
    for rows in sets:
        assert sum(label == 49 for _, label in rows) == len(rows) // 2
        for tokens, label in rows:
            text = bytes(tokens).decode()
            left, right = int(text[:2]), int(text[3:5])
            assert 64 <= left <= 95 and 64 <= right <= 95
            assert label == (49 if left < right else 48)
    for i, rows in enumerate(sets):
        pairs = {tuple(sorted((bytes(t)[:2], bytes(t)[3:5]))) for t, _ in rows}
        for other in sets[i + 1 :]:
            assert not pairs & {tuple(sorted((bytes(t)[:2], bytes(t)[3:5]))) for t, _ in other}
    for (a, la), (b, lb) in zip(sets[0], examples("B", "train"), strict=True):
        assert a[:2] + a[3:] == b[:2] + b[3:]
        assert a[2] == ord("<") and b[2] == ord(">")
        assert la + lb == 97


def test_replay_budget_and_current_exposure_control():
    current = list(examples("B", "train")[:16])
    memory = list(examples("A", "train")[:8])
    ordinary = practice_batch("ordinary", current, memory)
    replay = practice_batch("replay", current, memory)
    matched = practice_batch("matched_new", current, memory)
    assert len(ordinary) == len(replay) == len(matched) == 16
    assert replay[:8] == matched[:8] == current[:8]
    assert replay[8:] == memory and matched[8:] == current[:8]
    assert len(set(ordinary)) == 16 and len(set(matched)) == 8
    with pytest.raises(ValueError):
        practice_batch("unknown", current, memory)
    with pytest.raises(ValueError):
        practice_batch("replay", current, [])
