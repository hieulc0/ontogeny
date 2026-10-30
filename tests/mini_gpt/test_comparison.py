"""Local-safe label, representation and leakage checks; no neural imports."""

import pytest

from mini_gpt.data.comparison import Comparison, comparisons


def test_splits_are_balanced_and_pair_disjoint():
    groups = [comparisons(split) for split in ("train", "development", "test")]
    unordered = [{tuple(sorted((r.left, r.right))) for r in rows} for rows in groups]
    assert sum(len(s) for s in unordered) == 32 * 31 // 2
    for i, rows in enumerate(groups):
        assert sum(r.target == ord("1") for r in rows) == len(rows) // 2
        assert all(not unordered[i] & other for j, other in enumerate(unordered) if i != j)
    assert set(comparisons("train", pairs=16)) < set(comparisons("train", pairs=64))


def test_labels_and_inputs_are_independently_decodable():
    for row in comparisons("train"):
        a, b = row.tokens("bytes")[:-1], row.tokens("symbols")
        left, right = bytes(a).decode().split("<")
        assert row.target == (49 if int(left) < int(right) else 48)
        assert b == [128 + int(left), 128 + int(right), 61]
    assert Comparison(3, 27).target == 49
    assert Comparison(27, 3).target == 48


def test_invalid_requests_fail():
    with pytest.raises(ValueError):
        comparisons("unknown")
    for count in (0, True, 1000):
        with pytest.raises(ValueError):
            comparisons("train", pairs=count)
    with pytest.raises(ValueError):
        Comparison(1, 2).tokens("invalid")  # type: ignore[arg-type]


def test_regular_coverage_is_matched_and_held_out():
    from collections import Counter

    from mini_gpt.data.comparison import regular_comparisons

    def rows(split, degree):
        return regular_comparisons(
            split, values=tuple(range(32, 64)), degree=degree, max_degree=8, seed=771
        )

    low, high = rows("train", 2), rows("train", 8)
    assert len(low) == 64 and len(high) == 256
    assert set(low) < set(high)
    for degree, train in ((2, low), (8, high)):
        assert Counter(r.left for r in train) == {v: degree for v in range(32, 64)}
        assert Counter(r.right for r in train) == {v: degree for v in range(32, 64)}
    dev, test = rows("development", 2), rows("test", 2)
    assert dev == rows("development", 8) and test == rows("test", 8)
    assert len(dev) == len(test) == 368
    assert not set(dev) & set(test) and not (set(dev) | set(test)) & set(high)
    assert len(set(high) | set(dev) | set(test)) == 32 * 31
    assert not (set(high) | set(dev) | set(test)) & set(comparisons("test"))
    for group in (low, high, dev, test):
        assert sum(r.target == 49 for r in group) == len(group) // 2


@pytest.mark.parametrize("degree", [True, 0, 3, 10])
def test_regular_invalid_degree(degree):
    from mini_gpt.data.comparison import regular_comparisons

    with pytest.raises(ValueError):
        regular_comparisons("train", values=tuple(range(32)), degree=degree, max_degree=8, seed=771)
