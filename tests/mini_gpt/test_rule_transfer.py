"""Seeded rules and paired-analysis contracts; no neural work."""

import pytest

from mini_gpt.data.rules import generate, tasks
from mini_gpt.evaluation.curves import ARMS, normalized_area, paired_report
from mini_gpt.learning.segments import checkpoint_due


def test_families_are_disjoint_and_examples_reproducible(tmp_path):
    groups = [tasks(name, 42) for name in ("experience", "development", "confirmation")]
    identities = [{task.identity for task in group} for group in groups]
    assert not any(identities[i] & identities[j] for i in range(3) for j in range(i))
    for group in groups:
        for task in group:
            assert not set(task.support) & set(task.query)
            assert set(task.support) | set(task.query) == set(range(task.modulus))
            assert all("multiplier" not in item["prompt"] for item in task.examples("support"))
    assert generate(tmp_path / "a", "experience", 7) == generate(tmp_path / "b", "experience", 7)
    assert tasks("experience", 7) == tasks("experience", 7)
    assert tasks("experience", 7) != tasks("experience", 8)


def test_normalized_auc_uses_irregular_update_spacing():
    assert normalized_area([0, 1, 3], [0, 1, 1]) == pytest.approx(5 / 6)
    with pytest.raises(ValueError):
        normalized_area([0, 1, 1], [0, 1, 1])


def test_checkpoint_schedule_preserves_the_interruption_boundary():
    assert not checkpoint_due(6, 5, 15, 20)
    assert checkpoint_due(15, 5, 15, 20)
    assert checkpoint_due(25, 15, 25, 20)
    with pytest.raises(ValueError):
        checkpoint_due(5, 5, 15, 20)


def test_paired_report_uses_complete_crossed_control_and_is_reproducible():
    values = {
        "continuing": [0.2, 0.8],
        "frozen": [0.2, 0.2],
        "fresh": [0.1, 0.3],
        "unrelated": [0.1, 0.5],
    }
    rows = [
        {"model_seed": seed, "task": task, "arm": arm, "accuracy": values[arm]}
        for seed in [1, 2]
        for task in ["a", "b"]
        for arm in ARMS
    ]
    result = paired_report(rows, [0, 1], bootstrap_samples=100, seed=4)
    assert result == paired_report(rows, [0, 1], bootstrap_samples=100, seed=4)
    assert result["difference_vs_strongest_control"] == pytest.approx(0.2)
    assert result["bootstrap_interval_95"] == pytest.approx([0.2, 0.2])
    with pytest.raises(ValueError, match="complete"):
        paired_report(rows[:-1], [0, 1])
    rows[-1] = {**rows[-1], "accuracy": [0.2, 0.3]}
    with pytest.raises(ValueError, match="frozen"):
        paired_report(rows, [0, 1])
