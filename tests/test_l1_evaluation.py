"""Local-safe fake metrics and encodings, no neural execution."""

import pytest

from ontogeny.evaluation.baselines import RestPolicy
from ontogeny.evaluation.l1 import paired_interval
from ontogeny.neural.encoding import encode_policy
from ontogeny.world.environment import CellWorld


def test_policy_encoding_has_fixed_context_and_masked_unobserved_tiles():
    world = CellWorld()
    normal, inspected = encode_policy(world.observe()), encode_policy(world.observe(2))
    assert len(normal) == len(inspected) == 29
    assert normal[0] == 0 and 0 not in inspected[:25]
    assert max(normal + inspected) < 256
    assert normal[12] == inspected[12]
    assert RestPolicy().act(world.observe()).name == "REST"


def test_paired_interval_respects_initialization_clusters():
    rows = []
    for seed, effect in zip([42, 43, 44], [1, 2, 3], strict=True):
        for world in range(10):
            for arm in ["frozen", "learning"]:
                rows.append(
                    dict(
                        model_seed=seed,
                        world_seed=world,
                        variant="default",
                        arm=arm,
                        duration=20 + (effect if arm == "learning" else 0),
                    )
                )
    ci = paired_interval(rows, "frozen", "duration", "default")
    assert ci["mean_difference"] == 2
    assert ci["lower95"] < 0 < ci["upper95"]
    with pytest.raises(ValueError):
        paired_interval(rows[:20], "frozen", "duration", "default")


def test_continuation_comparator_uses_rng_state(monkeypatch):
    # Import the driver is safe: torch is only imported inside remote setup().
    import importlib
    import random
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "kaggle"))
    driver = importlib.import_module("run_l1")
    torch = SimpleNamespace(Tensor=type("FakeTensor", (), {}))
    np = SimpleNamespace(ndarray=type("FakeArray", (), {}))
    left, right = random.Random(2), random.Random(2)
    assert left is not right and driver.equal(left, right, torch, np)
    right.random()
    assert not driver.equal(left, right, torch, np)
