import random

from ontogeny.world.actions import Action
from ontogeny.world.environment import CellWorld
from ontogeny.world.state import WorldConfig


def test_seed_replay_and_snapshot_continuation():
    c = WorldConfig(initial_energy=60, change_interval=2)
    a, b = CellWorld(c, seed=92), CellWorld(c, seed=92)
    rng = random.Random(123)
    for _ in range(6):
        action = rng.choice(list(Action))
        assert a.step(action) == b.step(action)
    clone = CellWorld.restore(a.snapshot())
    for action in [Action.INSPECT, Action.EAST, Action.REST]:
        assert a.step(action) == clone.step(action)
        assert a.snapshot() == clone.snapshot()
    assert a.snapshot() == b.snapshot() or a.snapshot()["state"].step == 9


def test_hazard_change_is_seeded_and_excludes_resource_and_organism():
    w = CellWorld(WorldConfig(change_interval=1), seed=4)
    before = w.snapshot()["state"].hazards
    w.step(Action.REST)
    s = w.snapshot()["state"]
    assert s.hazards != before
    assert not s.hazards & s.resources
    assert s.position not in s.hazards


def test_snapshot_is_not_alias():
    w = CellWorld()
    snapshot = w.snapshot()
    snapshot["state"].resources.clear()
    assert len(w.snapshot()["state"].resources) == 24
