from dataclasses import FrozenInstanceError, fields

import pytest

from ontogeny.world.environment import CellWorld
from ontogeny.world.observations import Observation


def test_partial_immutable_observation():
    w = CellWorld()
    o = w.observe()
    assert {f.name for f in fields(Observation)} == {"tiles", "energy", "consequence", "radius"}
    assert len(o.tiles) == 9
    assert all(abs(x) <= 1 and abs(y) <= 1 for x, y, _ in o.tiles)
    with pytest.raises(FrozenInstanceError):
        o.__setattr__("energy", 0)
    assert w.observe().energy == 40


def test_hidden_far_tiles_do_not_enter_observation():
    w = CellWorld()
    snapshot = w.snapshot()
    snapshot["state"].position = (0, 0)
    snapshot["state"].resources = {(11, 11)}
    snapshot["state"].hazards = {(10, 10)}
    first = CellWorld.restore(snapshot).observe()
    snapshot["state"].resources = {(11, 10)}
    snapshot["state"].hazards = {(10, 11)}
    assert CellWorld.restore(snapshot).observe() == first
    with pytest.raises(ValueError):
        w.observe(12)
