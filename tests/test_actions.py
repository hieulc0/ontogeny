import pytest

from ontogeny.world.actions import Action
from ontogeny.world.environment import CellWorld


def fixture_world():
    w = CellWorld()
    snap = w.snapshot()
    s = snap["state"]
    s.position = (0, 0)
    s.resources = {(1, 0)}
    s.hazards = {(0, 1)}
    return CellWorld.restore(snap)


def test_movement_resource_consumption_and_hazard():
    w = fixture_world()
    assert w.step(Action.NORTH).observation.consequence == "blocked"
    assert w.step(Action.WEST).observation.consequence == "blocked"
    assert w.step(Action.EAST).observation.consequence == "moved"
    assert w.step(Action.INTERACT).collected
    assert not w.step(Action.INTERACT).collected
    w.step(Action.WEST)
    hit = w.step(Action.SOUTH)
    assert hit.hazard and hit.energy_delta == -7
    assert w.step(Action.REST).hazard


@pytest.mark.parametrize("invalid", [-1, 7, 0, "REST", None, True])
def test_invalid_action_does_not_mutate(invalid):
    w = fixture_world()
    before = w.snapshot()
    with pytest.raises(ValueError):
        w.step(invalid)
    assert w.snapshot() == before


def test_inspection_cost_and_radius():
    w = fixture_world()
    result = w.step(Action.INSPECT)
    assert result.energy_delta == -2
    assert result.observation.radius == 2 and len(result.observation.tiles) == 25
    assert w.step(Action.REST).observation.radius == 1
