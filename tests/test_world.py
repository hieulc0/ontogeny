import pytest

from ontogeny.world.actions import Action
from ontogeny.world.environment import CellWorld
from ontogeny.world.state import WorldConfig


def test_seeded_layout_is_disjoint():
    w = CellWorld(seed=3)
    s = w.snapshot()["state"]
    assert len(s.resources) == 24 and len(s.hazards) == 12
    assert not s.resources & s.hazards
    assert s.position not in s.resources | s.hazards


def test_terminal_and_horizon():
    w = CellWorld(WorldConfig(initial_energy=1))
    assert w.step(Action.REST).terminated
    with pytest.raises(RuntimeError):
        w.step(Action.REST)
    w = CellWorld(WorldConfig(max_steps=1))
    result = w.step(Action.REST)
    assert result.truncated and not result.terminated


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(width=0),
        dict(resources=144),
        dict(action_cost=0),
        dict(initial_energy=61),
        dict(hazards=-1),
        dict(max_steps=True),
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        WorldConfig(**kwargs)
