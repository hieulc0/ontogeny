import random

from ontogeny.world.actions import Action
from ontogeny.world.environment import CellWorld
from ontogeny.world.state import WorldConfig


def test_energy_ledger_across_tiny_seeded_episodes():
    for seed in range(10):
        w = CellWorld(seed=seed)
        rng = random.Random(seed)
        while True:
            before = w.observe().energy
            action = rng.choice(list(Action))
            t = w.step(action)
            cost = 2 if action == Action.INSPECT else 1
            expected = max(0, min(60, before - cost + 15 * t.collected - 6 * t.hazard))
            assert t.observation.energy == expected
            assert t.energy_delta == expected - before
            if t.terminated or t.truncated:
                break


def test_no_action_free_energy_or_infinite_rest():
    w = CellWorld(WorldConfig(resources=0, hazards=0))
    for i in range(40):
        t = w.step(Action.REST)
        assert t.observation.energy == 39 - i
    assert w.snapshot()["state"].terminated
