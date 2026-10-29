import random

from ontogeny.learning.experience import Experience
from ontogeny.world.actions import MOVES, Action
from ontogeny.world.observations import Observation


class RandomPolicy:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def act(self, observation: Observation) -> Action:
        return self.rng.choice(list(Action))

    def learn(self, experience: Experience) -> None:
        pass


class HeuristicPolicy(RandomPolicy):
    """Declared nonlearning control: collect locally; approach visible resources."""

    def act(self, observation: Observation) -> Action:
        tiles = {(x, y): kind for x, y, kind in observation.tiles}
        if tiles[0, 0] == "resource":
            return Action.INTERACT
        safe = [a for a, p in MOVES.items() if tiles[p] not in ("wall", "hazard")]
        resources = [(x, y) for x, y, kind in observation.tiles if kind == "resource"]
        if resources and safe:
            target = min(resources, key=lambda p: abs(p[0]) + abs(p[1]))
            return min(
                safe, key=lambda a: abs(target[0] - MOVES[a][0]) + abs(target[1] - MOVES[a][1])
            )
        return self.rng.choice(safe) if safe else Action.REST


class RestPolicy:
    def act(self, observation: Observation) -> Action:
        return Action.REST

    def learn(self, experience: Experience) -> None:
        pass
