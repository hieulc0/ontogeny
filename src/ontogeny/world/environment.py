import copy
import random
from dataclasses import dataclass
from typing import Any

from ontogeny.world.actions import MOVES, Action
from ontogeny.world.observations import Observation
from ontogeny.world.state import WORLD_VERSION, WorldConfig, WorldState


@dataclass(frozen=True)
class Transition:
    observation: Observation
    energy_delta: int
    collected: bool
    hazard: bool
    terminated: bool
    truncated: bool


class CellWorld:
    """Simulator. Pass observations to policies, never this object."""

    def __init__(self, config: WorldConfig | None = None, *, seed: int = 0):
        config = config or WorldConfig()
        self.config = config
        self._rng = random.Random(seed)
        positions = [(x, y) for y in range(config.height) for x in range(config.width)]
        self._rng.shuffle(positions)
        self._state = WorldState(
            positions[0],
            set(positions[1 : 1 + config.resources]),
            set(positions[1 + config.resources : 1 + config.resources + config.hazards]),
            config.initial_energy,
        )

    def observe(self, radius: int = 1) -> Observation:
        if radius not in (1, 2):
            raise ValueError("sensing radius must be 1 or 2")
        s, c = self._state, self.config
        tiles = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                p = (s.position[0] + dx, s.position[1] + dy)
                kind = "empty"
                if not (0 <= p[0] < c.width and 0 <= p[1] < c.height):
                    kind = "wall"
                elif p in s.resources:
                    kind = "resource"
                elif p in s.hazards:
                    kind = "hazard"
                tiles.append((dx, dy, kind))
        return Observation(tuple(tiles), s.energy, s.consequence, radius)

    def step(self, action: Action) -> Transition:
        if not isinstance(action, Action):
            raise ValueError("action must be an Action enum")
        s, c = self._state, self.config
        if s.terminated or s.truncated:
            raise RuntimeError("episode ended")
        before = s.energy
        s.energy -= c.inspect_cost if action == Action.INSPECT else c.action_cost
        s.consequence = "rest"
        collected = False
        if action in MOVES:
            dx, dy = MOVES[action]
            p = (s.position[0] + dx, s.position[1] + dy)
            if 0 <= p[0] < c.width and 0 <= p[1] < c.height:
                s.position = p
                s.consequence = "moved"
            else:
                s.consequence = "blocked"
        elif action == Action.INTERACT:
            collected = s.position in s.resources
            s.consequence = "collected" if collected else "no_resource"
            if collected:
                s.resources.remove(s.position)
                s.energy += c.resource_gain
        elif action == Action.INSPECT:
            s.consequence = "inspected"
        hazard = s.position in s.hazards
        if hazard:
            s.energy -= c.hazard_cost
            s.consequence = "hazard"
        s.energy = max(0, min(c.max_energy, s.energy))
        s.step += 1
        s.terminated = s.energy == 0
        s.truncated = s.step >= c.max_steps and not s.terminated
        if c.change_interval and s.step % c.change_interval == 0:
            # Relocate hazards only into empty, unoccupied cells after consequences.
            candidates = [
                (x, y)
                for y in range(c.height)
                for x in range(c.width)
                if (x, y) not in s.resources and (x, y) != s.position
            ]
            s.hazards = set(self._rng.sample(candidates, c.hazards))
        return Transition(
            self.observe(2 if action == Action.INSPECT else 1),
            s.energy - before,
            collected,
            hazard,
            s.terminated,
            s.truncated,
        )

    def snapshot(self) -> dict[str, Any]:
        """Evaluator/checkpoint API; explicitly privileged, never a policy input."""
        return copy.deepcopy(
            {
                "version": WORLD_VERSION,
                "config": self.config,
                "state": self._state,
                "rng": self._rng.getstate(),
            }
        )

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> "CellWorld":
        if snapshot["version"] != WORLD_VERSION:
            raise ValueError("world version mismatch")
        world = cls(snapshot["config"])
        world._state = copy.deepcopy(snapshot["state"])
        world._rng.setstate(snapshot["rng"])
        return world
