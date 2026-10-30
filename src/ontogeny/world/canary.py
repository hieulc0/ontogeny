"""Seeded interaction tasks isolated from navigation benchmarks."""

import random
from dataclasses import dataclass

from ontogeny.world.observations import Observation


@dataclass(frozen=True)
class Feedback:
    reward: float
    done: bool
    collected: bool


class CueWorld:
    def __init__(self, seed: int, *, delayed: bool = False):
        self._cue = random.Random(seed).randrange(2)
        self._length = 3 if delayed else 1
        self._step = 0
        self._successful = True

    def observe(self) -> Observation:
        tiles = tuple(
            (x, y, ("resource" if self._cue == 0 else "hazard") if (x, y) == (0, 0) else "empty")
            for y in range(-1, 2)
            for x in range(-1, 2)
        )
        return Observation(tiles, 40, ("start", "moved", "inspected")[self._step], 1)

    def step(self, action: int) -> Feedback:
        if type(action) is not int or action not in (0, 1):
            raise ValueError("expected one of two canary actions")
        if self._step >= self._length:
            raise RuntimeError("episode ended")
        beneficial = self._cue if self._step % 2 == 0 else 1 - self._cue
        self._successful &= action == beneficial
        self._step += 1
        done = self._step == self._length
        reward = (1.0 if self._successful else -1.0) if done else 0.0
        return Feedback(reward, done, done and self._successful)
