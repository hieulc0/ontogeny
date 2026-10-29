from typing import Protocol

from ontogeny.learning.experience import Experience
from ontogeny.world.actions import Action
from ontogeny.world.observations import Observation


class Policy(Protocol):
    def act(self, observation: Observation) -> Action: ...
    def learn(self, experience: Experience) -> None: ...
