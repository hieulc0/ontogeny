from dataclasses import dataclass

from ontogeny.world.actions import Action
from ontogeny.world.environment import Transition
from ontogeny.world.observations import Observation


@dataclass(frozen=True)
class Experience:
    observation: Observation
    action: Action
    outcome: Transition
