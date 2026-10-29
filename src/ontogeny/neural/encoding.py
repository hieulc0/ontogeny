from ontogeny.world.actions import Action
from ontogeny.world.observations import Observation

KINDS = {"empty": 1, "wall": 2, "resource": 3, "hazard": 4}
CONSEQUENCES = {
    name: i + 8
    for i, name in enumerate(
        ("start", "rest", "moved", "blocked", "collected", "no_resource", "inspected", "hazard")
    )
}


def encode(observation: Observation, action: Action) -> list[int]:
    """Version 1: row-major local tiles, radius, exact bounded energy, consequence, action."""
    if not 0 <= observation.energy <= 127 or observation.radius not in (1, 2):
        raise ValueError("encoding-v1 envelope exceeded; requires explicit migration")
    return [KINDS[k] for _, _, k in observation.tiles] + [
        16 + observation.radius,
        32 + observation.energy,
        CONSEQUENCES[observation.consequence],
        160 + int(action),
    ]
