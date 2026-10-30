from ontogeny.world.observations import Observation

KINDS = {"empty": 1, "wall": 2, "resource": 3, "hazard": 4}
CONSEQUENCES = {
    name: i + 8
    for i, name in enumerate(
        ("start", "rest", "moved", "blocked", "collected", "no_resource", "inspected", "hazard")
    )
}


def encode_policy(observation: Observation) -> list[int]:
    """Fixed 29-token partial observation; 0 means outside the sensed patch."""
    if observation.radius not in (1, 2) or not 0 <= observation.energy <= 127:
        raise ValueError("policy encoding envelope exceeded")
    tiles = {(x, y): KINDS[k] for x, y, k in observation.tiles}
    return [tiles.get((x, y), 0) for y in range(-2, 3) for x in range(-2, 3)] + [
        16 + observation.radius,
        32 + observation.energy,
        CONSEQUENCES[observation.consequence],
        167,
    ]
