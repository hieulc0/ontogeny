from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    # Relative offsets only. No seed, global position, map or RNG state.
    tiles: tuple[tuple[int, int, str], ...]
    energy: int
    consequence: str
    radius: int
