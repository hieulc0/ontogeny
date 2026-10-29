from collections.abc import Callable
from dataclasses import dataclass

from ontogeny.learning.experience import Experience
from ontogeny.organism.interfaces import Policy
from ontogeny.world.environment import CellWorld


@dataclass(frozen=True)
class EpisodeMetrics:
    duration: int
    collected: int
    hazard_exposures: int
    final_energy: int
    inspections: int
    observation_tiles: int
    terminated: bool
    truncated: bool


def run_episode(
    world: CellWorld,
    policy: Policy,
    *,
    learning: bool = False,
    record: Callable[[Experience], None] | None = None,
) -> EpisodeMetrics:
    observation = world.observe()
    steps = collected = hazards = inspections = tiles = 0
    while True:
        tiles += len(observation.tiles)
        action = policy.act(observation)
        outcome = world.step(action)
        experience = Experience(observation, action, outcome)
        if record:
            record(experience)
        if learning:
            policy.learn(experience)
        steps += 1
        collected += int(outcome.collected)
        hazards += int(outcome.hazard)
        inspections += int(action.name == "INSPECT")
        observation = outcome.observation
        if outcome.terminated or outcome.truncated:
            return EpisodeMetrics(
                steps,
                collected,
                hazards,
                observation.energy,
                inspections,
                tiles,
                outcome.terminated,
                outcome.truncated,
            )
