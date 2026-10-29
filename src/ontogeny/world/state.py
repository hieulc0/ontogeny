from dataclasses import dataclass

Position = tuple[int, int]
WORLD_VERSION = "cellworld-v1"


@dataclass(frozen=True)
class WorldConfig:
    width: int = 12
    height: int = 12
    resources: int = 24
    hazards: int = 12
    initial_energy: int = 40
    max_energy: int = 60
    max_steps: int = 100
    action_cost: int = 1
    inspect_cost: int = 2
    resource_gain: int = 15
    hazard_cost: int = 6
    change_interval: int = 20

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            min(
                self.width,
                self.height,
                self.initial_energy,
                self.max_energy,
                self.max_steps,
                self.action_cost,
                self.inspect_cost,
            )
            < 1
        ):
            raise ValueError("dimensions, energy, horizon and costs must be positive")
        if self.initial_energy > self.max_energy:
            raise ValueError("initial energy exceeds capacity")
        if self.resources + self.hazards >= self.width * self.height:
            raise ValueError("leave room for the organism")


@dataclass
class WorldState:
    position: Position
    resources: set[Position]
    hazards: set[Position]
    energy: int
    step: int = 0
    consequence: str = "start"
    terminated: bool = False
    truncated: bool = False
