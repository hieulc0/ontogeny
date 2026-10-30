"""Remote-only reference branch executed with the immutable historical source paths."""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("remote CUDA required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from adaptive_mini_gpt.training.learning_state import LearningState

    from ontogeny.execution.tasks import state
    from ontogeny.learning.experience import Experience
    from ontogeny.neural.policy import PolicyCell
    from ontogeny.world.environment import CellWorld
    from ontogeny.world.state import WorldConfig

    args.output.mkdir()
    owner = LearningState.load(args.checkpoint)
    torch.save(state(owner), args.output / "before.pt")
    if not owner.optimizer.state or owner.metadata["steps"] < 1:
        raise RuntimeError("historical test requires nonempty optimizer history")
    world = CellWorld(WorldConfig(initial_energy=100, max_energy=100, max_steps=5), seed=93021)
    observation = world.observe()
    owner.external["driver"] = {"world": world.snapshot(), "observation": observation}

    class RecordedPolicy(PolicyCell):
        def probabilities(self, observation):
            values = super().probabilities(observation)
            self.latest_probabilities = values
            return values

    cell = RecordedPolicy(owner, max_updates=3)
    rows = []
    while True:
        action = cell.act(observation)
        probs = cell.latest_probabilities
        outcome = world.step(action)
        cell.learn(Experience(observation, action, outcome))
        owner.metadata["interactions"] += 1
        owner.external["driver"] = {"world": world.snapshot(), "observation": outcome.observation}
        rows.append(
            {"probabilities": probs, "action": int(action), "energy": outcome.observation.energy}
        )
        observation = outcome.observation
        if outcome.terminated or outcome.truncated:
            break
    owner.metadata["episode_cursor"] += 1
    torch.save(state(owner), args.output / "after.pt")
    (args.output / "trace.json").write_text(json.dumps(rows))
    # Keep these RNG families explicitly present in the reference process.
    if not isinstance(owner.external["policy_rng"], random.Random) or np.__version__ != "2.1.3":
        raise RuntimeError("reference RNG/runtime mismatch")


if __name__ == "__main__":
    main()
