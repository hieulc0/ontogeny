"""No torch imports or neural computation until the explicit remote constructor."""

import importlib
import os
import random
from typing import Any

from ontogeny.learning.experience import Experience
from ontogeny.neural.encoding import encode
from ontogeny.world.actions import Action
from ontogeny.world.observations import Observation


def require_remote() -> None:
    if not (os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("COLAB_RELEASE_TAG")):
        raise RuntimeError("neural execution requires Kaggle or Colab CUDA; no local fallback")


class ConsequenceCell:
    """Experimental immediate-energy predictor, one CE objective, no survival planner.

    Owner must be an upstream LearningState with a MiniGPT model. The canonical
    concept-specific owner.step is deliberately not used for this new objective.
    """

    def __init__(self, owner: Any, *, seed: int, max_updates: int, learning: bool):
        require_remote()
        torch = importlib.import_module("torch")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        self.torch = torch
        self.owner = owner
        self.learning = learning
        self.max_updates = max_updates
        owner.external.setdefault("policy_rng", random.Random(seed))
        owner.external.setdefault("experiences", [])
        owner.metadata.setdefault("ontogeny_updates", 0)
        owner.metadata.setdefault("prediction_squared_error", 0.0)
        owner.metadata.setdefault("predictions", 0)
        # v1 outcome bins cover default world's exact energy deltas [-8, 15].
        self.values = torch.arange(-8, 16, device="cuda", dtype=torch.float32)

    def act(self, observation: Observation) -> Action:
        t = self.torch
        sequences = [encode(observation, a) for a in Action]
        self.owner.model.eval()
        with t.no_grad():
            logits, _ = self.owner.model(t.tensor(sequences, device="cuda"))
            expectations = logits[:, -1, :24].softmax(-1) @ self.values
        self._predictions = expectations.tolist()
        rng = self.owner.external["policy_rng"]
        if rng.random() < 0.15:
            return rng.choice(list(Action))
        return Action(int(expectations.argmax().item()))

    def learn(self, experience: Experience) -> None:
        if not self.learning:
            return
        owner, t = self.owner, self.torch
        if owner.metadata["ontogeny_updates"] >= self.max_updates:
            raise RuntimeError("gradient budget exhausted")
        delta = experience.outcome.energy_delta
        if not -8 <= delta <= 15:
            raise ValueError("outcome outside preregistered v1 envelope")
        owner.external["experiences"].append(experience)
        owner.external["experiences"] = owner.external["experiences"][-128:]
        owner.model.train()
        owner.optimizer.zero_grad(set_to_none=True)
        logits, _ = owner.model(
            t.tensor([encode(experience.observation, experience.action)], device="cuda")
        )
        loss = t.nn.functional.cross_entropy(
            logits[:, -1, :24], t.tensor([delta + 8], device="cuda")
        )
        if not t.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        loss.backward()
        norm = t.nn.utils.clip_grad_norm_(owner.model.parameters(), 1.0)
        if not t.isfinite(norm):
            raise RuntimeError("nonfinite gradients")
        owner.optimizer.step()
        owner.metadata["steps"] += 1
        owner.metadata["ontogeny_updates"] += 1
        owner.metadata["prediction_squared_error"] += (
            self._predictions[int(experience.action)] - delta
        ) ** 2
        owner.metadata["predictions"] += 1
