"""A bounded independent next-token trainer; no CellWorld or organism dependency."""

from torch import Tensor

from mini_gpt.learning.state import LearningState


def train_tokens(
    owner: LearningState, tokens: Tensor, targets: Tensor, *, max_updates: int
) -> dict:
    if owner.metadata["steps"] >= max_updates:
        raise RuntimeError("update budget exhausted")
    owner.model.train()
    _, loss = owner.model(tokens, targets)
    owner.backward(loss)
    norm = owner.step()
    return {"step": owner.metadata["steps"], "loss": float(loss.detach()), "gradient_norm": norm}
