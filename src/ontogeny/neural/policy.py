"""Remote-CUDA REINFORCE policy. No simulator truth enters this module."""

import importlib
import random
from typing import Any

from ontogeny.neural.adapter import require_remote
from ontogeny.neural.encoding import encode_policy
from ontogeny.world.actions import Action

require_remote()
torch = importlib.import_module("torch")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")


class PolicyNetwork(torch.nn.Module):
    def __init__(self, backbone: Any):
        super().__init__()
        self.backbone = backbone
        # Fresh task-specific head. No language logits are treated as action logits.
        self.action_head = torch.nn.Linear(backbone.config.vocab_size, len(Action))
        torch.nn.init.normal_(self.action_head.weight, std=0.01)
        torch.nn.init.zeros_(self.action_head.bias)

    def forward(self, tokens):
        logits, _ = self.backbone(tokens)
        return self.action_head(logits[:, -1, :])


class PolicyCell:
    def __init__(self, owner: Any, *, max_updates: int):
        self.owner = owner
        self.max_updates = max_updates
        owner.external.setdefault("policy_rng", random.Random(0))
        owner.external.setdefault("experience", [])
        owner.external.setdefault("baseline", [0.0] * 100)
        owner.external.setdefault("history", [])
        owner.metadata.setdefault("forward_examples", 0)
        owner.metadata.setdefault("gradient_examples", 0)

    def probabilities(self, observation):
        self.owner.model.eval()
        with torch.no_grad():
            logits = self.owner.model(torch.tensor([encode_policy(observation)], device="cuda"))
            if logits.shape != (1, 7) or not torch.isfinite(logits).all():
                raise RuntimeError("invalid action logits")
            probs = logits.softmax(-1)[0].tolist()
        # All seven actions are legal. Boundary moves and failed interaction have
        # real costs; masking them would supply a privileged behavior rule.
        self.owner.metadata["forward_examples"] += 1
        return probs

    def act(self, observation):
        probs = self.probabilities(observation)
        draw = self.owner.external["policy_rng"].random()
        total = 0.0
        for i, p in enumerate(probs):
            total += p
            if draw < total:
                return Action(i)
        return Action.REST  # Floating-point cumulative roundoff only.

    def learn(self, experience):
        self.owner.external["experience"].append(experience)
        if experience.outcome.terminated or experience.outcome.truncated:
            self.update()

    def update(self):
        owner = self.owner
        if owner.metadata["steps"] >= self.max_updates:
            raise RuntimeError("update budget exhausted")
        episode = owner.external["experience"]
        if not episode or len(episode) > 100:
            raise RuntimeError("invalid bounded episode buffer")
        returns = []
        value = 0.0
        for exp in reversed(episode):
            value = exp.outcome.energy_delta / 15.0 + 0.97 * value
            returns.append(value)
        returns.reverse()
        baseline = owner.external["baseline"]
        advantages = [r - baseline[i] for i, r in enumerate(returns)]
        tokens = torch.tensor([encode_policy(e.observation) for e in episode], device="cuda")
        actions = torch.tensor([int(e.action) for e in episode], device="cuda")
        owner.model.train()
        logits = owner.model(tokens)
        log_probs = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
        loss = -(log_probs * torch.tensor(advantages, device="cuda")).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite policy loss")
        owner.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        backbone_grad = sum(
            float(p.grad.square().sum().item())
            for p in owner.model.backbone.parameters()
            if p.grad is not None
        )
        head_grad = sum(
            float(p.grad.square().sum().item())
            for p in owner.model.action_head.parameters()
            if p.grad is not None
        )
        norm = torch.nn.utils.clip_grad_norm_(owner.model.parameters(), 1.0)
        if not torch.isfinite(norm) or not backbone_grad > 0 or not head_grad > 0:
            raise RuntimeError("nonfinite or absent neural gradient")
        owner.optimizer.step()
        owner.metadata["steps"] += 1
        owner.metadata["gradient_examples"] += len(episode)
        owner.external["history"].append(
            {
                "step": owner.metadata["steps"],
                "loss": loss.item(),
                "gradient_norm": norm.item(),
                "backbone_gradient_squared": backbone_grad,
                "head_gradient_squared": head_grad,
            }
        )
        for i, r in enumerate(returns):
            baseline[i] = 0.9 * baseline[i] + 0.1 * r
        owner.external["recent_experience"] = list(episode)
        owner.external["experience"] = []
