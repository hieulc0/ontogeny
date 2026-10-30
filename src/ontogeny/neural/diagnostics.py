"""Controlled policy-gradient diagnostics on externally bounded cue tasks."""

import math
import random
import statistics
from typing import Any

from ontogeny.neural.encoding import encode_policy
from ontogeny.world.canary import CueWorld


def distribution(owner: Any, observation: Any, torch: Any, actions: int) -> Any:
    owner.model.eval()
    with torch.no_grad():
        logits = owner.model(torch.tensor([encode_policy(observation)], device="cuda"))[0, :actions]
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite diagnostic logits")
        return logits.softmax(-1)


def sample(probabilities: list[float], rng: random.Random) -> int:
    value = rng.random()
    cumulative = 0.0
    for action, p in enumerate(probabilities):
        cumulative += p
        if value < cumulative:
            return action
    return len(probabilities) - 1


def update(owner: Any, episode: list, torch: Any) -> dict:
    baseline = owner.external.setdefault("canary_baseline", [0.0] * 3)
    returns, value = [], 0.0
    for _, _, reward in reversed(episode):
        value = reward + 0.97 * value
        returns.append(value)
    returns.reverse()
    advantages = [value - baseline[i] for i, value in enumerate(returns)]
    owner.model.train()
    logits = owner.model(torch.tensor([encode_policy(o) for o, _, _ in episode], device="cuda"))[
        :, :2
    ]
    actions = torch.tensor([a for _, a, _ in episode], device="cuda")
    logp = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
    loss = -(logp * torch.tensor(advantages, device="cuda")).mean()
    owner.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(owner.model.parameters(), 1.0)
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise RuntimeError("nonfinite canary update")
    owner.optimizer.step()
    owner.metadata["steps"] += 1
    for i, value in enumerate(returns):
        baseline[i] = 0.9 * baseline[i] + 0.1 * value
    owner.external["canary_experience"] = episode
    record = dict(
        loss=loss.item(), gradient_norm=norm.item(), returns=returns, advantages=advantages
    )
    owner.external.setdefault("canary_history", []).append(record)
    return record


def trial(
    owner: Any, seed: int, delayed: bool, torch: Any, *, learning: bool, random_policy: bool = False
) -> dict:
    world = CueWorld(seed, delayed=delayed)
    rng = owner.external["policy_rng"] if learning else random.Random(seed + 900000)
    episode, predictions = [], []
    while True:
        observation = world.observe()
        probs = [0.5, 0.5] if random_policy else distribution(owner, observation, torch, 2).tolist()
        action = sample(probs, rng)
        feedback = world.step(action)
        predictions.append(
            dict(probabilities=probs, action=action, observation=encode_policy(observation))
        )
        episode.append((observation, action, feedback.reward))
        if feedback.done:
            break
    if learning:
        record = update(owner, episode, torch)
    else:
        record = {}
    owner.external["canary_world"] = world
    return dict(
        success=feedback.collected,
        reward=sum(x[2] for x in episode),
        entropy=statistics.mean(
            -sum(p * math.log(max(p, 1e-30)) for p in r["probabilities"]) for r in predictions
        ),
        predictions=predictions,
        **record,
    )


def qualify(rows: list[dict], training: list[dict]) -> dict:
    successes = [r["success"] for r in rows if r["arm"] == "learning"]
    differences = {}
    for control in ("random", "frozen"):
        other = [r["success"] for r in rows if r["arm"] == control]
        values = [int(a) - int(b) for a, b in zip(successes, other, strict=True)]
        mean = statistics.mean(values)
        differences[control] = dict(
            mean=mean, lower95=mean - 1.96 * statistics.stdev(values) / math.sqrt(len(values))
        )
    early = statistics.mean(r["success"] for r in training[:256])
    late = statistics.mean(r["success"] for r in training[-256:])
    passed = (
        statistics.mean(successes) >= 0.9
        and late > early + 0.1
        and all(d["lower95"] > 0.15 for d in differences.values())
    )
    return dict(
        status="PASS" if passed else "FAIL",
        accuracy=statistics.mean(successes),
        first_quarter=early,
        last_quarter=late,
        paired_differences=differences,
    )
