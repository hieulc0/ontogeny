"""GPU-only smoke admission, training and sealed evaluation; never local inference."""

import argparse
import importlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ontogeny.evaluation.baselines import HeuristicPolicy, RandomPolicy, RestPolicy
from ontogeny.evaluation.l1 import qualification
from ontogeny.learning.experience import Experience
from ontogeny.neural.adapter import require_remote
from ontogeny.neural.persistence import UPSTREAM_SHA, digest, verify_import, verify_source
from ontogeny.world.environment import CellWorld
from ontogeny.world.state import WORLD_VERSION, WorldConfig


def setup(upstream):
    require_remote()
    verify_source(upstream)
    sys.path.insert(0, str(upstream.resolve() / "src"))
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    if not torch.cuda.is_available() or torch.__version__.split("+")[0] != "2.6.0":
        raise RuntimeError("pinned torch 2.6.0 and CUDA required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("set numerical policy before Python starts")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name in (
        "model.transformer",
        "model.config",
        "training.learning_state",
        "training.optimizer_contract",
    ):
        module = importlib.import_module("adaptive_mini_gpt." + name)
        verify_import(upstream, module)
    return (
        torch,
        np,
        importlib.import_module("adaptive_mini_gpt.training.learning_state").LearningState,
    )


def new_owner(seed, torch, np, Owner):
    from ontogeny.neural.policy import PolicyNetwork

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = importlib.import_module("adaptive_mini_gpt.model.config").ModelConfig
    backbone = importlib.import_module("adaptive_mini_gpt.model.transformer").MiniGPT
    model = PolicyNetwork(
        backbone(config(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0))
    ).cuda()
    owner = Owner(
        model,
        lr=0.0005,
        external={"policy_rng": random.Random(seed)},
        metadata={
            "upstream_sha": UPSTREAM_SHA,
            "initialization": "fresh-no-historical-knowledge",
            "protocol": "l1-reinforce-v1",
            "seed": seed,
        },
    )
    membership = importlib.import_module(
        "adaptive_mini_gpt.training.optimizer_contract"
    ).optimizer_membership
    assert not any(membership(owner.model, owner.optimizer).values())
    return owner


def advance(owner):
    from ontogeny.neural.policy import PolicyCell

    cell = PolicyCell(owner, max_updates=257)
    driver = owner.external["driver"]
    world = CellWorld.restore(driver["world"])
    observation = driver["observation"]
    probs = cell.probabilities(observation)
    action = cell.act(observation)
    outcome = world.step(action)
    cell.learn(Experience(observation, action, outcome))
    owner.external["driver"] = {"world": world.snapshot(), "observation": outcome.observation}
    return {"probabilities": probs, "action": int(action), "outcome": asdict(outcome)}


def equal(a, b, torch, np):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k], torch, np) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y, torch, np) for x, y in zip(a, b, strict=True))
    return a == b


def comparable(owner):
    snapshot = owner.snapshot()
    model, optimizer, _, external, metadata = snapshot["roots"]
    return (
        model.state_dict(),
        optimizer.state_dict(),
        external,
        metadata,
        snapshot["gradients"],
        snapshot["rng"],
        model.training,
        [p.requires_grad for p in model.parameters()],
    )


def smoke(args, torch, np, Owner, owner=None):
    from ontogeny.neural.policy import PolicyCell

    if owner is None:
        owner = new_owner(42, torch, np, Owner)
        clone = new_owner(42, torch, np, Owner)
        assert equal(owner.model.state_dict(), clone.model.state_dict(), torch, np)
        del clone
    before_steps = owner.metadata["steps"]
    cell = PolicyCell(owner, max_updates=257)
    world = CellWorld(WorldConfig(max_steps=10), seed=91823)
    observation = world.observe()
    initial = {k: p.detach().clone() for k, p in owner.model.named_parameters()}
    for _ in range(9):
        action = cell.act(observation)
        outcome = world.step(action)
        cell.learn(Experience(observation, action, outcome))
        observation = outcome.observation
    owner.external["driver"] = {"world": world.snapshot(), "observation": observation}
    parent = args.output / "continuation-parent.pt"
    owner.save(parent)
    expected = advance(owner)
    expected_state = comparable(owner)
    uninterrupted = args.output / "uninterrupted.pt"
    owner.save(uninterrupted)
    assert owner.metadata["steps"] == before_steps + 1
    assert any(not torch.equal(initial[k], p) for k, p in owner.model.named_parameters())
    command = [
        sys.executable,
        "kaggle/run_l1.py",
        "--mode",
        "continue",
        "--upstream",
        str(args.upstream),
        "--output",
        str(args.output),
        "--checkpoint",
        str(parent),
    ]
    subprocess.run(command, check=True, timeout=180)
    actual = json.loads((args.output / "resumed-step.json").read_text())
    assert json.loads(json.dumps(expected)) == actual
    restored = Owner.load(args.output / "resumed.pt")
    assert equal(expected_state, comparable(restored), torch, np)
    # Further experience after reload, not just inference weights.
    for obj in (owner, restored):
        obj.external["driver"] = {
            "world": CellWorld(seed=991).snapshot(),
            "observation": CellWorld(seed=991).observe(),
        }
    owner.save(args.output / "post-admission.pt")
    return {
        "status": "PASS",
        "fresh_process_exact": True,
        "updates": owner.metadata["steps"],
        "pending_experiences_restored": 9,
        "initialization_exact": True,
        "parent_sha256": digest(parent),
        "uninterrupted_sha256": digest(uninterrupted),
    }


def episode(policy, config, seed, *, learning, deadline, log):
    world = CellWorld(config, seed=seed)
    observation = world.observe()
    energies = [observation.energy]
    actions = []
    collected = hazards = tiles = 0
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError("experiment wall-time budget exhausted")
        tiles += len(observation.tiles)
        action = policy.act(observation)
        outcome = world.step(action)
        exp = Experience(observation, action, outcome)
        if log:
            log.write(json.dumps(asdict(exp)) + "\n")
        if learning:
            policy.learn(exp)
        collected += outcome.collected
        hazards += outcome.hazard
        actions.append(int(action))
        observation = outcome.observation
        energies.append(observation.energy)
        if outcome.terminated or outcome.truncated:
            if learning:
                policy.owner.external["driver"] = {
                    "world": world.snapshot(),
                    "observation": observation,
                }
            return {
                "duration": len(actions),
                "collected": collected,
                "hazard_exposures": hazards,
                "resource_efficiency": collected / len(actions),
                "termination": "exhausted" if outcome.terminated else "horizon",
                "energy_trajectory": energies,
                "actions": actions,
                "observation_tiles": tiles,
            }


def experiment(args, spec, torch, np, Owner):
    from ontogeny.neural.policy import PolicyCell

    deadline = time.monotonic() + spec["max_seconds"]
    rows: list[dict[str, Any]] = []
    training = []
    changed = []
    config = WorldConfig()
    log = (args.output / "experiences.jsonl").open("x")
    checkpoints = []
    for seed in spec["model_seeds"]:
        owner = new_owner(seed, torch, np, Owner)
        cell = PolicyCell(owner, max_updates=spec["max_updates"])
        initial = args.output / f"{seed}-initial.pt"
        owner.save(initial)
        initial_parameters = {k: p.detach().clone() for k, p in owner.model.named_parameters()}
        for index in range(spec["train_episodes"]):
            metrics = episode(
                cell,
                config,
                spec["train_seed_start"] + index,
                learning=True,
                deadline=deadline,
                log=log,
            )
            row = {"model_seed": seed, "episode": index, **metrics}
            training.append(row)
            with (args.output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if (index + 1) % 64 == 0:
                owner.save(args.output / f"{seed}-episode-{index + 1}.pt")
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "episode": index + 1,
                            "collected": metrics["collected"],
                            "duration": metrics["duration"],
                        }
                    ),
                    flush=True,
                )
        changed.append(
            any(
                not torch.equal(initial_parameters[k], p) for k, p in owner.model.named_parameters()
            )
        )
        owner.metadata["parent_sha256"] = digest(initial)
        final = args.output / f"{seed}-final.pt"
        owner.save(final)
        audit_dir = args.output / f"{seed}-trained-continuation"
        audit_dir.mkdir()
        audit_args = argparse.Namespace(output=audit_dir, upstream=args.upstream)
        trained_admission = smoke(audit_args, torch, np, Owner, Owner.load(final))
        (audit_dir / "admission.json").write_text(json.dumps(trained_admission, indent=2))
        checkpoints.append(
            {
                "seed": seed,
                "initial": digest(initial),
                "final": digest(final),
                "updates": owner.metadata["steps"],
                "forward_examples": owner.metadata["forward_examples"],
                "gradient_examples": owner.metadata["gradient_examples"],
            }
        )
        (args.output / f"{seed}-gradients.json").write_text(
            json.dumps(owner.external["history"], indent=2)
        )
        # Each neural arm starts evaluation from its immutable checkpoint. All RNG
        # and world resets are external, and no evaluation calls learn().
        for arm in ("rest", "random", "heuristic", "frozen", "learning"):
            current = Owner.load(initial if arm == "frozen" else final)
            evaluation_state = comparable(current)
            start_weights = {k: v.clone() for k, v in current.model.state_dict().items()}
            for split, start, count in (
                ("dev", spec["dev_seed_start"], spec["dev_worlds"]),
                ("sealed", spec["sealed_seed_start"], spec["sealed_worlds"]),
            ):
                for variant in ("default", "hazard_shift"):
                    cfg = (
                        config
                        if variant == "default"
                        else WorldConfig(hazards=18, change_interval=10)
                    )
                    for world_seed in range(start, start + count):
                        policy_seed = seed * 100000 + world_seed
                        if arm == "rest":
                            policy = RestPolicy()
                        elif arm == "random":
                            policy = RandomPolicy(policy_seed)
                        elif arm == "heuristic":
                            policy = HeuristicPolicy(policy_seed)
                        else:
                            current.external["policy_rng"] = random.Random(policy_seed)
                            policy = PolicyCell(current, max_updates=spec["max_updates"])
                        metrics = episode(
                            policy, cfg, world_seed, learning=False, deadline=deadline, log=None
                        )
                        row = {
                            "model_seed": seed,
                            "arm": arm,
                            "split": split,
                            "variant": variant,
                            "world_seed": world_seed,
                            **metrics,
                        }
                        rows.append(row)
                        with (args.output / "evaluation.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
            assert equal(start_weights, current.model.state_dict(), torch, np)
            # counters/RNG can change in evaluation; optimizer and learning buffers cannot.
            assert equal(evaluation_state[1], current.optimizer.state_dict(), torch, np)
            assert current.metadata["steps"] == evaluation_state[3]["steps"]
    log.close()
    sealed = [r for r in rows if r["split"] == "sealed"]
    report = qualification(sealed, training, changed, True)
    report.update(
        {
            "checkpoints": checkpoints,
            "evaluation_episodes": len(rows),
            "training_episodes": len(training),
            "parameter_updates": changed,
            "all_outcomes": "evaluation.jsonl",
            "all_training": "training.jsonl",
        }
    )
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2))


def main():
    require_remote()
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "experiment", "continue"], default="smoke")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch, np, Owner = setup(args.upstream)
    if args.mode == "continue":
        owner = Owner.load(args.checkpoint)
        result = advance(owner)
        (args.output / "resumed-step.json").write_text(json.dumps(result))
        owner.save(args.output / "resumed.pt")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = {
        "ontogeny_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_sha": UPSTREAM_SHA,
        "world_version": WORLD_VERSION,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "gpu_count": torch.cuda.device_count(),
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        "mode": args.mode,
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    spec_path = Path("configs/l1_experiment.toml")
    spec = tomllib.loads(spec_path.read_text())
    assert spec["learning_rate"] == 0.0005 and spec["gamma"] == 0.97
    assert spec["history_capacity"] == 100 and spec["max_updates"] == 256
    (args.output / "protocol.toml").write_text(spec_path.read_text())
    started = time.monotonic()
    try:
        admission = smoke(args, torch, np, Owner)
        (args.output / "admission.json").write_text(json.dumps(admission, indent=2))
        if args.mode == "experiment":
            experiment(args, spec, torch, np, Owner)
        runtime.update(status="COMPLETE", seconds=time.monotonic() - started)
    except BaseException:
        import traceback

        (args.output / "failure.txt").write_text(traceback.format_exc())
        runtime.update(status="FAILED", seconds=time.monotonic() - started)
        raise
    finally:
        (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
