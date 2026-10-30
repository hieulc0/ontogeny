"""Bounded episode tasks using the unchanged canonical policy and learner owner."""

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ontogeny.execution.artifacts import confined, write_json
from ontogeny.execution.contracts import ExecutionContext, Request
from ontogeny.execution.worker import remote_runtime
from ontogeny.learning.experience import Experience
from ontogeny.neural.encoding import encode_policy
from ontogeny.neural.persistence import digest
from ontogeny.world.environment import CellWorld
from ontogeny.world.state import WorldConfig


def equal(left: Any, right: Any, torch: Any, np: Any) -> bool:
    if isinstance(left, random.Random):
        return isinstance(right, random.Random) and left.getstate() == right.getstate()
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equal(left[k], right[k], torch, np) for k in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            equal(a, b, torch, np) for a, b in zip(left, right, strict=True)
        )
    return left == right


def state(owner: Any) -> tuple:
    snapshot = owner.snapshot()
    model, optimizer, scheduler, external, metadata = snapshot["roots"]
    return (
        model.state_dict(),
        optimizer.state_dict(),
        scheduler,
        external,
        metadata,
        snapshot["gradients"],
        snapshot["rng"],
        model.training,
        [p.requires_grad for p in model.parameters()],
    )


def bind(owner: Any, request: Request) -> None:
    lineage = {
        "run_id": request.run_id,
        "trial_id": request.trial_id,
        "arm_id": request.arm_id,
        "protocol_sha256": request.protocol_sha256,
        "source": request.source,
        "objective": request.objective,
        "world": request.world,
        "limits": vars(request.limits),
    }
    if "execution_lineage" in owner.metadata:
        previous = owner.metadata["execution_lineage"]
        if previous["source"] != request.source:
            raise RuntimeError("checkpoint source lineage mismatch")
        if request.operation != "evaluate" and previous != lineage:
            raise RuntimeError("resume may not change lineage, objective or global budgets")
    elif request.initialization["kind"] != "fresh":
        raise RuntimeError("checkpoint lacks this worker full-state lineage")
    else:
        owner.metadata["execution_lineage"] = lineage
        owner.metadata["interactions"] = 0
        owner.metadata["episode_cursor"] = 0


def act_step(
    owner: Any,
    cell: Any,
    request: Request,
    torch: Any,
    output: Path,
    *,
    learning: bool,
    deadline: float,
) -> dict:
    if time.monotonic() >= deadline:
        raise TimeoutError("cooperative task deadline")
    if learning and owner.metadata["interactions"] >= request.limits.interactions:
        raise RuntimeError("global interaction budget exhausted")
    driver = owner.external["driver"]
    world = CellWorld.restore(driver["world"])
    observation = driver["observation"]
    # Capture the actual policy sample without an additional sampling/forward call.
    rng = owner.external["policy_rng"]
    rng_before = rng.getstate()
    action = cell.act(observation)
    probs = cell.latest_probabilities
    outcome = world.step(action)
    experience = Experience(observation, action, outcome)
    if learning:
        cell.learn(experience)
        owner.metadata["interactions"] += 1
    owner.external["driver"] = {"world": world.snapshot(), "observation": outcome.observation}
    if torch.cuda.max_memory_allocated() > request.limits.torch_memory_bytes:
        raise RuntimeError("Torch allocation budget exceeded")
    row = {
        "run_id": request.run_id,
        "trial_id": request.trial_id,
        "arm_id": request.arm_id,
        "split": request.split,
        "episode": owner.metadata["episode_cursor"],
        "step": world.snapshot()["state"].step,
        "observation": encode_policy(observation),
        "probabilities": probs,
        "action": int(action),
        "rng_before_sha256": hashlib.sha256(repr(rng_before).encode()).hexdigest(),
        "energy_delta": outcome.energy_delta,
        "energy": outcome.observation.energy,
        "collected": outcome.collected,
        "hazard": outcome.hazard,
        "terminated": outcome.terminated,
        "truncated": outcome.truncated,
        "observation_tiles": len(observation.tiles),
    }
    with (output / "predictions.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


def run_task(
    request: Request,
    context: ExecutionContext,
    upstream: Path,
    inputs: Path,
    output: Path,
    *,
    resume_probe: Path | None = None,
    expected_hash: str | None = None,
) -> None:
    remote_runtime(request, context, upstream)
    from ontogeny.neural.policy import PolicyCell
    from ontogeny.neural.runtime import fresh_owner, initialize

    torch, np, Owner = initialize(upstream)
    torch.cuda.reset_peak_memory_stats()
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "predictions.jsonl").touch(exist_ok=True)
    (output / "learning.jsonl").touch(exist_ok=True)
    start = time.monotonic()
    deadline = start + request.limits.wall_seconds
    if resume_probe is not None:
        if digest(resume_probe) != expected_hash:
            raise RuntimeError("continuation parent corrupted")
        owner = Owner.load(resume_probe)
    elif request.initialization["kind"] == "fresh":
        owner = fresh_owner(request.initialization["seed"], torch, np, Owner)
    else:
        owner = Owner.load(confined(inputs, request.inputs[0].path))
    bind(owner, request)

    class RecordedCell(PolicyCell):
        def probabilities(self, observation):
            probabilities = super().probabilities(observation)
            self.latest_probabilities = list(probabilities)
            return probabilities

    cell = RecordedCell(owner, max_updates=request.limits.updates)
    initial_steps = owner.metadata["steps"]
    checkpoint_records = []

    def save(name: str, parent: str | None = None) -> Path:
        path = output / "checkpoints" / name
        owner.save(path)
        checkpoint_records.append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": digest(path),
                "parent_sha256": parent,
                "updates": owner.metadata["steps"],
                "interactions": owner.metadata["interactions"],
            }
        )
        return path

    if resume_probe is not None:
        row = act_step(owner, cell, request, torch, output, learning=True, deadline=deadline)
        torch.save(state(owner), output / "resumed_state.pt")
        write_json(output / "resumed_step.json", row)
        save("resumed.pt", expected_hash)
        with (output / "learning.jsonl").open("a") as stream:
            stream.write(json.dumps(owner.external["history"][-1]) + "\n")
        return
    rows = []
    checkpoint_parent = request.inputs[0].sha256 if request.inputs else None
    continuation = None
    if request.operation == "continuation_check":
        world = CellWorld(WorldConfig(**request.world["config"]), seed=request.world["seed"])
        owner.external["driver"] = {"world": world.snapshot(), "observation": world.observe()}
        initial = save("initial.pt", checkpoint_parent)
        for _ in range(9):
            rows.append(
                act_step(owner, cell, request, torch, output, learning=True, deadline=deadline)
            )
        if len(owner.external["experience"]) != 9:
            raise RuntimeError("pending experience fixture did not reach intended boundary")
        parent = save("pending.pt", checkpoint_parent)
        expected_row = act_step(
            owner, cell, request, torch, output, learning=True, deadline=deadline
        )
        rows.append(expected_row)
        expected_state = state(owner)
        save("uninterrupted.pt", digest(parent))
        child = output / "resumed"
        child.mkdir()
        command = [
            sys.executable,
            "-m",
            "ontogeny.execution.tasks",
            "--request",
            str(output / "request.json"),
            "--context",
            json.dumps(vars(context)),
            "--upstream",
            str(upstream),
            "--inputs",
            str(inputs),
            "--output",
            str(child),
            "--resume-probe",
            str(parent),
            "--expected-hash",
            digest(parent),
        ]
        subprocess.run(command, check=True, timeout=max(1, int(deadline - time.monotonic())))
        actual_row = json.loads((child / "resumed_step.json").read_text())
        # Exclude no state fields: global RNG was captured before loading the other branch.
        actual_state = torch.load(child / "resumed_state.pt", weights_only=False)
        matches = {
            name: equal(a, b, torch, np)
            for name, a, b in zip(
                (
                    "model",
                    "optimizer",
                    "scheduler",
                    "external",
                    "metadata",
                    "gradients",
                    "rng",
                    "training_mode",
                    "requires_grad",
                ),
                expected_state,
                actual_state,
                strict=True,
            )
        }
        if expected_row != actual_row or not all(matches.values()):
            raise RuntimeError(f"fresh-process continuation mismatch: {matches}")
        # Independently run the historical runner/policy from the same initial state.
        # This tests instrumentation equivalence, rather than mirroring the new loop.
        from ontogeny.organism.cell import run_episode

        reference = Owner.load(initial)
        reference_world = CellWorld.restore(reference.external["driver"]["world"])
        reference_policy = PolicyCell(reference, max_updates=request.limits.updates)
        reference_actions = []

        def record_reference(experience):
            reference_actions.append(int(experience.action))
            reference.metadata["interactions"] += 1
            reference.external["driver"] = {
                "world": reference_world.snapshot(),
                "observation": experience.outcome.observation,
            }
            with (output / "reference_experience.jsonl").open("a") as stream:
                stream.write(json.dumps(asdict(experience)) + "\n")

        run_episode(reference_world, reference_policy, learning=True, record=record_reference)
        reference_matches = equal(expected_state, state(reference), torch, np)
        if not reference_matches or reference_actions != [r["action"] for r in rows]:
            raise RuntimeError("common worker altered historical learner behavior")
        reference_path = output / "checkpoints" / "reference.pt"
        reference.save(reference_path)
        checkpoint_records.append(
            {
                "path": "checkpoints/reference.pt",
                "sha256": digest(reference_path),
                "parent_sha256": digest(initial),
                "updates": reference.metadata["steps"],
                "interactions": reference.metadata["interactions"],
            }
        )
        with (output / "reference_learning.jsonl").open("a") as stream:
            stream.write(json.dumps(reference.external["history"][-1]) + "\n")
        continuation = {
            "exact": True,
            "historical_runner_exact": reference_matches,
            "groups": matches,
            "pending_experiences": 9,
            "sample_and_transition_exact": expected_row == actual_row,
        }
        write_json(output / "continuation.json", continuation)
        # The resumed copy is a discarded audit branch, never substituted for the learner.
        actual_updates = 3
        physical_interactions = len(rows) + 1 + len(reference_actions)
    else:
        before_evaluation = state(owner) if request.operation == "evaluate" else None
        cursor = owner.metadata["episode_cursor"] if request.operation == "train" else 0
        while cursor < request.world["episodes"]:
            driver = owner.external.get("driver")
            if (
                request.operation == "evaluate"
                or driver is None
                or (driver["world"]["state"].terminated or driver["world"]["state"].truncated)
            ):
                world = CellWorld(
                    WorldConfig(**request.world["config"]), seed=request.world["seed"] + cursor
                )
                owner.external["driver"] = {
                    "world": world.snapshot(),
                    "observation": world.observe(),
                }
            while True:
                row = act_step(
                    owner,
                    cell,
                    request,
                    torch,
                    output,
                    learning=request.operation == "train",
                    deadline=deadline,
                )
                rows.append(row)
                if len(rows) > request.limits.interactions:
                    raise RuntimeError("physical interaction budget exceeded")
                if row["terminated"] or row["truncated"]:
                    break
            cursor += 1
            if request.operation == "train":
                owner.metadata["episode_cursor"] = cursor
                checkpoint = save(f"episode-{cursor}.pt", checkpoint_parent)
                checkpoint_parent = digest(checkpoint)
        if before_evaluation is not None:
            after = state(owner)
            if not all(equal(before_evaluation[i], after[i], torch, np) for i in (0, 1, 2, 5)):
                raise RuntimeError("evaluation mutated model, optimizer, scheduler or gradients")
            for key, value in before_evaluation[3].items():
                if key not in ("policy_rng", "driver") and not equal(
                    value, after[3][key], torch, np
                ):
                    raise RuntimeError("evaluation mutated retained learning state")
            for key, value in before_evaluation[4].items():
                if key != "forward_examples" and not equal(value, after[4][key], torch, np):
                    raise RuntimeError("evaluation mutated learning counters/metadata")
            if owner.metadata["steps"] != initial_steps:
                raise RuntimeError("evaluation performed learning")
            path = output / "checkpoints" / "evaluated.pt"
            shutil.copyfile(confined(inputs, request.inputs[0].path), path)
            checkpoint_records.append(
                {
                    "path": "checkpoints/evaluated.pt",
                    "sha256": digest(path),
                    "parent_sha256": request.inputs[0].sha256,
                    "updates": owner.metadata["steps"],
                    "interactions": owner.metadata["interactions"],
                }
            )
        actual_updates = owner.metadata["steps"] - initial_steps
        physical_interactions = len(rows)
    with (output / "learning.jsonl").open("a") as log:
        new_updates = owner.metadata["steps"] - initial_steps
        for record in owner.external["history"][-new_updates:] if new_updates else []:
            if request.operation != "evaluate":
                log.write(json.dumps(record) + "\n")
    write_json(
        output / "checkpoint_manifest.json",
        {"schema_version": 1, "checkpoints": checkpoint_records},
    )
    write_json(
        output / "metrics.json",
        {
            "updates": actual_updates,
            "interactions": physical_interactions,
            "logical_updates": owner.metadata["steps"],
            "logical_interactions": owner.metadata["interactions"],
            "peak_torch_memory_bytes": torch.cuda.max_memory_allocated(),
            "collected": sum(r["collected"] for r in rows),
            "hazard_exposures": sum(r["hazard"] for r in rows),
            "observation_tiles": sum(r["observation_tiles"] for r in rows),
            "final_energy": rows[-1]["energy"] if rows else None,
            "mean_entropy": sum(
                -sum(p * math.log(max(p, 1e-30)) for p in r["probabilities"]) for r in rows
            )
            / max(1, len(rows)),
            "continuation_exact": continuation is not None,
            "task_wall_seconds": time.monotonic() - start,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("request", "upstream", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--resume-probe", type=Path)
    parser.add_argument("--expected-hash")
    args = parser.parse_args()
    run_task(
        Request.read(args.request),
        ExecutionContext(**json.loads(args.context)),
        args.upstream,
        args.inputs,
        args.output,
        resume_probe=args.resume_probe,
        expected_hash=args.expected_hash,
    )


if __name__ == "__main__":
    main()
