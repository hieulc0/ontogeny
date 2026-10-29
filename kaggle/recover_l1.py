"""Recover fixed final checkpoints from failed experiment v1; NO retraining/tuning."""

import argparse
import json
import random
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from run_l1 import comparable, episode, equal, setup, smoke

from ontogeny.evaluation.baselines import HeuristicPolicy, RandomPolicy, RestPolicy
from ontogeny.evaluation.l1 import qualification
from ontogeny.neural.adapter import require_remote
from ontogeny.neural.persistence import digest
from ontogeny.world.state import WorldConfig


def main():
    require_remote()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch, np, Owner = setup(args.upstream)
    from ontogeny.neural.policy import PolicyCell

    candidates = list(Path("/kaggle/input").rglob("44-final.pt"))
    if not candidates:
        archives = list(Path("/kaggle/input").rglob("checkpoint_bundle.zip"))
        if len(archives) != 1:
            raise RuntimeError("verified recovery archive unavailable")
        identity = json.loads(
            Path("experiments/l1_single_cell/experiment-v1/recovery-input.json").read_text()
        )
        if digest(archives[0]) != identity["archive_sha256"]:
            raise RuntimeError("recovery archive hash mismatch")
        extracted = Path("/tmp/ontogeny-recovery-input")
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(extracted)
        candidates = list(extracted.rglob("44-final.pt"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one immutable source bundle, found {candidates}")
    source = candidates[0].parent
    manifest = json.loads(
        Path("experiments/l1_single_cell/experiment-v1/artifacts.json").read_text()
    )
    for name, expected in manifest.items():
        if digest(source / name) != expected["sha256"]:
            raise RuntimeError(f"failed input provenance: {name}")
    if args.mode == "records":
        args.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        records = []
        for seed in (42, 43, 44):
            owner = Owner.load(source / f"{seed}-final.pt")
            assert owner.metadata["steps"] == 256
            history = owner.external["history"]
            assert len(history) == 256
            if (source / f"{seed}-gradients.json").exists():
                assert history == json.loads((source / f"{seed}-gradients.json").read_text())
            (args.output / f"{seed}-gradients.json").write_text(json.dumps(history, indent=2))
            records.append(
                {
                    "seed": seed,
                    "checkpoint_sha256": digest(source / f"{seed}-final.pt"),
                    "parameters": sum(p.numel() for p in owner.model.parameters()),
                    "head_parameters": sum(p.numel() for p in owner.model.action_head.parameters()),
                    "updates": owner.metadata["steps"],
                    "gradient_examples": owner.metadata["gradient_examples"],
                }
            )
            assert time.monotonic() - started < 60
        failed_audit = Owner.load(source / "44-trained-continuation/continuation-parent.pt")
        (args.output / "records.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "records": records,
                    "failed_44_audit_updates": failed_audit.metadata["steps"],
                    "seconds": time.monotonic() - started,
                    "source_sha": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "new_optimizer_updates": 0,
                },
                indent=2,
            )
        )
        return
    shutil.copytree(source, args.output)
    (args.output / "failure.txt").rename(args.output / "original-failure.txt")
    original_runtime = json.loads((args.output / "runtime.json").read_text())
    (args.output / "runtime.json").rename(args.output / "training-runtime.json")
    runtime = {
        **original_runtime,
        "status": "RECOVERING",
        "ontogeny_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_source_sha": original_runtime["ontogeny_sha"],
        "recovery_packages": subprocess.check_output(
            [__import__("sys").executable, "-m", "pip", "freeze"], text=True
        ),
        "training_seconds": original_runtime["seconds"],
        "recovery": "unchanged final models; missing audit/evaluation only",
    }
    started = time.monotonic()
    deadline = started + 1800
    try:
        broken = args.output / "44-trained-continuation"
        broken.rename(args.output / "44-trained-continuation-failed")
        broken.mkdir()
        child_args = argparse.Namespace(output=broken, upstream=args.upstream)
        audit = smoke(child_args, torch, np, Owner, Owner.load(args.output / "44-final.pt"))
        (broken / "admission.json").write_text(json.dumps(audit, indent=2))
        for seed in (42, 43, 44):
            audit_dir = args.output / f"{seed}-trained-continuation"
            assert json.loads((audit_dir / "admission.json").read_text())["status"] == "PASS"
            assert all(json.loads((audit_dir / "continuation-fields.json").read_text()).values())
        rows = [json.loads(s) for s in (args.output / "evaluation.jsonl").read_text().splitlines()]
        assert len(rows) == 1600 and {r["model_seed"] for r in rows} == {42, 43}
        # Complete exactly the missing initialization. No policy or checkpoint selection.
        for arm in ("rest", "random", "heuristic", "frozen", "learning"):
            current = Owner.load(
                args.output / ("44-initial.pt" if arm == "frozen" else "44-final.pt")
            )
            before = comparable(current)
            for split, start, count in [("dev", 20000, 16), ("sealed", 30000, 64)]:
                for variant in ("default", "hazard_shift"):
                    config = (
                        WorldConfig()
                        if variant == "default"
                        else WorldConfig(hazards=18, change_interval=10)
                    )
                    for world_seed in range(start, start + count):
                        seed = 44 * 100000 + world_seed
                        if arm == "rest":
                            policy = RestPolicy()
                        elif arm == "random":
                            policy = RandomPolicy(seed)
                        elif arm == "heuristic":
                            policy = HeuristicPolicy(seed)
                        else:
                            current.external["policy_rng"] = random.Random(seed)
                            policy = PolicyCell(current, max_updates=256)
                        metrics = episode(
                            policy, config, world_seed, learning=False, deadline=deadline, log=None
                        )
                        row = dict(
                            model_seed=44,
                            arm=arm,
                            split=split,
                            variant=variant,
                            world_seed=world_seed,
                            **metrics,
                        )
                        rows.append(row)
                        with (args.output / "evaluation.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
            assert equal(before[0], current.model.state_dict(), torch, np)
            assert equal(before[1], current.optimizer.state_dict(), torch, np)
            assert before[3]["steps"] == current.metadata["steps"]
        training = [
            json.loads(s) for s in (args.output / "training.jsonl").read_text().splitlines()
        ]
        changed, checkpoints = [], []
        for seed in (42, 43, 44):
            initial_path, final_path = [
                args.output / f"{seed}-{s}.pt" for s in ("initial", "final")
            ]
            initial = Owner.load(initial_path)
            final = Owner.load(final_path)
            changed.append(
                not equal(initial.model.state_dict(), final.model.state_dict(), torch, np)
            )
            checkpoints.append(
                dict(
                    seed=seed,
                    initial=digest(initial_path),
                    final=digest(final_path),
                    updates=final.metadata["steps"],
                    forward_examples=final.metadata["forward_examples"],
                    gradient_examples=final.metadata["gradient_examples"],
                )
            )
        report = qualification([r for r in rows if r["split"] == "sealed"], training, changed, True)
        report.update(
            checkpoints=checkpoints,
            evaluation_episodes=len(rows),
            training_episodes=len(training),
            parameter_updates=changed,
        )
        (args.output / "qualification.json").write_text(json.dumps(report, indent=2))
        runtime.update(status="COMPLETE", seconds=time.monotonic() - started)
    except BaseException:
        import traceback

        (args.output / "recovery-failure.txt").write_text(traceback.format_exc())
        runtime.update(status="FAILED", seconds=time.monotonic() - started)
        raise
    finally:
        (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
