"""Pinned GPU diagnosis using training artifacts and prerequisite-gated canaries."""

import argparse
import importlib
import json
import math
import subprocess
import time
import zipfile
from pathlib import Path

from ontogeny.neural.diagnostics import distribution, qualify, trial
from ontogeny.neural.encoding import encode_policy
from ontogeny.neural.persistence import digest
from ontogeny.neural.runtime import fresh_owner, initialize
from ontogeny.world.canary import CueWorld
from ontogeny.world.observations import Observation


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def original_source():
    archives = list(Path("/kaggle/input").rglob("checkpoint_bundle.zip"))
    candidates = list(Path("/kaggle/input").rglob("44-final.pt"))
    if not candidates:
        require(len(archives) == 1, "expected immutable input archive")
        identity = json.loads(
            Path("experiments/l1_single_cell/experiment-v1/recovery-input.json").read_text()
        )
        require(digest(archives[0]) == identity["archive_sha256"], "archive mismatch")
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall("/tmp/diagnostic-input")
        candidates = list(Path("/tmp/diagnostic-input").rglob("44-final.pt"))
    require(len(candidates) == 1, "ambiguous checkpoint source")
    root = candidates[0].parent
    manifest = json.loads(
        Path("experiments/l1_single_cell/experiment-v1/artifacts.json").read_text()
    )
    for seed in (42, 43, 44):
        for stage in ("initial", "final"):
            name = f"{seed}-{stage}.pt"
            require(digest(root / name) == manifest[name]["sha256"], "checkpoint lineage mismatch")
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch, np, Owner = initialize(args.upstream)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        root = original_source()
        # Only original TRAINING observations, never evaluation rows or new sealed rollouts.
        require(
            digest(root / "experiences.jsonl")
            == json.loads(
                Path("experiments/l1_single_cell/experiment-v1/artifacts.json").read_text()
            )["experiences.jsonl"]["sha256"],
            "training experience hash mismatch",
        )
        entries = [json.loads(s) for s in (root / "experiences.jsonl").read_text().splitlines()]
        episodes, current = [], []
        for e in entries:
            current.append(e)
            if e["outcome"]["terminated"] or e["outcome"]["truncated"]:
                episodes.append(current)
                current = []
        require(len(episodes) == 768 and not current, "incomplete training history")
        diagnostics = []
        for index, seed in enumerate((42, 43, 44)):
            initial = Owner.load(root / f"{seed}-initial.pt")
            final = Owner.load(root / f"{seed}-final.pt")
            changes = []
            for (name, a), (_, b) in zip(
                initial.model.named_parameters(), final.model.named_parameters(), strict=True
            ):
                changes.append(
                    dict(
                        name=name,
                        change_norm=(b - a).norm().item(),
                        optimizer_state_present=b in final.optimizer.state,
                    )
                )
            data = [e for ep in episodes[index * 256 : (index + 1) * 256] for e in ep]
            chosen = data[:128] + data[-128:]
            endpoint = []
            for label, owner in [("initial", initial), ("final", final)]:
                for e in chosen:
                    o = e["observation"]
                    obs = Observation(
                        tuple(tuple(t) for t in o["tiles"]),
                        o["energy"],
                        o["consequence"],
                        o["radius"],
                    )
                    p = distribution(owner, obs, torch, 7).tolist()
                    endpoint.append(
                        dict(
                            checkpoint=label,
                            observation=encode_policy(obs),
                            probabilities=p,
                            entropy=-sum(x * math.log(max(x, 1e-30)) for x in p),
                        )
                    )
            diagnostics.append(dict(seed=seed, parameters=changes, endpoints=endpoint))
        (args.output / "original_diagnosis.json").write_text(json.dumps(diagnostics))
        probe = fresh_owner(91, torch, np, Owner)
        obs = CueWorld(6).observe()
        tokens = torch.tensor([encode_policy(obs)], device="cuda")
        before = probe.model(tokens)[0].log_softmax(-1)[5].item()
        probe.optimizer.zero_grad(set_to_none=True)
        loss = -probe.model(tokens)[0].log_softmax(-1)[5]
        loss.backward()
        gradients = {
            n: p.grad.norm().item() for n, p in probe.model.named_parameters() if p.grad is not None
        }
        require(all(math.isfinite(v) for v in gradients.values()), "nonfinite probe gradients")
        require(
            any(v > 0 for n, v in gradients.items() if n.startswith("backbone.")),
            "missing backbone gradient",
        )
        require(
            any(v > 0 for n, v in gradients.items() if n.startswith("action_head.")),
            "missing action head gradient",
        )
        parameters = {n: p.detach().clone() for n, p in probe.model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(probe.model.parameters(), 1.0)
        probe.optimizer.step()
        probe.metadata["steps"] += 1
        require(
            any(
                not torch.equal(parameters[n], p)
                for n, p in probe.model.named_parameters()
                if n.startswith("backbone.")
            ),
            "optimizer did not update backbone",
        )
        require(
            any(
                not torch.equal(parameters[n], p)
                for n, p in probe.model.named_parameters()
                if n.startswith("action_head.")
            ),
            "optimizer did not update head",
        )
        probe.save(args.output / "positive-advantage.pt")
        after = probe.model(tokens)[0].log_softmax(-1)[5].item()
        require(after > before, "positive advantage failed to increase selected log probability")
        membership = importlib.import_module(
            "adaptive_mini_gpt.training.optimizer_contract"
        ).optimizer_membership
        require(
            not any(membership(probe.model, probe.optimizer).values()),
            "optimizer ownership failure",
        )
        (args.output / "gradient_test.json").write_text(
            json.dumps(
                dict(
                    before=before,
                    after=after,
                    gradients=gradients,
                    changed={
                        n: not torch.equal(parameters[n], p)
                        for n, p in probe.model.named_parameters()
                    },
                )
            )
        )
        results = {}
        for delayed in (False, True):
            task = "delayed" if delayed else "immediate"
            results[task] = []
            for seed in (42, 43, 44):
                owner = fresh_owner(seed, torch, np, Owner)
                initial = args.output / f"{task}-{seed}-initial.pt"
                owner.save(initial)
                training = []
                with (args.output / f"{task}-{seed}-training.jsonl").open("x") as log:
                    for i in range(1024):
                        require(
                            time.monotonic() - started < 1200, "diagnostic time budget exhausted"
                        )
                        row = trial(owner, 70000 + i, delayed, torch, learning=True)
                        training.append(row)
                        log.write(json.dumps(row) + "\n")
                final = args.output / f"{task}-{seed}-final.pt"
                owner.metadata["parent_sha256"] = digest(initial)
                owner.save(final)
                rows = []
                for arm, path in [("random", initial), ("frozen", initial), ("learning", final)]:
                    evaluated = Owner.load(path)
                    weights = {n: p.clone() for n, p in evaluated.model.state_dict().items()}
                    steps = evaluated.metadata["steps"]
                    for i in range(512):
                        row = trial(
                            evaluated,
                            80000 + i,
                            delayed,
                            torch,
                            learning=False,
                            random_policy=arm == "random",
                        )
                        rows.append(dict(arm=arm, **row))
                    require(steps == evaluated.metadata["steps"], "evaluation updated learner")
                    require(
                        all(
                            torch.equal(v, evaluated.model.state_dict()[n])
                            for n, v in weights.items()
                        ),
                        "evaluation changed weights",
                    )
                (args.output / f"{task}-{seed}-evaluation.json").write_text(json.dumps(rows))
                result = qualify(rows, training)
                results[task].append(dict(seed=seed, **result))
            (args.output / "canary_results.json").write_text(json.dumps(results, indent=2))
            if not all(r["status"] == "PASS" for r in results[task]):
                break
        (args.output / "completion.json").write_text(
            json.dumps(
                dict(
                    status="COMPLETE",
                    seconds=time.monotonic() - started,
                    ontogeny_sha=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(),
                    canaries=results,
                )
            )
        )
    except BaseException:
        import traceback

        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
