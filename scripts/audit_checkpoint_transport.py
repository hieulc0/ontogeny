"""Audit exported JSON and byte hashes only; never deserialize neural state locally."""

import argparse
import json
from pathlib import Path

from ontogeny.execution.artifacts import verify_bundle, write_json
from ontogeny.execution.contracts import Request
from ontogeny.neural.persistence import digest


def read(root: Path, name: str):
    return json.loads((root / name).read_text())


def audit(parent: Path, resumed: Path, evaluated: Path) -> dict:
    for root in (parent, resumed, evaluated):
        verify_bundle(root, Request.read(root / "request.json"))
    before = read(parent, "pending_state.json")
    expected = read(parent, "expected_state.json")
    after = read(resumed, "final_state.json")
    fields = (
        "step",
        "observation",
        "probabilities",
        "action",
        "rng_before_sha256",
        "energy_delta",
        "energy",
        "collected",
        "hazard",
        "terminated",
        "truncated",
        "observation_tiles",
    )
    expected_row = json.loads((parent / "predictions.jsonl").read_text().splitlines()[-1])
    actual_rows = [
        json.loads(row) for row in (resumed / "predictions.jsonl").read_text().splitlines()
    ]
    input_hash = Request.read(resumed / "request.json").inputs[0].sha256
    eval_request = Request.read(evaluated / "request.json")
    eval_metrics = read(evaluated, "metrics.json")
    result = {
        "input_sha256": input_hash,
        "input_is_exported_recovery": input_hash == digest(parent / "interrupted/recovery.pt"),
        "loaded_state_exact": read(resumed, "loaded_state.json") == before,
        "state_groups": {name: after[name] == expected[name] for name in expected},
        "next_sample_transition_exact": len(actual_rows) == 1
        and all(expected_row[name] == actual_rows[0][name] for name in fields),
        "one_update": read(resumed, "metrics.json")["updates"] == 1,
        "evaluation_same_input": eval_request.inputs[0].sha256 == input_hash,
        "evaluation_checkpoint_identical": digest(evaluated / "checkpoints/evaluated.pt")
        == input_hash,
        "evaluation_state_restored": read(evaluated, "final_state.json") == before,
        "evaluation_zero_updates": eval_metrics["updates"] == 0,
        "evaluation_20_interactions": eval_metrics["interactions"] == 20,
        "process_interruption_exact": read(parent, "interruption.json")["complete_state_exact"],
    }
    result["pass"] = all(
        all(value.values()) if isinstance(value, dict) else value
        for key, value in result.items()
        if key != "input_sha256"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    outcomes = {}
    for case in manifest["cases"]:
        name = case["origin"] + "-to-" + case["destination"]
        try:
            outcomes[name] = audit(*(Path(case[key]) for key in ("parent", "resumed", "evaluated")))
        except Exception as error:
            outcomes[name] = {"pass": False, "error": str(error), "kind": type(error).__name__}
    write_json(
        args.output, {"cases": outcomes, "all_pass": all(v["pass"] for v in outcomes.values())}
    )


if __name__ == "__main__":
    main()
