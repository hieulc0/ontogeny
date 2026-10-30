"""Remote-only bounded source-consolidation verification and artifact publication."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-id", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    started = time.monotonic()
    report = {"execution": "FAILED", "science": "NOT_RUN", "native_id": args.native_id}

    def run(command, name, timeout):
        with (output / (name + ".log")).open("x") as log:
            subprocess.run(
                command,
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )

    try:
        from mini_gpt.runtime import configure_determinism

        torch, np = configure_determinism()
        from mini_gpt.checkpoint.identity import file_digest

        SOURCE_REVISION = json.loads(
            (source / "experiments/integration/reference.json").read_text()
        )["source_revision"]

        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        actual_reference = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(os.environ["REFERENCE_NEURAL_SRC"]).parent,
            text=True,
        ).strip()
        if actual_reference != SOURCE_REVISION:
            raise RuntimeError("reference source mismatch")
        inventory = json.loads((source / "docs/decisions/neural_source_inventory.json").read_text())
        for name, item in inventory["files"].items():
            path = Path(os.environ["REFERENCE_NEURAL_SRC"]) / "adaptive_mini_gpt" / name
            if file_digest(path) != item["sha256"]:
                raise RuntimeError("reference component hash mismatch: " + name)
        (output / "source_inventory.json").write_text(json.dumps(inventory, indent=2))
        runtime = {
            "source_sha": revision,
            "reference_sha": actual_reference,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "visible_devices": torch.cuda.device_count(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "physical_gpus": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                text=True,
            ).splitlines(),
            "packages": sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            ),
        }
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
        os.environ["MIGRATION_TEST_RECORDS"] = str(output / "test_records")
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/mini_gpt/test_neural_gpu.py",
                "--junitxml",
                str(output / "neural-tests.xml"),
                "--basetemp",
                str(output / "neural-cases"),
            ],
            "neural-tests",
            300,
        )
        root = ET.parse(output / "neural-tests.xml").getroot()
        suites = list(root.iter("testsuite"))
        counts = {
            key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        if counts["tests"] != 6 or any(counts[key] for key in ("failures", "errors", "skipped")):
            raise RuntimeError("expected six actual neural checks with no skipped/failing cases")
        report["neural_tests"] = "PASS"
        report["neural_test_count"] = 6
        # Independent and embodied workers must run without reference source on PYTHONPATH.
        os.environ["PYTHONPATH"] = str(source / "src")
        config = {
            "vocab_size": 32,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 32,
            "context_length": 16,
            "dropout": 0.0,
        }
        (output / "model.json").write_text(json.dumps(config))
        (output / "batches.json").write_text(
            json.dumps([{"tokens": [[1, 2, 3, 4]], "targets": [[2, 3, 4, 5]]}])
        )
        run(
            [
                sys.executable,
                "-m",
                "mini_gpt.cli",
                "--config",
                str(output / "model.json"),
                "--data",
                str(output / "batches.json"),
                "--updates",
                "2",
                "--output",
                str(output / "independent"),
            ],
            "independent",
            60,
        )
        from mini_gpt.learning.state import LearningState

        independent = LearningState.load(output / "independent/step-2.pt")
        if independent.metadata["steps"] != 2 or not independent.optimizer.state:
            raise RuntimeError("independent trainer did not persist genuine updates")
        report["independent_training"] = "PASS"
        from ontogeny.execution.artifacts import verify_bundle, write_json
        from ontogeny.execution.cli import qualification_request
        from ontogeny.execution.contracts import ExecutionContext

        request = qualification_request(revision, file_digest(source / "docs/MIGRATION.md"))
        write_json(output / "request.json", request.to_dict())
        (output / "inputs").mkdir()
        context = ExecutionContext("kaggle", args.native_id, "attempt-1")
        run(
            [
                sys.executable,
                "-m",
                "ontogeny.execution.worker",
                "--request",
                str(output / "request.json"),
                "--context",
                json.dumps(vars(context)),
                "--inputs",
                str(output / "inputs"),
                "--output",
                str(output / "organism"),
            ],
            "organism",
            120,
        )
        report["organism_receipt"] = verify_bundle(output / "organism", request)
        report["organism_training"] = "PASS"
        report["execution"] = "SUCCEEDED"
        report["historical_input_sha256"] = os.environ["MIGRATION_CHECKPOINT_SHA256"]
        report["scope"] = (
            "engineering-only exact dense equivalence, trusted import and continuation"
        )
    except Exception as error:
        report["error"] = {
            "kind": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        report["wall_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(report, indent=2))
    if report["execution"] != "SUCCEEDED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
