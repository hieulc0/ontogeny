Selected sanitized records from the checkpoint transport qualification.
Overall INCONCLUSIVE: three origin/destination pairs PASS; Kaggle-to-Colab NOT RUN
because Colab refused a T4 assignment. L.1 remains Learning FAIL.

See ../../docs/CHECKPOINT_QUALIFICATION.md for protocol, failures, source pins,
reproduction, resource accounting and limits. qualification.json gives semantic gate
status; audit.json preserves the mechanical missing-bundle failure for the unrun pair.
completed_bundles.json contains JSON evidence only. Binary checkpoints and complete
artifact-indexed bundles remain under ignored runs/checkpoint-transport/. These selected
records are not complete bundles and cannot be passed to the bundle verifier.

protocols/ preserves the original and amended preregistrations. batch_launch.py.txt is
an immutable launcher snapshot, not a local executable. No CLI sessions, authentication
files, signed download URLs or raw credential-bearing transport traces are included.
