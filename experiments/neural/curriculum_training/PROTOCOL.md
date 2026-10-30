# Curriculum continuation pilot

Preregistered before execution. Continue the existing research student, parent checkpoint
SHA-256 8e4ea63d5e3c3719b30364614005b631d202dbf09cb666796f89f0fd2b35df7d,
using the internal Mini-GPT engine. This is not the organism. Generate the starter data
with seed 42 and verify its train-batches SHA-256
f2841088feefa520c1d545326585d158e61c31c990071937140357191c49708f.
Train exactly three additional AdamW updates in one segment, with no reset of optimizer,
RNG or model, using the existing byte-answer objective and sequential dataset cursor.
Save complete segment-start and every update. Separately replay the first update in a fresh
process, comparing all nine checkpoint-state groups; its update is discarded, not part of
the continuing student. Stop on mismatch or non-finite values.

Measure answer-byte loss and teacher-forced answer-byte accuracy before and after on the
first 24 development examples from the deterministic development file, grouped by domain.
Evaluations run in separate processes and cannot alter the training state. These are
exploratory engineering measurements: one initialization, three updates, and familiar
templates cannot qualify generalization, learning to learn, or architecture improvement.
Do not inspect the test split. Preserve negative results and all full states.

Budget: three continuing plus one audit update; at most 5,000 training and audit input
bytes, 48 evaluation examples, 300 seconds worker time, 3 GiB peak Torch allocation per
process, 1 GiB result size, one T4 with one visible device. Provider platform limit 600
seconds; retain 2.5 Kaggle GPU hours and require margin 0.01 hour. One Colab assignment
with positive readiness; fallback to Kaggle only after confirmed pre-start unavailability.
No automatic retries, CPU fallback or switching an ambiguous started job. Export and verify
checkpoint bytes before releasing Colab; preserve raw failures.

## Pre-training provider recovery amendment

The first Colab T4 passed readiness, but upload of the parent checkpoint failed with an
SSL unexpected EOF before the worker started. The owned session was stopped and cleanup
confirmed. No neural update occurred. Admit one separate Kaggle continuation attempt;
there is no ambiguous active job. Preserve the same parent, dataset, updates, evaluation,
audit, memory and output limits. Quota observed 2.61 h; retain the 2.5 h reserve plus
0.01 h margin using a 340-second platform cap. Tighten worker task cap to 200 seconds
and training wall cap to 120 seconds, leaving 140 seconds for platform setup/export.
This shorter cap may cause an infrastructure failure; do not retry automatically or
report a started job as complete without verified artifacts. Check quota again immediately
before submission. Retain both attempts and do not change the scientific interpretation.

## Colab bounded-transfer attempt

The prior Colab upload sent the full 155 MiB checkpoint as one base64-encoded HTTP
request; its TLS connection closed unexpectedly. The exact server-side cause is unknown.
Admit one new Colab attempt after the user requested this repair. Upload and download
through 8 MiB parts, with the parent SHA-256 verified remotely before any neural work;
verify all result parts, archive hash and indexed file hashes locally before release.
Preserve the previous attempts. Use the same parent, training objective, three updates,
one discarded audit update, development examples and 200-second task / 3 GiB Torch /
1 GiB result limits. Provider execution is bounded at 600 seconds, with finite per-part
transport timeouts and a maximum 1,100 MiB archive. One positively verified T4 and no
CPU fallback. If allocation or staging fails, do not claim neural execution. A started
or ambiguous worker must be reconciled and must not be restarted automatically. Kaggle
fallback remains subject to the original 600-second cap and 2.5-hour reserve.
The transfer mechanism is an engineering repair, not evidence of learning-to-learn.
