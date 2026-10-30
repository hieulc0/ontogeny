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
