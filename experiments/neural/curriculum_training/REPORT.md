# Curriculum continuation attempt

**Training NOT RUN; continuation verification NOT RUN.** The existing student remains
at its prior verified two-update checkpoint. No learning-to-learn, subject competence,
architecture optimization or structural-change claim follows from these attempts.

The first attempt used source fd495f54e12711c0564e43544fa68efaae01970f. Colab supplied a
Tesla T4 and passed CUDA readiness. Uploading the 155 MiB parent checkpoint failed with
an SSL unexpected EOF before worker execution. The owned session was stopped and cleanup
confirmed. The private session log and controller journal are under
`runs/curriculum-continuation/`. No neural training began.

The separately preregistered Kaggle recovery used source
48ba66f557e8d3819af5cffa8a38d83c8b391fff and parent SHA-256
`8e4ea63d5e3c3719b30364614005b631d202dbf09cb666796f89f0fd2b35df7d`.
The checkpoint uploaded successfully as private dataset
`sparklinghydraulics0/ontogeny-curriculum-48ba66f557e8`, but the job was submitted before
Kaggle marked the dataset ready. The submission response warned that the input was
invalid and not attached; the controller incorrectly accepted the job-pushed response.
Job `sparklinghydraulics0/ontogeny-curriculum-run-48ba66f557/1` ended ERROR. Its worker
raised FileNotFoundError for the parent before loading a checkpoint or invoking training.
No update, evaluation or continuation audit ran. The controller journal and complete
Kaggle logs are under `runs/curriculum-kaggle-recovery/`. The dataset later reported ready.

The GPU quota was 2.61 h before submission and 2.56 h afterward (rounded provider values),
with a 2.5 h reserve. The provider reported a refresh timestamp of 2026-10-31 00:00 UTC;
that is a provider-reported estimate, not a guaranteed allocation. No additional attempt
is admitted by this protocol. The controller now checks dataset readiness before submitting
and treats a rejected input as a submitted job requiring reconciliation, preventing the
specific silent error above. These are local code changes, not a remote qualification.

The original protocol, failed attempts and source revisions remain preserved. The next
attempt requires a newly frozen provider plan, sufficient quota and a verified attached
checkpoint; only recovered, hash-verified artifacts can establish completion. All actual
neural execution and checkpoint loading remains remote CUDA only.
