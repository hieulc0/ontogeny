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

## Colab chunked-transfer continuation

**Engineering PASS; learning-to-learn NOT QUALIFIED.** On source
`2a80a54c70697a761c8aa98e7ec4a86f4b0c0ea2`, Colab session
`ontogeny-5b19287a39ec1ff2` supplied a verified Tesla T4. The original 162,303,514-byte
parent transferred in 20 chunks; its remote full-file SHA-256 matched the declared parent.
The same student then completed three continuing updates, from step 2 to step 5, plus one
separate discarded audit update. The fresh-process repeat of step 3 matched the saved step 3
in all nine state groups: model, optimizer, scheduler, external state, metadata, gradients,
RNG, training mode and requires-grad flags. Colab's owned session was released and cleanup
confirmed after the result was recovered.

The worker processed 149 training input bytes and 48 audit input bytes. Step 3, 4 and 5
training losses were 6.0483, 5.4726 and 4.0708, respectively, on **different examples**;
these numbers are not a same-task learning curve. Peak reported training Torch allocation
was 693,401,088 bytes (approximately 661 MiB). Worker task wall time was 26.98 seconds,
excluding provider allocation, setup and artifact transfer. Runtime was Torch 2.6.0+cu124
and NumPy 2.1.3. The full 650,392,013-byte result archive was recovered in 78 chunks,
verified by archive and per-file hashes, and its checkpoints are retained under ignored
`runs/curriculum-colab-chunked/result/student/`.

On 24 held-out **development** examples (82 target bytes, teacher forcing), pooled answer
byte loss changed from 5.8343 to 3.7633 and byte accuracy from 2/82 (2.44%) to 25/82
(30.49%). This is a small before/after diagnostic with familiar templates, one student,
no frozen control and no full-answer accuracy measurement. It does not establish broad
subject competence, generalization, causal attribution against controls, learning-to-learn
or improved architecture. The test split was not evaluated. Results by domain, exact
checkpoint hashes and step logs are in the adjacent `results/` files.

The continuing full checkpoint is `runs/curriculum-colab-chunked/result/student/step-5.pt`,
SHA-256 `1b712fd967ef31148925fa79bdfa092cf5eb64a6c3ad5f3e1ee9b492b9008e42`. It is a
research student and is not automatically substituted into an organism. The old parent and
all intermediate checkpoints remain immutable. Remote-only neural loading still applies.
The prior TLS EOF is consistent with a very large single request but its server-side cause
was not established. Chunked transfer succeeded on this attempt; it is not a guarantee for
all files, runtimes or networks.
