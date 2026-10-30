# Segment interruption verification

Preregistered engineering experiment, not a learning qualification. Continue the real
research student from `step-5.pt`, SHA-256
`1b712fd967ef31148925fa79bdfa092cf5eb64a6c3ad5f3e1ee9b492b9008e42`, and the starter
training batches SHA-256 `f2841088feefa520c1d545326585d158e61c31c990071937140357191c49708f`.
Create two independent branches from that immutable checkpoint. Each begins the same
20-update segment. One runs uninterrupted; the other stops after ten updates and resumes
from its full step-15 checkpoint in a fresh process to step 25. Compare the final model,
optimizer, scheduler, external sampler state, metadata, gradients, RNG, model training
mode and requires-grad flags exactly, and evaluate logits in eval mode on one fixed input.
The branches use identical source, device count, T4 runtime and explicit segment metadata.
Keep the branch parents and all generated checkpoints; no checkpoint is promoted to an
organism. Abort on any mismatch or nonfinite outcome. This tests a declared orderly stop
and fresh-process resume, not arbitrary VM loss or cross-provider portability.

Budget: 40 physical optimizer updates, no more than 5000 input bytes, 300 seconds worker
time, 3 GiB Torch allocation per process, at most 1 GiB indexed output; provider setup
and execution cap 900 seconds. Use one positively verified Colab T4 with chunked verified
input/output; if unavailable, report not run and preserve the attempt. Do not automatically
retry an ambiguous worker. All actual neural work remains remote-only. No model architecture
or learning algorithm change is admitted. Record exact source and results.
