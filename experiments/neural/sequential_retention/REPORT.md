# Sequential ordering retention: observed result

**Remote execution COMPLETE; replay retention/acquisition qualification FAIL; replay-benefit
gate FAIL.** All three parents passed the development admission gate, so all nine branches
ran. No failed seed was discarded, no additional allocation occurred, and no thresholds or
checkpoints were selected after observing outcomes. This was one contextual-reversal study,
not a demonstration of long-horizon continual learning or autonomous self-improvement.

Source `b9d1cf99518d37853abfa38638831698f620caf2`; validated Colab Tesla T4 session
`ontogeny-0f3bebb8d1c76307`; Torch 2.6.0+cu124 and NumPy 2.1.3. The original attempt
completed and its owned session cleanup was confirmed before the subsequent conversation
continuation. The local process handle had expired; the completed journal and recovered
artifacts were used, with no new allocation, evaluation or training. All file hashes were
rechecked after continuation, along with every finite update log and its step sequence.

A is less-than and B is greater-than, distinguished by a visible operator byte. Both use
the same pair distribution on integers 64..95. Each held-out task has 368 balanced examples,
disjoint from training including reversed pairs, and disjoint from previous operand ranges.
This specifically tests conditional relation reversal and interference; the tasks are
complements, not independent families of knowledge. A below-chance score after B can reflect
systematic reversal rather than random loss of all numerical information.

A-parent held-out accuracy was 92.66%, 95.92%, 92.93% for seeds 707/808/909; B at these frozen
parents was 6.52%, 4.08%, 7.61%. Thus initial A competence was strong, unlike a retention claim
based on protecting a weak baseline. A development admission scores were 91.58%, 92.93%, 92.39%.

| Arm | Final A, seeds 707 / 808 / 909 | Final B, seeds 707 / 808 / 909 | Mean A | Mean B | Qualification |
|---|---|---|---:|---:|---|
| ordinary | 4.35% / 5.71% / 25.54% | 95.65% / 97.01% / 89.40% | 11.87% | 94.02% | FAIL |
| replay | 87.50% / 88.59% / 91.30% | 85.05% / 85.60% / 86.68% | 89.13% | 85.78% | FAIL |
| matched_new | 5.43% / 8.70% / 25.00% | 94.02% / 94.57% / 75.00% | 13.04% | 87.86% | FAIL |

Replay raises mean retained A accuracy by 77.26 percentage points versus ordinary training
and 76.09 points versus the matched-new-feedback control. However, it loses 5.16, 7.34, 1.63
points of A competence across the seeds: two violate the five-point drop limit, two finish
below 90% on A, and all finish below 90% on B. Its mean B score is 8.24 points below ordinary,
violating the benefit gate's five-point protected-objective tolerance. The separate replay
benefit gate therefore fails despite substantial descriptive retention improvement.

The ordinary and matched_new arms also fail joint acquisition/retention. Ordinary B
accuracy itself falls below 90% for seed 909. The matched_new control sees exactly the same
4096 distinct-within-batch B slots as replay, duplicated to 8192 processed B presentations;
replay processes 4096 B plus 4096 remembered A presentations. Ordinary processes 8192 B slots.
These are presentation counts, not counts of unique facts across the entire study. All
arms take 512 additional optimizer steps, with equal total processed tokens. Replay is an
engineered practice schedule, not a learned decision about what experience to seek.

Every parent recorded 256 observed A examples in owned replay memory. All branches retain
that same memory for audit, including the non-rehearsing controls. Therefore this study
does not compare storage-free training against replay storage costs. Separate replay-memory
bytes were not instrumented; peak Torch allocation and full checkpoint sizes were recovered.
This limits the cost accounting and is not evidence that replay storage is free.

Each branch started from its seed's same complete A checkpoint at step 512 with nonempty
optimizer state and continued to step 1024. Full initial, A-parent and final snapshots
preserve sampler/memory, optimizer, gradients, RNGs and parent hashes. Same-code full-state
loading/evaluation ran remotely. No fresh-process next-update equivalence was added, no
checkpoint schema conversion occurred, and neither the existing 10M student nor the organism
was replaced. Canonical neural model and optimizer algorithms were unchanged.

Resource use: 6144 physical updates, 98304 processed training examples, 589824 training input
tokens, 178848 evaluation input tokens. Worker wall 117.345 seconds, peak Torch 79,802,368
bytes (76.1 MiB), recovered archive 28,958,088 bytes. Setup/allocation/transfer are outside
worker time; provider billing and total GPU duration were not returned. One Colab allocation,
no Kaggle job. Fifteen complete checkpoints and raw logs were recovered and hash-verified.

LOCAL: 22 inspected non-neural tests, Ruff lint/format, Pyright and generated bootstrap
syntax passed. After recovery, all 6144 update losses/norms were finite and step sequences
matched 1..512 for A parents and 513..1024 for continuing arms; all artifact hashes matched.
No model was constructed, run or loaded locally. REMOTE: source/import/data identity,
CUDA/runtime admission, finite training/parameters, optimizer ownership, branch-step and
nonempty-moment checks passed. Complete scientific execution does not make its gates pass.

[Raw results](results/result.json), [all updates](results/updates.jsonl),
[gate summary](results/summary.json), [complete export inventory](results/manifest.json).
Binary checkpoints remain under ignored `runs/sequential-retention/result/`; private provider
logs remain outside committed evidence. The evidence directory is not the full binary bundle.

The test cohort is consumed. The next justified study would separate conditional task-cue
binding from interference under a new preregistration: joint-training controls can test
whether this fixed representation can learn both relations together before attributing
failure to a sequential-learning mechanism. Any budget or schedule comparison must use
fresh held-out inputs and preserve this failure. No third task, growth, structural controller,
L.2 or plasticity qualification is admitted by the present result.
