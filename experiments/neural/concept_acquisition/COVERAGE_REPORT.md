# Matched-coverage acquisition: observed result

**Primary acquisition gate PASS; remote execution COMPLETE.** This is a narrow
qualification of supervised integer ordering on the declared cohort. It is not evidence
of retention, autonomous practice selection, learning to learn, plasticity maintenance,
structural development or L.1 improvement. The previous experiment's failures stand.

Source `cb013147fc455175bb0a15bb4cf4187ffede72c8`, Colab native session
`ontogeny-4953351498113018`, validated Tesla T4, Torch 2.6.0+cu124, NumPy 2.1.3.
One allocation, no retry/fallback, all outputs recovered and hash-verified, owned session
cleanup confirmed. Six fresh research states completed 512 updates each; the original
10M student and organism were not modified. Canonical neural algorithms and checkpoint
schema 2 are unchanged; no conversion or independent-weight promotion occurred.

| Training examples | Updates | Test accuracy, seeds 404 / 505 / 606 | Mean | Gate |
|---|---:|---|---:|---|
| 64 | 128 | 91.03% / 89.40% / 86.96% | 89.13% | FAIL |
| 64 | 512 | 91.30% / 91.85% / 84.24% | 89.13% | FAIL |
| 256 | 128 | 88.04% / 91.85% / 85.87% | 88.59% | FAIL |
| 256 | 512 | 94.02% / 95.65% / 94.02% | 94.57% | PASS |

Each endpoint was fixed before training, saved before testing, and scored on 368 balanced
held-out examples. The primary condition was 256 examples/512 updates; all three scores
exceeded 90%, with mean 94.57%. Frozen initialization was 0% under unrestricted 256-way
argmax; the label-only constant baseline is 50%. The >=20 percentage-point gain criterion
also passed, but the meaningful competence evidence is the absolute held-out accuracy.
The three other conditions failed their across-seed gates and remain in the report.

Both training sets contain every integer 32..63 in both positions, exactly 2 or 8 times per
operand. They share the same held-out pairs. The full datasets have equal per-operand
frequency within each set; actual sampled total operand exposures after 512 updates ranged
473–573 for the smaller set and 472–556 for the larger. These counts are identical across
model seeds because sampling is independently fixed. Thus exposure is matched in expectation,
not bit-for-bit between training sizes. Representation, sequence length, model capacity,
learning rate and per-update batch size are identical between conditions.

At 512 updates, larger partner coverage improves all three paired test scores, with mean
advantage 5.43 percentage points. For the larger set, extending 128 to 512 updates improves
all three seeds (mean +5.98 points). For the smaller set, longer training has mixed effects
and exactly zero mean change; one seed declines. The larger set's 128-update mean is
slightly lower than the smaller set's, so additional examples alone were not sufficient
at the short budget. Final training accuracy is 100% for the smaller set and 99.61–100%
for the larger. This supports the combination of more partners and sufficient optimization
in this cohort; it does not prove one universal learning mechanism or a capacity limit.

No architecture change was necessary to pass this task. The fourfold update budget is a
real cost, not a claim of improved compute efficiency. Different operand ranges and seeds
prevent direct causal attribution of the change from the previous study. The public split
is now consumed, and three initializations with correlated reversed pairs do not justify
precise population confidence or broad arithmetic/general intelligence claims. This is
held-out combination interpolation, not extrapolation to arbitrary numbers or new tasks.

Resource use: 3072 updates, 49152 training examples, 294912 training input tokens and 134784
evaluation input tokens. Worker wall 58.586 seconds; peak Torch 79,804,416 bytes (76.1 MiB).
Recovered archive 30,239,920 bytes. Setup, allocation and transfer are outside worker time;
provider billing/total GPU duration was not returned. No Kaggle quota was used.

LOCAL: 20 inspected non-neural tests passed, Ruff lint/format and Pyright passed, generated
bootstrap parsed successfully. Data hashes, exact balance/coverage, reversal and old-cohort
disjointness were checked. All 3072 finite update records and 18 checkpoint hashes were
independently rechecked after collection without locally loading models. REMOTE: CUDA,
source/import/data identity, gradient/ownership/first-update parameter checks, bounded
training, finite parameters and same-source full-checkpoint loading/evaluation passed.
The 0/128/512 states retain sampler history and parent hashes. This experiment does not
claim a fresh-process next-update continuation equivalence test.

Raw metrics and checkpoint identities: [result.json](coverage-results/result.json).
All updates: [updates.jsonl](coverage-results/updates.jsonl).
Gate summary: [summary.json](coverage-results/summary.json).
Complete export inventory: [manifest.json](coverage-results/manifest.json).
Binary checkpoints and per-student files remain in ignored
`runs/comparison-coverage/result/`; the committed evidence directory is not the full bundle.
Private provider logs are retained separately under ignored runs.

The next scientific question is sequential acquisition and retention under one persistent
state, with frozen controls and matched replay costs. That requires a separate protocol,
new tasks/evaluation data and absolute competence gates. This pass admits designing that
study; it does not activate structural changes, additional cells or autonomous evolution.
