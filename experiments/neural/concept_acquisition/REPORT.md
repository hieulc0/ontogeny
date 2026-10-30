# Concept acquisition: observed result

**Remote execution COMPLETE; preregistered acquisition qualification FAIL in all four
conditions.** This preserves the negative result; there was no tuning, checkpoint
selection, second allocation or growth intervention after seeing the test results.

Source `07b07198877a632f37790c53757cfc1a5e840172`, Colab native session
`ontogeny-373c0fd59f859857`, validated Tesla T4, Torch 2.6.0+cu124 and NumPy 2.1.3.
All 12 fresh independent students completed 128 updates. Their immutable initial parents
and final full-state checkpoints were recovered with file hashes verified; the owned
session was stopped and cleanup confirmed. No persistent student/organism was replaced.
Checkpoint schema 2 and canonical neural algorithms are unchanged. These smaller research
students are distinct histories, not a continuation of the existing 10M student.

| Representation | Training examples | Test accuracy, seeds 101 / 202 / 303 | Mean | Gate |
|---|---:|---|---:|---|
| bytes | 32 | 75.00% / 71.63% / 79.81% | 75.48% | FAIL |
| bytes | 128 | 90.38% / 85.58% / 87.98% | 87.98% | FAIL |
| symbols | 32 | 63.46% / 62.98% / 60.58% | 62.34% | FAIL |
| symbols | 128 | 76.44% / 75.00% / 71.63% | 74.36% | FAIL |

Each test has 208 held-out balanced examples. The gate required >=90% for every seed,
plus >=20 percentage points mean improvement over frozen initialization. All frozen
initializations scored 0% under unrestricted 256-token argmax; a label-only constant
predictor would score 50%. Therefore the large gain from the frozen model must not be
interpreted as equally large concept competence. Every trained run exceeded 50%, but only
one run exceeded 90%. Reported learning is narrow supervised acquisition, not self-directed
experience selection, retention, plasticity maintenance, transfer or evolution.

Training accuracy ranged 92.97–100%. More data improved final test performance in every
paired seed/representation. Byte prompts outperformed symbolic tokens in every paired
seed/data condition. Development performance from 32 to 128 updates was not uniformly
monotonic. This supports further investigation of coverage, representation and
optimization; it does not isolate one cause, establish capacity insufficiency, or prove
that any permanent architecture constraint prevents learning. The smaller set covers 23
of 32 operands; the larger covers all 32, confounding pair coverage with symbol coverage.
Byte inputs use 6 tokens versus 3 for symbolic inputs, so their compute differs. All arms
have the same model capacity and number of gradient updates.

Cost: 1536 optimizer updates, 24576 training examples, 110592 training input tokens and 61236
evaluation input tokens. Worker wall 25.415 seconds, peak Torch allocation 75,569,664 bytes
(72.1 MiB), archive 33,680,084 bytes. These exclude Colab allocation, framework installation
and transfer time; billed GPU duration/cost was not returned. One allocation only, no Kaggle
quota spent. The provider setup/transport dominated the elapsed attempt.

LOCAL: 15 explicitly inspected non-neural tests passed, Ruff lint/format and Pyright passed,
and generated bootstrap syntax was checked. No local model construction, forward/backward
or checkpoint loading occurred. REMOTE: nonzero finite gradients, optimizer membership,
first-update output-weight changes and complete cohort execution passed. This run does
not add a fresh-process next-update checkpoint-continuation qualification.

Raw metrics, all update losses/norms and checkpoint identities are in
[results/result.json](results/result.json); the complete export hash inventory is in
[results/manifest.json](results/manifest.json). Binary checkpoints and per-arm records
remain under ignored `runs/concept-acquisition/result/`. The sanitized evidence directory
is not itself the full checkpoint bundle. Private provider logs remain in ignored runs.

The public held-out comparison split is now consumed. Do not tune on it and report a
second score as independent confirmation. Next scientific work needs a new frozen cohort,
matched coverage controls and an optimization schedule specified before observing its
independent test. Sequential-retention and structural-development gates remain closed.

A subsequent launcher hardening preserves UNKNOWN on ambiguous allocation/cleanup
errors instead of inferring cleanup from absent local assignment data. Its mock regression
passes; it was not an additional GPU attempt or a change to the measured worker.
