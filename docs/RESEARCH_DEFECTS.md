# Research defect register

| ID | Status | Evidence and implication |
|---|---|---|
| RD-001 | Confirmed objective limitation | Exhaustion makes undiscounted energy-change return identical (-40/15) across resource counts. Discounted return still varies; causal contribution unqualified. |
| RD-002 | Confirmed observability gap in records | Original logs omit per-step probabilities, entropy and advantages. Endpoint probes cannot reconstruct learning-time collapse. |
| RD-003 | Hypothesis | Sparse positive collection and long delayed credit may overwhelm the policy-gradient signal. Test immediate then delayed canaries. |
| RD-004 | Hypothesis | Seed 44 action bias/credit variance causes regression. Parameter changes alone do not explain utility; inspect recorded behavior and training-only endpoints. |
| RD-005 | Confirmed interface limitation | Policy sees one observation, not retained experience. Memory adequacy for navigation is untested; immediate resource detection needs no added history. |

No entry authorizes editing the original benchmark or reusing consumed sealed worlds.

## Diagnostic execution evidence

RD-003 remains a causal hypothesis: immediate canary PASS and delayed sequence FAIL
localize a limitation before navigation, but task differences include binding complexity.
RD-004 now has confirmed behavioral evidence (seed 44 interaction suppression and
higher costly exposure); the neural cause is unresolved. Endpoint entropy .418 for
seed 43 identifies a strong interaction bias; seed 44 entropy 1.494 is not complete
single-action collapse. No optimizer sign/ownership defect was detected: controlled
positive advantage raises selected-action log probability and all intended groups update.

| ID | Status | Evidence and implication |
|---|---|---|
| RD-006 | Confirmed diagnostic design limitation | Delayed task differs in both reward timing and stage-dependent action mapping. Its FAIL cannot identify timing as the sole cause; use a matched future timing comparison. |
| RD-007 | Qualification gap | New canary checkpoints reload for evaluation, but fresh-process next-update equivalence on CueWorld was not executed. Historical CellWorld H3 must not be generalized automatically. |
| RD-008 | Provenance gap | Diagnostic core Torch/CUDA/GPU/source are captured; no separate full interpreter/pip inventory. No cross-runtime reproducibility claim. |

Immediate CANARY PASS, delayed FAIL, L.1 STILL FAIL. No CellWorld repair admitted,
no new sealed rollouts, no capacity increase or second cell. Exact evidence and
preregistered stop: LEARNING_DIAGNOSIS.md and experiments/learning_diagnosis/results.

## Execution qualification gaps (separate from learning diagnosis)

| ID | Status | Evidence and implication |
|---|---|---|
| EX-001 | Confirmed infrastructure failure; bounded recovery passed | Kaggle package setup exhausted a 300 s platform cap. Explicit 600 s sequential batch amortized setup and passed unchanged public worker cases; original failure retained. |
| EX-002 | Availability blocker | Final Colab allocation refused with no assignment. Kaggle→Colab continuation/evaluation NOT RUN; no bidirectional portability claim. |
| EX-003 | Qualification gap | Imported fixture checkpoint precedes the first update, so optimizer moments/gradients are empty. Exact post-update equality does not qualify migration of an already trained optimizer. |
| EX-004 | Qualification gap | SIGKILL after complete publication passes, but arbitrary VM loss, periodic durable external ACKs and unexported-tail accounting remain unqualified. Automatic migration stays disabled. |

These entries do not explain or reverse the L.1 behavioral failure. Execution evidence
and source/runtime limits: CHECKPOINT_QUALIFICATION.md.
