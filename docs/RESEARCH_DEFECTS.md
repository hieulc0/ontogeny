# Research defect register

| ID | Status | Evidence and implication |
|---|---|---|
| RD-001 | Confirmed objective limitation | Exhaustion makes undiscounted energy-change return identical (-40/15) across resource counts. Discounted return still varies; causal contribution unqualified. |
| RD-002 | Confirmed observability gap in records | Original logs omit per-step probabilities, entropy and advantages. Endpoint probes cannot reconstruct learning-time collapse. |
| RD-003 | Hypothesis | Sparse positive collection and long delayed credit may overwhelm the policy-gradient signal. Test immediate then delayed canaries. |
| RD-004 | Hypothesis | Seed 44 action bias/credit variance causes regression. Parameter changes alone do not explain utility; inspect recorded behavior and training-only endpoints. |
| RD-005 | Confirmed interface limitation | Policy sees one observation, not retained experience. Memory adequacy for navigation is untested; immediate resource detection needs no added history. |

No entry authorizes editing the original benchmark or reusing consumed sealed worlds.
