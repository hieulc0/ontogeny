# Development roadmap

## Current evidence and engineering gate

One canonical mini_gpt engine now supports independent neural training and organism
learning. Migration verification is recorded in MIGRATION.md; source consolidation
cannot promote a scientific stage. The initial 10,054,016-parameter student completed
a two-update engineering pilot with exact checkpoint continuation. Its next engineering
step is explicit new training segments carrying optimizer/state across subjects; see
experiments/neural/initial_student/REPORT.md. The previous integration/provider records remain
valid only at their original source/runtime revisions.

L.0 environment engineering passed. L.1 acquisition/held-out behavior failed; continuity
passed. Immediate cue learning passed, while the delayed canary failed. A.13 binding
failed; B.0 established a continuing state without qualifying initial competence,
positive forward transfer or benefit from a learned practice policy. These negative
results guide the next experiments; no checkpoint or organism is promoted by a refactor.

## Next research: reliable learning from finite feedback

After the engineering gates pass, prioritize small acquisition experiments rather than
more infrastructure, parameters, cells or environmental complexity. Independent students
and the continuing organism share code but own separate state/history. Do not substitute
an independently trained specialist for the organism.

The first embodied diagnostic should isolate timing from stage-dependent binding. The
old delayed canary changed both, so its failure cannot establish delayed credit as the
sole cause. Preregister four matched tasks:

| Observable action mapping | Immediate consequences | Terminal consequences |
|---|---|---|
| Cue only | Same cue-dependent action at all three steps | Same mapping and horizon; feedback paid at the end |
| Cue + stage | Beneficial action alternates with visible stage | Same alternating mapping; feedback paid at the end |

Keep cue/stage observations, action effects, model/head, initialization, optimizer,
interactions and update budgets identical. If c_t is the observed +/-1 consequence,
dense feedback pays each c_t, while terminal feedback pays 0,0,sum(c_t). Use gamma=1 in
all four diagnostic cells so matched action sequences have equal total feedback; this
is a declared diagnostic choice, not a CellWorld repair. Correct action labels never
enter policy gradients. Random/frozen policies and unchanged learner are controls.

Before neural work, exhaustively verify all 2 cues × 8 action sequences: reward equality
across timing pairs, deterministic transitions, partial-observation contract, terminal
logic and no hidden labels. Use multiple model initializations, paired disjoint training/
evaluation worlds, fixed final checkpoints, complete per-step telemetry, uncertainty
and per-initialization failures. Freeze finite budgets/effect-size gates before training;
a bounded engineering canary does not authorize scaling a failed acquisition cohort.

In parallel, an independent neural study may test tiny concept acquisition from verified
finite feedback using the canonical hidden_states and one declared readout. Keep input
representation/readout, optimization and curriculum interventions separate. Compare
unchanged/frozen and strong simple learnability controls; controls do not become deployed
neural assistance. Require training acquisition before interpreting retention, and unseen
examples before claiming representation reuse. Historical concept checkpoints that were
not migrated are not silently assumed to initialize this student.

The larger pre-consolidation diagnostic proposal remains available in Ontogeny Git at
670709ca2d319baf3073be0f7ce709f3f8d664d1. It was a proposal, not a completed study or a
binding permission to spend its full suggested budget. New execution needs its own
current-code preregistration and resource admission.

## One correction, then original-world qualification

Only a confirmed defect or discriminating diagnostic supports one minimal intervention.
Do not bundle reward shaping, representation, optimizer, curriculum and exploration
changes. Match total observation/update/validation costs against the unchanged learner.
Use fresh independent evaluation boundaries; consumed sealed worlds stay closed.

Return to frozen CellWorld with random, rest, fixed heuristic, frozen neural and learning
controls only after the small task qualifies. Measure resource collection/efficiency,
energy/hazards, return, termination, entropy, adaptation and compute. Require robust
improvement over frozen/random across initializations and unseen worlds, with explicit
viability/effect-size floors. Preserve the heuristic gap. Duration caps, parameter
updates and rising training return alone cannot qualify L.1.

Every actual trained final student needs full-state fresh-process next-update evidence.
Explicit dense format conversion is not arbitrary topology or cross-provider migration.
Missing reverse-provider, long-rollout and durable off-VM-ACK recovery gates remain
separate engineering work; they need not postpone a bounded study on qualified Kaggle.

## Longer-term capability gates

| Milestone | Research question | Admission evidence |
|---|---|---|
| L.0 | Meaningful reproducible consequences | Maintain seeded dynamics, resource and observation invariants |
| L.1 | Useful behavior acquired through own actions | Independent acquisition/transfer and full state continuity; currently FAIL |
| L.2 | One learner adapts and retains | L.1 first; sequential rule changes, competence-before-retention, history-matched controls |
| L.3 | Cooperation adds value | L.1/L.2 first; communication cost and capacity-matched single-cell control; no assigned-role emergence claim |
| L.4 | Useful organization develops | Reversible memory/module/growth/pruning proposals, matched active/stored resources and retained competence |
| L.5 | Learning processes improve future learning | Experience/strategy/topology choices beat strongest fixed controls on unfamiliar tasks at matched total cost |

Controlled consolidation is its own research mode. Test compatible continuation,
rehearsal, distillation or another explicit method before transferring independent
improvements into an organism. Preserve immutable parents and recovery paths.

Architecture, context, memory, connectivity and capacity have no permanent ceilings.
Every experiment remains finite, independently evaluated, interruptible and reversible.
No automatic milestone advancement, source rewriting, self-deployment or unrestricted
real-world resource acquisition is part of this roadmap.
