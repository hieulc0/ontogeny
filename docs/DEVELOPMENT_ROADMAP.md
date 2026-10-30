# Unified foundation priority

The current engineering consolidation is documented in MIGRATION.md. It extracts one
canonical mini_gpt engine for independent neural research and organism learning. No new
scientific stage or architecture growth is admitted by a package merge. After numerical/
state equivalence, the next research priority is reliable learning from finite feedback:
minimal concept acquisition and matched action/credit diagnostics under the shared engine.
Keep separate continuing individuals; future consolidation requires retained competence.
The detailed diagnostic proposal below remains proposed, not automatically executed.

# Capability-gated research and engineering roadmap

Status: DRAFT PLAN, not an experiment preregistration or implementation authorization.
Based on repository f4a7700e11f970ca984d9498edc675be9177e833 and the completed
[learning diagnosis](LEARNING_DIAGNOSIS.md). No training is launched by this plan.
Provider research checked 2026-10-30; execution details: [portable remote design](REMOTE_EXECUTION.md).

## Current evidence and immediate decision

| Question | Observed evidence | What remains unproved |
|---|---|---|
| Can neural feedback change intended parameters? | Positive-advantage/ownership test PASS | Whether changes improve a complex behavior |
| Can this student learn one observable association? | All three immediate canaries 100% held-out | General intelligence or broad concept transfer |
| Can it learn the tested sequence? | Delayed successes 15.2/15.6/18.9%, FAIL | Whether delay, stage binding or feedback conjunction causes failure |
| Can it navigate and collect reliably? | L.1 H1/H2 FAIL; heuristic much stronger | Robust independent behavioral learning |
| Is learning persistent? | Historical CellWorld H3 PASS | CueWorld fresh-process next update and cross-provider continuation |

The next scientific build should be a **matched credit-and-binding diagnostic**, not
another large CellWorld sweep. The old delayed test changed horizon, action mapping,
reward timing and success conjunction. It localizes failure before navigation but
cannot justify one particular repair. Endpoint action entropy and telescoping reward
are evidence to investigate, not proof that an actor-critic, larger network or memory
will fix the problem. Keep the actual Mini-GPT backbone/head and current capacity.

Two engineering prerequisites are small enough to tackle alongside experimental design:
complete telemetry/continuation, and a provider-neutral worker. Do not let building a
cloud scheduler replace scientific progress. Ship a minimal contract/Kaggle wrapper first;
Colab routing is a separate qualified increment, not a requirement to redesign the learner.

## Ordered near-term work

| Order | Deliverable | Gate before next work |
|---|---|---|
| A | Complete telemetry, CueWorld continuation, common job contract | Recorded losses reconstruct; complete trusted state resumes exactly on current runtime |
| B | Existing Kaggle behind common worker | Historical smoke reproduces; inputs/outputs validate; no learning change |
| C | Colab adapter and bounded fallback | Same worker and outputs on T4; verified export/cleanup; provider compatibility documented |
| D | Matched credit/binding diagnosis | Independent per-cell outcomes and factorial contrasts; no confounded causal claim |
| E | One evidence-supported repair on the small task | Beats unchanged control under matched work and unseen worlds |
| F | Frozen original CellWorld requalification on fresh seeds | Robust acquisition/generalization, viability and efficiency, full state continuity |
| G | Persistent adaptation and retention | L.1 first qualifies; then L.2 can be preregistered |

A and design of D may proceed before C. D can run on qualified Kaggle while Colab is
unavailable. No stage advances because code exists or a provider allocates successfully.
Each executable study needs a frozen separate preregistration before submission.

## A. Measurement and continuation before further hypotheses

Add a complete observation/action/feedback record without changing behavior. Capture
input encoding schema, action validity mask, logits/probabilities, sampled action, action
RNG identity, reward components, discounted return, baseline-before-update, advantage,
loss, named gradient summaries, optimizer counters and inference/update/sensory costs.
Hidden simulator truth stays in privileged evaluation records. Test state-dependent
sensing and legal-action masking from the declared interface, not private maps.

Capture full Python/Torch/CUDA/cuDNN/driver/package inventory and source/import hashes.
Persist the sampler cursor, episode cursor, owned world state, pending experience,
optimizer/scheduler, gradients, baselines and all used RNG streams. Checkpoints are
immutable with parent hashes and atomic export acknowledgement. Prove a new process
matches the next distribution/action, environment transition and actual update, including
a checkpoint in the middle of an unfinished episode. Compare complete state, not only
weights. This closes RD-007/RD-008 without changing a learning objective.

Engineering acceptance: malformed input, insufficient budget, wrong provenance and
missing outputs reject explicitly; optimizer membership correct; positive-advantage test
passes; telemetry reconstructs reward/advantage/loss within declared numerical tolerance;
evaluation leaves learning state unchanged. Passing these does not requalify CellWorld.
No reliance on runtime `assert` for safeguards. Keep old pickle class paths intact.

## D. Matched experiment: isolate credit from binding

### Factorial tasks

All four cells have the SAME three-step horizon, binary cue, two actions, available cue
and stage observations, encoding, model, optimizer, episode budget and final metrics.
Stage is always visible, even when irrelevant. No supervised action labels reach the
student; correctness is evaluated by environment consequences of its sampled actions.

| Mapping factor | Immediate feedback | Terminal feedback |
|---|---|---|
| Cue only | Beneficial action depends on cue at every step | Identical mapping, delayed feedback only |
| Cue + stage | Beneficial action alternates with observed stage | Identical alternating mapping, delayed feedback only |

Let c_t be +1/-1 observed correctness consequence. Dense task pays c_t at each step;
terminal task pays 0,0,sum(c_t). Thus every identical action sequence has identical
undiscounted total feedback, action effects and success metric. Use gamma=1 for ALL
four cells in this diagnostic cohort to avoid changing the optimized discounted objective
when moving reward in time. This common diagnostic choice is not a proposed CellWorld
repair or an assertion of equivalence to the old gamma=.97 run. Keep signed sequence
success (all three correct) as a secondary metric and sum return as the primary target.

This also removes the old all-or-nothing reward conjunction from the factorial study.
If all four cells learn but the old conjunction task still fails, conjunction/signal
structure becomes a separate candidate cause; do not call it a delay or binding finding.
Return to the unchanged old task on fresh diagnostic seeds only under a separate
preregistered confirmation, without tuning its original held-out outcomes.

Local environment tests must exhaustively verify all 2 cues x 8 action sequences for
all four cells: total reward equality for paired timing conditions, cue/stage observability,
valid actions, deterministic transitions, terminal logic and no hidden-label leakage.
These exhaustive simulator controls are never behavior supplied to the learner.

### Proposed cohort and boundaries

- Five initialization replicates 42..46, reported individually; same initial model state
  reused across the four matched cells within a replicate. Fresh lineages explicitly
  declared; each arm persists across its own episodes with no silent optimizer resets.
- 2,048 training episodes per cell/initialization, one update per episode, <=6,144
  interactions per arm. Random and frozen controls share evaluation seeds/horizon.
- Disjoint manifests: proposed generator namespaces 1100000.. for training, 2100000..
  development, 3100000.. final qualification, with explicit nonoverlap and generator hashes.
  Development 256 worlds, final 512 worlds per arm/cell/init. Freeze final checkpoint;
  no choosing a curve peak. Those ranges are proposals, not yet generated sealed data.
- Finite maximum 40,960 trained updates across 4 cells x 5 inits, plus declared audit
  overhead. Diagnostic GPU budget cap 8 allocated session-hours including setup and
  retries; stop/defer if measured estimates exceed it. This is a ceiling, not an estimate
  that the current small networks need that much compute.
- No hyperparameter search in this first cohort. If acquisition is inconclusive, preserve
  it and preregister a new training-only design with new final worlds, not more favorable seeds.

Preregister initial/final accuracy, return, success-area-under-learning-curve, entropy,
cue/stage conditional action preferences, hazard/energy as NOT_APPLICABLE in these tasks,
updates-to-threshold, resource/compute cost, and checkpoint exactness. Conditional action
probabilities matter more than global entropy: deterministic correct association is useful;
constant confident action choice can be failure.

### Analysis and decision rules

Estimate dense-minus-terminal effects within mapping, cue-only-minus-binding effects
within timing, and their interaction. Pair both initializations and world seeds. Use
hierarchical paired resampling of initialization and world blocks; report per-initialization
results and acknowledge only five independent trained replicates. Do not treat all
rollouts as independent model replications. Freeze the confidence method and multiple
primary contrasts before generating final results; report all contrasts, not just significant ones.

Proposed per-cell acquisition gate: >=80% full-sequence success for every initialization,
paired 95% improvement lower bound >.15 over both random/frozen, and positive late-minus-
early return improvement or documented early attainment above the fixed threshold.
The last clause must be precisely preregistered so very fast learning is not incorrectly
failed by a first-quarter ceiling. Success threshold, learning windows and confirmatory
comparisons are finalized BEFORE running, not adjusted to these proposed results.

| Outcome pattern | Permitted conclusion | Next candidate study |
|---|---|---|
| Dense cue-only fails | No reliable acquisition on this formulation | Representation/optimizer validation on this task; stop escalation |
| Dense works, terminal fails within same mapping | Reward timing contributes under this budget | One credit/advantage correction, unchanged model/task |
| Cue-only works, stage mapping fails in both timings | Binding/observation utilization contributes | One information-equivalent encoding/readout correction |
| Only terminal binding fails | Interaction between binding and feedback timing | One matched small-task intervention; no sole-cause claim |
| All work | Old failure not explained by these two factors alone | Isolate all-or-nothing feedback next; preserve old negative result |
| Mixed seeds / wide intervals | INCONCLUSIVE on causal attribution | Report uncertainty; no favorable-seed promotion |

This experiment tests conditional causal effects inside these simulators, not a universal
law of intelligence or an explanation of every historical parameter update.

## E. One repair, chosen after evidence

Do not implement a menu of PPO, replay, curriculum, reward shaping and memory together.
Select one category supported by D, preregister one comparison and keep unchanged controls.
Gradient-estimator variance and baselines are legitimate research targets; value-based
advantage estimators explicitly trade bias and variance, rather than guaranteeing success.
[Original GAE paper](https://arxiv.org/abs/1506.02438). This is motivation for a later
controlled study, not a decision to add GAE/a critic or increase capacity now.

Possible bounded branches:

- Timing effect: test one action-independent baseline/advantage correction. Specify its
  exact equation, state ownership and compute before execution; no simultaneous entropy,
  optimizer, reward or architecture change.
- Binding effect: test one observable-information-equivalent token layout/stage encoding
  correction within current context and parameter envelope; no oracle action tags.
- Original reward limitation after canary competence: test a single explicit feedback
  correction that distinguishes resource collection from telescoping net energy. Keep
  simulator dynamics and independent viability metrics unchanged. This is a new objective
  hypothesis, not a retrospective repair of a score.

Five initializations, matched initial checkpoints, fixed interactions/updates/sensory
budget and paired worlds. Record any additional gradient/query overhead instead of
calling unequal computations matched. Require repeated improvement beyond unchanged,
frozen and random, retained immediate-cue performance and reproducible continuation.
If an unavoidable head/parameter change is proposed, stop and seek a separate capacity-
matched architecture decision; it is outside this minimal no-growth repair cohort.

Failure/inconclusive keeps L.1 FAIL. A small-task PASS alone is CANARY PASS, L.1 STILL
FAIL. Do not automatically start navigation because the most favorable initialization works.

## F. Original CellWorld qualification, only after canary evidence

Freeze original dynamics and observations. Never reopen the consumed L.1 sealed cohort.
Use new world manifests, paired independent initializations, original/random/rest/heuristic/
frozen/repaired policies and the preregistered feasible hazard variation. Keep the unchanged
original learner as a controlled arm with identical training work. The fixed heuristic is
an external reference, never teacher actions or a fallback in the neural policy.

Evaluate acquisition on training-only curves, then untouched transfer: resources, duration,
energy trajectory, exhaustion versus horizon, hazard/action, collection/action, conditional
interaction on observed resource, return, entropy, observation/query/update costs and
retention of earlier qualified canaries. Control return/accounting exploits separately.

Require corrected-minus-unchanged and corrected-minus-random/frozen paired lower95 bounds
above zero for resources and resource efficiency on both environments; duration must not
regress and a preregistered viability floor must be met in every initialization. Report
rest/heuristic gaps explicitly. Fix exact effect-size/viability thresholds in the new
protocol before training; these draft gates are not an amendment of historical L.1.
No PASS based on duration alone, reward alone, gradient movement or provider success.

Fresh-process full-state next-update continuation is required on the actual trained
final checkpoint. Cross-provider migration is allowed only under REMOTE_EXECUTION's
qualified compatibility profile; otherwise keep each arm provider-affine. Resume is not
an opportunity to replace a struggling learner with a specialist or fresh optimizer.

A full new study can qualify a new L.1 candidate; historical FAILs remain unchanged.
If sealed results fail, preserve them and stop. Suggested stage envelope <=8 Kaggle
session-hours plus a separately declared final confirmation allowance, not unbounded
quota-driven searching. Final scientific cohort allocation is frozen before its outcomes.

## Long-term development after L.1

| Milestone | Capability question | Required controls and promotion evidence |
|---|---|---|
| L.0 | Reproducible meaningful consequences | Existing engineering PASS; maintain invariants |
| L.1 | Reliable learning through own actions | Independent acquisition/transfer and full learner continuation; currently FAIL |
| L.2 | One continuing learner adapts and retains | Actual sequential rule changes, competence-before-retention gates, history-matched reset/static/replay controls, delayed retention and adaptation cost |
| L.3 | Cooperation adds value | Two cells only after L.1/L.2; communication costs, capacity-matched single-cell baseline, no assigned-role emergence claim |
| L.4 | Organization can develop | Reversible allocation/growth/pruning under matched active/stored capacity and retention evidence |
| L.5 | Learning processes improve future learning | Experience selection/learning-strategy/topology proposals beat strongest fixed controls on new tasks at matched total resource cost |

At L.2, introduce context/history or episodic reuse only when tasks require it and
measure its causal contribution. At later stages, study active experience choice,
self-supervised world prediction and differentiated organization as separate hypotheses.
A prediction model and a behavior policy need not be added in the same experiment.
Architecture/memory/context have no permanent ceiling, but every growth/migration needs
finite budgets, immutable parents, independent qualification and rollback. No autonomous
source rewriting, deployment or real-world resource acquisition.

## Definition of done for the next implementation request

Deliver a frozen matched-task protocol, exhaustive environment tests, complete telemetry,
new-state continuation tests, common worker I/O contract and a first qualified Kaggle
adapter. Add Colab with explicit account authorization and real allocation/export tests;
only then enable automatic routing. Keep semantic refactoring separate from learning
changes and preserve old serialized class paths. Use behavior-based module names.

Before any GPU submission: publish exact executable SHAs, verify immutable inputs,
check available quota, reserve setup/export/retry costs and freeze stop conditions.
The plan itself did not implement or launch experiments. Subsequently the common
worker/Kaggle foundation was implemented and engineering-qualified; see
EXECUTION_QUALIFICATION.md for the exact limited scope. Colab/routing and the matched
scientific study remain proposed, with no automatic advancement.
