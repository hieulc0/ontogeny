# Matched-coverage acquisition and optimization

Preregistration: comparison-coverage-001. Freeze source before execution. The original
comparison-acquisition-001 protocol/results remain immutable; its runner is reproducible
at 07b07198877a632f37790c53757cfc1a5e840172. This follow-up uses the current runner.

Question: with every operand observed equally often in each training set, does partner
coverage or longer fixed optimization improve held-out integer ordering? No change to
the canonical model, objective, optimizer algorithm, or existing student/organism state.
All six research students are explicitly fresh, with one persistent LearningState each.

Cohort: integers32..63 (all operands and pairs disjoint from the previous0..31 cohort),
byte representation only, new initialization seeds404/505/606. ModelConfig: vocab256,
width64, layers2, heads4, FF256, context8, dropout0; other fields default. AdamW lr0.002,
weight decay0, clip1.0, no scheduler. LR stays constant throughout; duration alone changes.
The byte representation was chosen using the previous study, so this is an independent
follow-up of that choice, not an unbiased rediscovery of its benefit.

Shuffle the32 vertices with Random(771), connect each vertex to the next1 or4 vertices
on the cyclic ordering, and include both directions of every edge. The resulting nested
training sets contain64 or256 examples. Each operand appears exactly2 or8 times in each
input position. Uniform Random(444) sampling of16 distinct examples per update gives
matched expected per-operand exposure per update; log actual counts, which need not match
exactly. The larger graph supplies more partners, not previously unseen operands.
Development/test exclude ALL128 unordered edges of the larger graph. Shuffle the remaining
368 unordered edges with Random(772), take first184 for development and last184 for test;
include both orientations, giving368 balanced examples in each. No reversal leakage.
The smaller graph's unused larger-graph edges are not evaluation examples. The generated
split is public held-out evaluation, not a sealed external test. Fix dataset hashes before
submission using local non-neural generation; never inspect model outcomes before freeze.

Each state trains512 updates. Persist full checkpoints at0,128,512 with linked parent
hashes and sampler state. Step128 is a prespecified short-budget endpoint, not a selected
winner. Development/train scores at0,32,128,256,512 provide diagnostic curves only.
After all training for each state ends, evaluate BOTH frozen endpoints and initialization
on the same independent test. Full snapshots restore RNG/state around evaluations; no
post-test training, early stopping, parameter tuning or additional trial is admitted.
One-token answers use unrestricted256-way argmax, no label masking or answer prefix.

Primary gate: the256-example/512-update condition must reach >=90% exact test accuracy
for EACH seed and >=20 percentage points mean gain over its matched frozen initial model.
Report the same gates for the other three prespecified size/horizon conditions as secondary
results, without substituting a secondary winner for a failed primary gate. Report all
paired size and duration contrasts, raw numerators, individual seeds, mean and range.
Three seeds and correlated reversed pairs do not support precise population confidence.
The label-only constant baseline is50%, even if unconstrained frozen models score0%.
Parameter count and input sequence length are identical across conditions. Longer training
costs4x the updates: any improvement is not claimed to be free or more compute-efficient.
This task tests interpolation within a new operand range, not extrapolation to arbitrary
integers, new tasks, retention, self-selected practice, continual plasticity or growth.

Budgets:3072 physical optimizer updates,49152 training examples,294912 training tokens;
fixed finite evaluation calls <=200000 input tokens, worker600 seconds, Torch3 GiB,
export256 MiB, provider setup/execution1200 seconds. One validated Colab T4 allocation;
no retry/fallback. Refusal means NOT RUN; ambiguous worker/cleanup means INCONCLUSIVE
and blocks another allocation. All real neural execution and loading is remote CUDA only.
Pin Torch2.6.0+cu124/NumPy2.1.3 and process-start CUBLAS determinism. Verify committed
source and imports; keep tracebacks, per-update append-only logs, partial and final states.
Stop on missing ownership, nonfinite/zero gradients, unchanged first-step output weights,
nonfinite parameters, source mismatch or exhausted limits. No silent failed-arm omission.
Recover hash-verified complete outputs before releasing the owned session and claiming
completion. These checkpoints do not replace any continuing learner; schema2 is unchanged.
A pass qualifies only this acquisition cohort, not L.1 or structural development.
