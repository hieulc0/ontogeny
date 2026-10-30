# Sequential ordering acquisition and retention

Preregistered ordering-retention-001; no outcomes observed. Source is the committed SHA
recorded before allocation. No input checkpoints: three fresh research students, seeds
707/808/909, learn A before their complete states branch into paired continuing arms.
No branch substitutes independent weights into the organism or the previous research student.

Task A: two-digit `aa<bb=`; task B: `aa>bb=`. One byte0/1 target, unrestricted256-way
argmax. The operator is visible; this is a narrow contextual reversal/interference task,
not contradictory labels for an identical observation or an independent concept family.
Integer range64..95 is untouched by previous cohorts. Both tasks share underlying pairs
so differences arise from relation cues, not operand distributions. Their train/dev/test
pairs are identical and disjoint (including reversals) across splits. Regular graph degree8,
32 vertices shuffled with Random(991):256 train examples,368 dev,368 test per task.
Fix serialized token/target SHA-256 in data.json before execution. Public held-out splits
are not a sealed external benchmark. No old consumed test or checkpoint is used.

Model/optimizer unchanged: vocab256,width64,layers2,heads4,FF256,context8,dropout0;
remaining ModelConfig defaults. AdamW lr0.002, decay0, clipping1.0, no scheduler.
Train A512 updates, batch16, Random(444). Store only observed A token/target records
in a bounded256-example replay dictionary owned by LearningState. B RNG Random(555)
is also declared at initialization; no optimizer/RNG reset at the A→B transition.
Keep complete initial and A-final checkpoints, metadata, parent hashes and actual memory.

Before any B training, ALL three A-final development accuracies must reach90%. If any
fails, stop the sequential cohort without selecting favorable seeds; export all A states
and evaluate A-final tests once for diagnosis. Report sequential work NOT RUN due to
failed prerequisite. Test outcomes never determine whether a branch is run.

If admitted, load each same-code A-final full checkpoint into each of three distinct arms:
ordinary:16 B examples/update; replay:8 B +8 observed A examples/update;
matched_new:the same8 B examples duplicated to16. Every arm takes512 additional updates,
16 processed examples/update and equal input-token budgets. Sample16 B records using
persisted B RNG; replay and matched_new use the first8, so they see identical B records.
Replay samples8 from its owned observed memory using the continuing A RNG. Arms share
initial state/history, never mutable neural state. Frozen A-final is a no-update control.
All arms draw the same B proposal pool; unused proposals supply no gradient. Record both
processed B presentations and unique B slots before within-batch duplication. Ordinary
gets twice as many B slots as replay; matched_new controls that acquisition-exposure cost.
Replay is engineered rehearsal, not learned autonomous practice. Memory bytes and use
count as costs even though all branches retain the same frozen memory for audit purposes.

Save complete endpoints at A-step512 and final total-step1024. Development curves for
both tasks at B offsets0,128,512 are descriptive only. Freeze every final endpoint before
its independent test evaluation; eval snapshots/restores full state and does no learning.
Never select checkpoints or change the512-update budgets from curves or final results.

Primary qualification for replay requires for EACH seed: A-before test>=90%; final A
and B test>=90%; A drop<=5 percentage points. Additionally mean B gain versus frozen
A-parent>=20 points. Report every seed and all ordinary/matched_new failures. Separately
qualify replay benefit only if primary qualification passes AND replay mean retained-A
accuracy exceeds BOTH training controls by>=5 points while mean B accuracy is no more
than5 points worse than either. These are descriptive small-cohort gates, not statistical
proof; three seeds and paired reverse examples cannot establish precise uncertainty.
Never call preservation of a weak baseline retention. Constant label baseline50% matters
when unconstrained random models score0%. Two tasks do not demonstrate long-horizon
plasticity, positive forward transfer, independent-family generalization or evolution.

Limits:6144 physical updates (1536 A +4608 branch),98304 processed training examples,
589824 training tokens; <=250000 evaluation tokens; worker600 seconds, Torch3 GiB,
output256 MiB, provider setup/execution1200 seconds. One Colab T4 assignment, no retry
or fallback; refuse local execution. Pin Torch2.6.0+cu124/NumPy2.1.3 and deterministic
process environment. Verify source/import/data/checkpoint identities, finite gradients/
parameters, optimizer ownership, nonempty moments and unchanged step512 at branch start.
Stop on integrity/budget failure and preserve append-only logs, partial endpoints and raw
tracebacks. Recover/hash-verify all artifacts and confirm owned-session cleanup. Report
NOT RUN/FAIL/INCONCLUSIVE honestly. Existing schema2 and model classes remain unchanged;
no next-update exactness, organism upgrade, L.2, growth or third task is admitted here.
