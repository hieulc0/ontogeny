# Fixed-capacity concept acquisition diagnosis

Preregistration: comparison-acquisition-001. Status: prepared; no outcomes observed.
This is a fresh research cohort, not a replacement for the continuing step-5 student
or organism. No topology, checkpoint schema, optimizer algorithm or live learner changes.

Hypothesis: a small canonical Mini-GPT can acquire integer ordering on unseen operand
combinations. This tests narrow supervised acquisition, not autonomous evolution,
continual plasticity, general arithmetic, or learning to learn.

Freeze the committed worker and data generator before submission; record its full Git SHA
in the launch journal and every checkpoint. No checkpoint inputs: each seed is a declared
fresh lineage. Run 3 seeds (101,202,303), 2 representations (two-digit byte prompts or
one symbolic token per integer), and 2 nested training sets (16 or 64 unordered pairs).
Both directions of every pair remain in the same split, exactly balancing labels. SHA-256
of `comparison:a:b` modulo 5 assigns unordered pairs over integers 0..31 to training
(0,1,2), development (3), or independent final test (4). No equal operands. No previously
consumed world or rule-transfer confirmation data is used. All data are synthetic/public;
the final split is held out from updates, not a cryptographically sealed external benchmark.

Model: vocabulary256, width64, depth2, heads4, FF256, context8, absolute positions,
dropout0; default remaining ModelConfig fields. Architecture is identical in every arm.
AdamW lr0.002, weight decay0, no scheduler, clipping1.0. Every update samples16 examples
without replacement using persistent Random(444), for128 updates per state. Same initial
seed yields matching initial weights across arms; comparisons are paired by seed. The
32-update development measurement tests shorter optimization without selecting it as a
final checkpoint. Record raw loss and gradient norm on every step. Only final-position
labels are supervised; predictions use unrestricted 256-way argmax, no supplied answer
prefix, masking to valid labels, or teacher forcing. Wrong/nonlabel outputs count wrong.
The simple target needs exactly one output token, so answer accuracy is exact accuracy.

Primary acquisition gate, evaluated independently for each prespecified arm: final-test
accuracy >=90% for EACH of the three initializations and mean improvement >=20 percentage
points over its corresponding frozen initial model. Report all four arms and failures;
no best-seed selection or claim of a general method from the best arm. Descriptive paired
comparisons: 64 versus16 pairs at128 updates diagnoses coverage; step32 versus128 on
unchanged development examples diagnoses optimization; symbols versus bytes diagnoses
representation jointly with sequence length/compute. These contrasts do not uniquely
identify causal mechanisms or prove insufficient capacity. Report individual scores and
min/max across seeds; three seeds do not establish precise population uncertainty.

First-update parameter-change audit, finite/nonzero gradients and optimizer membership
are engineering stop gates. Train accuracy distinguishes failure to fit from failure to
generalize. No candidate selection or parameter tuning uses final-test results. A follow-up
requires a new protocol and untouched combinations/ranges. Fixed final checkpoint step128
is saved before test evaluation. Preserve immutable initial parents, full final learning
states, sampler RNG, source, logs and checkpoint hashes; eval has dropout disabled and
performs no updates. No existing checkpoint class or schema is changed.

Maximum1536 physical updates,24576 training examples,147456 training tokens; evaluation
has a fixed finite set at steps0,32,128 and final frozen/test calls. Worker600 seconds,
Torch allocation3 GiB, export256 MiB, provider setup/execution1200 seconds. One validated
Colab T4 assignment; no fallback/retry under this protocol. Refusal or uncertain execution
is NOT RUN/INCONCLUSIVE with raw failure and cleanup state. No local neural work.
Commit/push the exact source before allocation; use pinned Torch2.6.0+cu124/NumPy2.1.3,
verified imports, CUDA admission and deterministic process-start environment. Recover
and hash-verify outputs before claiming completion. Stop on budget, nonfinite state,
ownership/source mismatch; retain partial records and traceback. Acquisition failure
leaves sequential-retention, plasticity, structural adaptation and L.2 gates closed.
