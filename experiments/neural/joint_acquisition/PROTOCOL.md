# Joint acquisition versus temporal interference

Preregistration ordering-joint-001. Freeze committed source and data hashes before one
remote attempt. Three fresh seed parents1010/1111/1212; each branches into three independent
continuing LearningStates with identical initial model/optimizer/RNG state. No input learned
checkpoint, architecture change, organism promotion or reset at a task boundary.

Tasks A/B: visible `<`/`>` relations on integers100..131. New operands and pairs exclude all
previous cohorts. Generate degree8 regular graphs on indices0..31 with shuffle1223, then
add100 to each operand. Both tasks share256 train,368 dev,368 test examples, with reversed
pairs kept together and query pairs excluded from training. Hash serialized token/target
rows and feedback streams in data.json. Three-digit byte prompts have8 tokens rather than
prior six-token prompts; input length is matched within this study but confounds cross-study
comparisons. Task families are still complementary ordering, not independent domains.

Model unchanged: vocab256,width64,layers2,heads4,FF256,context8,dropout0; remaining
ModelConfig defaults. AdamW lr0.002, decay0, clipping1.0, no scheduler. Batch16 and exactly
1024 updates per arm. No development-based stop or checkpoint selection. Both streams
are generated from training-only examples:512 samples of16 records without replacement
using Random(444) for A and Random(555) for B. Every arm consumes EXACTLY the same ordered
8192 examples per task, not merely equal expected exposure. Store streams and cursors in
LearningState.external. Inputs/feedback availability is controlled by the experimental
driver; no autonomous experience-selection claim is made.

Arms: joint consumes8 A +8 B every update; alternating consumes16 A on odd updates and16 B
on even updates; sequential consumes16 A for512 updates then16 B for512. All process16384
examples,131072 input tokens and1024 gradients. They differ in temporal order and gradient
batch composition; joint training is not isolated from the effect of minibatch mixing.
Random initialization, per-task record order, total exposure, capacity and compute-operation
counts are matched. Wall time/memory are measured rather than assumed equal. No replay
arm is included: this diagnoses simultaneous capacity and scheduling, not a replay revision.

Save complete initial, step512 and step1024 states with immutable hashes and linked parents.
Development scores for both tasks at steps0/512/1024, training scores at1024 only. Freeze
all final states before scoring test data. Evaluate final states, shared frozen initialization,
and sequential step512 (to establish prior A competence) on both held-out tasks. All evaluations
snapshot/restore full state and never update. Unrestricted256-token argmax; incorrect or
nonlabel outputs count wrong. All checkpoints/curves reported regardless of performance.

Primary joint-acquisition gate: BOTH tasks >=90% on EVERY seed at step1024, and mean gain
>=20 percentage points per task versus matched frozen initialization. Alternating/sequential
gates reported descriptively; their success cannot substitute for a failed primary joint
condition. Separate temporal-interference gate requires primary PASS; sequential A at512
>=90% on every seed; sequential A drop>5 points for at least two seeds; mean joint final A
exceeds mean sequential final A by>=5 points; joint mean B at most5 points below sequential B.
If joint fails, this does not establish a capacity ceiling: input representation, task binding
or optimization may still limit performance. Report final training fit and all seed outcomes.
Three seeds and dependent reversed examples do not support precise population confidence.
A label-only constant baseline is50%, even if unrestricted random predictions score0%.

Limits:9216 optimizer updates,147456 processed examples,1179648 training input tokens;
<=350000 evaluation input tokens; worker600 seconds; Torch3 GiB; export256 MiB;
provider setup/execution1200 seconds. One validated Colab T4, no retry/fallback. Refusal
means NOT RUN; uncertain execution/cleanup means INCONCLUSIVE and blocks allocation.
Pin Torch2.6.0+cu124/NumPy2.1.3 and deterministic startup. Commit/push exact source;
verify source/import/data/checkpoint identity, finite updates/weights and optimizer binding.
Append per-update logs, save fixed endpoints, preserve partial records/tracebacks on failure.
Recover all artifact hashes and confirm cleanup before claiming completion. Full schema2
states remain compatible with the same code; no new next-update continuation qualification.
Prior failures remain immutable. No growth, third task, L.2, learned controller or revised
learning schedule after observing this test is authorized by this protocol.
