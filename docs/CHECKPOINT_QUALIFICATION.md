# Checkpoint transport and recovery qualification

Status: preregistered, remote NOT RUN. Engineering only; L.1 remains Learning FAIL.
This extends the common-worker and Colab routing records without rewriting them.

## Questions and admission

Can declared trusted checkpoints be uploaded, resumed and evaluated through the public
worker on both providers? Does the next sampled action, transition and gradient update
match an uninterrupted branch, in both provider directions? Can a checkpoint published
before an abrupt process loss be recovered without resetting the student?

Keep the actual MiniGPT/PolicyCell/AdamW objective unchanged. Pin upstream
8567836fb2c8c915d442b7c2fbedb4fd9b30396d and the published executable Ontogeny SHA in every
request. Use Torch 2.6.0+cu124, NumPy 2.1.3, deterministic CUDA, Tesla T4, exactly one
visible device. Capture actual Python/driver/package versions; a matching nominal
profile alone does not prove numerical portability. Only explicitly trusted same-source
full-object checkpoints may load. Hash and size verification precede neural loading.
No historical checkpoint is migrated and no consumed sealed world is used.

## Fixed fixture and sequence

Use the existing initialization 42, world 91823, ten-step horizon, energy 100 fixture.
Separate parent trial IDs identify Colab and Kaggle lineages. Each parent runs the
existing 21-interaction / three-physical-update continuation and historical-reference
fixture, exporting its nine-experience pending checkpoint and uninterrupted terminal
state. Driver episode cursor must reflect the completed episode in all branches.

Within each parent fixture, launch a child that loads the pending checkpoint, atomically
publishes a complete recovery checkpoint, then receives SIGKILL before another action.
The supervising process must observe that signal, verify the recovery checkpoint's full
state against the parent, and continue from it in a fresh process. This tests abrupt
process loss after publication, not arbitrary VM loss or recovery of an unexported tail.
Preserve the killed branch and signal evidence. It adds no forward pass/update.

After export and release/termination of each parent job, upload each parent's declared
pending checkpoint through each provider. For each of four origin/destination pairs:

1. Public train request restores that checkpoint (no seed), completes its one remaining
   action and episode update. Compare all nine full-state group fingerprints and the
   action probabilities/sample/transition against that origin's uninterrupted reference.
2. Public evaluate request restores the same pending checkpoint, runs two development
   worlds 92823 and 92824 with the same ten-step configuration and no learning. Check
   zero updates, unchanged retained state and restored RNG/driver/mode/counters. The
   emitted checkpoint must be byte-identical to its input. Evaluation is engineering
   isolation evidence, not behavioral qualification.

The input contract stays provider-neutral: logical path, size, SHA-256, checkpoint role
and explicit trusted flag. Colab uploads only that file, then remotely stages/verifies it.
Kaggle uses a private checkpoint-only dataset per parent, with mounted-file hash checks.
Never upload CLI sessions, credentials, repositories or unrelated checkpoints. Preserve
private dataset/native job identities and package manifests. No paid resource selection.

## Gates and accounting

Upload must reproduce declared bytes; corrupted/ambiguous inputs fail before model load.
Resume must preserve source, lineage, objective, global budgets and all model/optimizer,
experience/baseline/history, environment/observation and Python/NumPy/Torch/CUDA RNG state.
For exact portability require bit equality, without tolerances or dropped state groups.
Per-group fingerprints encode tensor dtype/shape/bytes, arrays, dataclasses, RNGs and
container structure; never compare serialized checkpoint file bytes for numerical
continuation (serialization bytes can differ). Compare public action trace separately.
Evaluation must restore the complete input owner after its isolated branch, not save
mutated evaluation RNG as the continuing student. Duplicate completed requests cannot
silently create a new continuing student.

Within-provider public resume/evaluation gates precede admitting a portability PASS.
A transport/compatibility failure is retained as FAIL/NOT RUN, never reinitialized.
If cross-provider equality fails, record exact mismatched groups and stop migration
admission. Do not tune algorithms or runtimes against those outcomes in this trial.
No automatic migration/retry is enabled by these tests.

Budget: at most five Kaggle jobs (one parent, four public operations), each capped at
300 platform seconds, with fresh quota admission and a 2.5 h reserve. Initial observed
quota on 2026-10-30 was 3.00 h; insufficient quota blocks remaining tests. At most five
successful Colab assignments (one parent, four public operations), one allocation per
operation, bounded readiness and at most 1,050 s setup/execution each. No new allocation
retry on refusal. Use the shared controller's availability backoff. Parent worker cap
120 s, 3 updates/22 interactions; each resumed task uses the same global lineage budget,
performs at most 1 update/1 interaction; evaluation at most 20 interactions/0 updates.
Each output cap 64 MiB and Torch-allocated memory cap 128 MiB. Total successful neural
work is at most 10 updates and 126 interactions across independent/discarded branches.
This excludes CUDA readiness tensors; log all task/assignment/platform time separately.
No silent budget extension, CPU fallback or checkpoint choice based on performance.

## Intended outcome and limits

Report per-origin/destination upload, resume, evaluation, exact continuation and recovery
results. Keep all failures and local mock counts. Source IDs, native IDs, receipts,
state fingerprints and costs will be appended after execution. Large complete bundles
and checkpoints stay under ignored runs/, selected sanitized evidence in experiments/.
Even a full engineering PASS does not qualify self-directed learning or L.1 acquisition.
Automatic mid-training migration, durable periodic off-VM acknowledgments and abrupt
VM/preemption recovery remain unqualified until separately implemented and fault-tested.

## Recorded cap failure and transport-only recovery protocol

Original Kaggle evaluation 965fef14111e3cca0a73b444aac766cec1bbf2ec37c00e0101791096dbf2a6b9
was stopped at the 300 s platform cap. It reached package setup around 301 s; a partial
worker directory has no complete result. Preserve CANCEL_ACKNOWLEDGED/native duration
failure and all partial outputs. It is an infrastructure FAIL, not an evaluation PASS.
Dependent controllers were stopped before either cross-provider case was allocated.
Kaggle quota after that failure: 2.80 h; the 2.5 h reserve remains binding.

Before any recovery execution, admit one sequential, fail-closed Kaggle batch capped at
600 platform seconds. It runs the exact original evaluation request, then the already
frozen Colab-origin train and evaluation requests through separate public common-worker
processes. Both private datasets remain checkpoint-only/hash-verified. Reuse package
setup within that one GPU job; do not change worker source a88d7d386eb502e4ffae5bde7dc504441824c76d,
upstream, request bytes, objectives, input checkpoints, worlds, numerical profile or
comparison gates. The new adapter's published source is separately recorded. Verify
each child's expected state fingerprints before starting its successor. All bundles,
including earlier successful children on later failure, must be exported and retained.
No automatic retry: this is a separately recorded manual infrastructure recovery.

The total Kaggle reservation ceiling remains 1,500 s: three prior 300 s jobs plus one
600 s batch replaces the two unsubmitted 300 s jobs. At most four native Kaggle jobs,
with the evaluation retried once under an explicit attempt identity; no further retry.
Worker budgets stay unchanged, counting any partial neural work from the stopped job
as unknown rather than claiming zero. Colab retains its two unsubmitted cross-provider
assignments only after Kaggle evaluation passes. If quota admission fails or the batch
fails, stop and report the remaining gates NOT RUN/INCONCLUSIVE. Automatic migration
remains disabled. This amendment preserves the original failure and does not turn its
300 s cap into an assumed pass.
