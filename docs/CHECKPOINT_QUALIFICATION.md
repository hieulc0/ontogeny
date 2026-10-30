# Checkpoint transport and recovery qualification

Status: remote execution complete to available prerequisites; overall INCONCLUSIVE.
Engineering only; L.1 remains Learning FAIL.
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

## Measured outcome

Overall **INCONCLUSIVE**: reverse transfer was blocked by provider availability.
The tested fixture passed upload, public resume and public evaluation on both providers.
One cross-provider direction passed; no bidirectional or automatic-migration PASS.
L.1 remains Learning FAIL; no scientific acquisition cohort or later stage was run.

| Checkpoint origin → execution provider | Upload/resume | Exact next action/transition/update | Evaluation isolation |
|---|---|---|---|
| Colab → Colab | PASS | PASS, all nine state groups | PASS, 0 updates, byte-identical checkpoint |
| Kaggle → Kaggle | PASS | PASS, all nine state groups | PASS in recovery batch; original capped attempt failed |
| Colab → Kaggle | PASS | PASS, all nine state groups | PASS, 0 updates, byte-identical checkpoint |
| Kaggle → Colab | NOT RUN | NOT RUN | NOT RUN |

The fixture now records completed episode cursors consistently in uninterrupted,
resumed and reference branches; this driver metadata correction does not change the
original scientific runner or learning objective. Evaluation explicitly restores the
complete owner after its isolated branch, including RNG, driver, mode and counters.

Both parents passed fresh-process continuation, historical-runner comparison and the
SIGKILL-after-publication recovery test. Every public input was the exported recovery
checkpoint from that killed child, with nine pending experiences. Each executed public
train arm performed exactly one remaining interaction and one update; its loaded state,
all final-state groups and sampled trace matched the uninterrupted origin branch. Each
executed evaluation ran 20 interactions/0 updates, restored all state groups, and emitted
a checkpoint identical to its input. There were no observed numerical state mismatches.

Worker source: a88d7d386eb502e4ffae5bde7dc504441824c76d. Recovery adapter/protocol source:
74bc0e87f428748fe358cd6a6418d8e5d7f6dc15. All eight completed workers cloned the original
worker SHA and upstream 8567836fb2c8c915d442b7c2fbedb4fd9b30396d; the recovery adapter did
not migrate checkpoints to a different worker implementation. Colab was Python 3.13.16,
CUDA driver 580.82.07, one physical T4. Kaggle was Python 3.13.15, driver 580.178.04,
two physical T4s with exactly one visible to Torch. Both used Torch 2.6.0+cu124/CUDA12.4,
NumPy 2.1.3 and deterministic algorithms. Exactness is measured for this actual runtime
pair, not promised for arbitrary runtimes merely sharing these package names.

Completed parent recovery checkpoint hashes:

- Colab: 0ef35a0a2abd132e34a4b29cb865c26fd3ef3a58eadc3fd69b6e1dc0d8e15db2.
- Kaggle: 0cb59bf71cd5516d1d15a9dc512cb24c08d1a47add9a2a0b344096ab0f239470.

| Execution | Native identity | Outcome |
|---|---|---|
| Colab parent | ontogeny-5c476c89397a84ee | SUCCEEDED, exported, released |
| Colab same-provider resume | ontogeny-030ae4626f70a8e1 | SUCCEEDED, exported, released |
| Colab same-provider evaluation | ontogeny-9c702a707dd07b3a | SUCCEEDED, exported, released |
| Kaggle parent | sparklinghydraulics0/ontogeny-exec-5e7b0d60a1f6d9f7 v1 | COMPLETE |
| Kaggle same-provider resume | sparklinghydraulics0/ontogeny-exec-cfe783c0c13e7243 v1 | COMPLETE |
| Kaggle original evaluation | sparklinghydraulics0/ontogeny-exec-965fef14111e3cca v1 | Duration cap; CANCEL_ACKNOWLEDGED, normalized CANCELLED |
| Kaggle recovery batch | sparklinghydraulics0/ontogeny-exec-44c9b56a1a8e4a53 v1 | COMPLETE; three verified public worker bundles |
| Final Colab allocation | ontogeny-7295213f28dedb22 | Refused; no assignment, no neural work |

The original Kaggle evaluation spent about 301 s reaching the end of pinned package
installation and left an incomplete worker directory. It cannot be counted as a PASS.
The sequential recovery batch amortized setup across the unchanged evaluation and two
Colab-origin operations; each child had an exact-state gate before its successor. A DNS
error interrupted its first download from kaggleusercontent.com. A second download into
a separate directory recovered the same completed native job; no GPU work was repeated.
Raw failed collections contain signed URLs, so only sanitized failure metadata/hash is
versioned. The original stopped GPU job and its partial files are retained separately.

The final Colab create returned a precondition refusal (possible active-session, usage
or capacity limit), not a usable assignment. Its private session store was empty. The
controller set its 30-minute availability backoff and stopped after one create; it did
not poll, terminate unrelated sessions, use another accelerator or substitute a Kaggle
run for the missing reverse-direction test. All three successfully owned Colab sessions
were released. No known task-owned session remains active.

## Costs and limits

Eight complete bundles: 9 physical updates, 105 physical interactions, 1,217 observation
tiles, 38.435 s summed task time and 95.261 s summed outer worker time. Peak Torch
allocation was 79,681,024 bytes. These include successful parent audit/reference branches.
The stopped attempt has zero recorded prediction rows and no complete neural counters;
unobserved partial work is not silently counted as zero. These times exclude dependency
setup/allocation/download. Colab controller-observed allocation-to-release intervals
were about 239.7, 303.5 and 227.6 s (file timestamp observations, not billed GPU time).
Four Kaggle native jobs reserved at most 1,500 s total; reported GPU quota moved from
3.00 h to 2.74 h, retaining the 2.5 h reserve. Quota is rounded provider accounting,
not a CUDA kernel profiler. Full package inventories and physical GPU counts are retained.

Scope limitations remain binding:

- The pending input precedes the first optimizer update: optimizer moments and gradients
  are initially empty. The resulting nonempty post-update states match exactly, but
  importing a previously trained optimizer and making its next update is NOT qualified.
- One initialization, ten-step fixture; no mid-episode hazard relocation at interval 20,
  long rollouts, changed objectives or scientific generalization evaluation.
- Abrupt process loss after a complete checkpoint publication is tested. Arbitrary VM
  loss, unexported work, periodic durable external checkpoint acknowledgments, lost-tail
  accounting and automatic mid-training migration remain unqualified.
- Kaggle → Colab remains NOT RUN, not a numerical FAIL. The raw auditor reports a
  missing bundle; the qualification record maps that to the documented admission refusal.
- Historical students and old two-visible-device checkpoints were not migrated.

93 manually inspected local-safe tests, Ruff check/format, Pyright and syntax checks
passed. LOCAL: environment/fake/JSON/hash checks only; no neural imports or checkpoint
loads. REMOTE: all real neural forward/backward/load/evaluation checks on verified CUDA.
Upstream source, world dynamics, algorithm/capacity and historical learning results are unchanged.

## Reproduction and next admission

Full raw outputs are under runs/checkpoint-transport/. Versioned selected evidence and
immutable protocols: experiments/checkpoint_transport/. That directory is not a complete
binary bundle; each completed_bundles.json entry identifies its complete raw bundle.
The mechanical auditor is scripts/audit_checkpoint_transport.py and reads JSON/hashes only.

Use controller source 74bc0e87f428748fe358cd6a6418d8e5d7f6dc15 and worker source a88d7d386eb502e4ffae5bde7dc504441824c76d.
Reproductions are new explicitly recorded executions, not implicit retries of historical
requests. New checkpoint bytes require regenerating input SHA/size declarations.

```bash
# Freeze source and original protocol; qualification-request is the engineering fixture.
git show a88d7d386eb502e4ffae5bde7dc504441824c76d:docs/CHECKPOINT_QUALIFICATION.md > runs/new-protocol.md
ontogeny-exec qualification-request --source-sha a88d7d386eb502e4ffae5bde7dc504441824c76d --protocol runs/new-protocol.md --output runs/new-parent.json
# Give parent requests distinct trial_id values before execution, then validate them.
ontogeny-exec validate --request runs/new-parent.json
ontogeny-exec run --request runs/new-parent.json --inputs empty-inputs/ --directory runs/new-colab --ledger runs/new-controller --provider colab
# For Kaggle parent: prepare with --platform-seconds 300, then submit/status/collect.
# Stop/record infrastructure failure if setup exceeds that cap; no implicit resubmission.
ontogeny-exec checkpoint-request --parent-request runs/new-parent.json --checkpoint runs/new-colab/download/result/interrupted/recovery.pt --operation train --output runs/new-resume.json
ontogeny-exec checkpoint-request --parent-request runs/new-parent.json --checkpoint runs/new-colab/download/result/interrupted/recovery.pt --operation evaluate --output runs/new-evaluate.json
# Stage only the declared trusted checkpoint at inputs/checkpoints/parent.pt.
ontogeny-exec run --request runs/new-resume.json --inputs inputs/ --directory runs/new-resume --ledger runs/new-controller --provider colab
# Explicit Kaggle transport: upload-inputs; prepare --dataset returned-owner/slug;
# submit/status/collect with a bounded platform cap and explicit quota reserve.
```

The evaluation request helper deliberately implements this engineering fixture's two
worlds; generic scientific evaluation is supplied as an explicitly validated request,
not inferred from these smoke results. For the recovery batch, use public
kaggle.prepare(principal, job, platform_seconds=600, dataset_sources=(...),
batch=((principal, expected_pending_groups), (resume, expected_terminal_groups),
(evaluate, expected_pending_groups))). Then submit/status/collect in its new explicit
attempt ledger. batch_submission.json and batch_launch.py.txt preserve exact submitted
requests/gates/code. The principal and children still call the unchanged common worker.

Next: finish the unavailable reverse-direction case after Colab becomes eligible in a
new explicit attempt, preserving the refusal/backoff record. Then preregister a stronger
continuation fixture with existing optimizer moments, gradients and multiple completed
updates before interruption; compare the next updates in both directions. Only afterward
implement periodic external checkpoint export/ACK and kill-after-ACK recovery with
explicit lost-work accounting. Keep automatic checkpoint routing disabled until those
gates pass. No L.1 promotion, L.2 or additional cell is admitted by this engineering work.
