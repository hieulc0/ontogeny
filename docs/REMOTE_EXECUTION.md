# Portable remote execution — design draft

Status: Kaggle foundation and fresh-job Colab readiness/routing implemented;
Colab common-worker engineering qualification PASS. Public checkpoint fixture transport
and evaluation pass on both providers; Colab→Kaggle exactness passes, reverse direction
is blocked. See [checkpoint qualification](CHECKPOINT_QUALIFICATION.md). Automatic migration
and arbitrary VM interruption recovery remain unqualified. See [COLAB_ROUTING.md](COLAB_ROUTING.md). Research checked 2026-10-30. Actual implemented scope, tested source and limits:
[EXECUTION_QUALIFICATION.md](EXECUTION_QUALIFICATION.md). The larger design below includes
unimplemented scheduling/portability features. Fresh jobs now accept validated Colab
or Kaggle CUDA; historical experiments and checkpoint identities remain unchanged.

Current implementation uses request schema2 and the internal mini_gpt engine; normal
launchers clone only Ontogeny. Earlier qualification numbers in this document belong
to their pinned pre-consolidation sources. See MIGRATION.md for the new code/state
contract and measured verification. The larger design remains a proposal where marked.

## Decision

Expose one training/job contract, implemented by a common remote worker. A controller
selects Colab or Kaggle and transports artifacts. The learner receives device, input
paths, limits and owned state; it never branches on provider names or directory layouts.
Provider identity remains visible in logs/provenance. Scientific equivalence is tested,
not inferred from a shared interface or a T4 label.

Default routing: reuse a compatible healthy session owned by this job if one exists;
otherwise attempt Colab T4 once when eligible. On unavailable/quota/cooldown allocation,
try Kaggle after a fresh quota and session-capacity check. If neither is admissible,
queue with a reason and next eligibility time. Never silently use CPU, enlarge budgets,
change algorithms, create another account, or retry allocation in a tight loop.

## What providers actually promise

Colab's free resources and session lengths are dynamic; neither six GPU hours nor a
fixed cooldown is a published guarantee. The FAQ describes variable usage limits and
up-to-12-hour free notebook lifetimes subject to availability/use. Our scheduler must
use observations and conservative deadlines, not assume a six-hour replenishment clock.
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

Kaggle describes weekly GPU quota as 30 hours or sometimes higher. The live account
report is authoritative for admission; an advertised allowance is not current remaining
time or guaranteed capacity. [Kaggle GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage).

Google supplies a Colab CLI with T4 allocation, remote execution and artifact retrieval;
a request can still fail because of account entitlement/capacity. Authentication may
require a one-time human step; it must not hang an unattended controller.
[Official CLI](https://github.com/googlecolab/google-colab-cli),
[Google announcement](https://developers.googleblog.com/introducing-the-google-colab-cli/).
Use that supported workflow; no browser keep-alive, SSH workaround or quota evasion.

Local read-only inspection confirmed the installed Kaggle CLI supports
`kaggle quota --format json`: resource, used, remaining, total and refreshAt. Its API
exposes time_used, total_time_allowed and quota_refresh_time. Prefer numeric API values
where possible because formatted CLI hours are rounded. Pin and test the adapter's CLI
version/output parser. Never confuse notebook accelerator quota with other Kaggle APIs'
quotas. [Official implementation](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py).
No authenticated quota/account request was made during planning.

## Boundaries and proposed package layout

```text
execution/
  contracts.py       request, result, lease and normalized failure records
  controller.py      routing, reservations, leases, reconciliation
  artifacts.py       hash verification, local/durable transfer acknowledgements
  worker.py          common preflight, bounded invocation, output publication
  providers/
    colab.py         allocate/status/execute/transfer/stop
    kaggle.py        quota/submit/status/output/cancel
```

This is a proposed responsibility map, not a requirement to create empty files/classes.
Start with a single-process controller and a small atomic file-backed journal. No
scheduler cluster, database service, workflow engine or neural algorithm rewrite.

A provider implements capability inspection, availability/quota observation, allocation
or batch submission, status reconciliation, artifact transfer and release/cancel. It
normalizes Colab interactive leases and Kaggle asynchronous kernels to one lifecycle;
it does not pretend they expose the same native operations or durability guarantees.
Training, evaluation and checkpoint publication are worker tasks, not provider logic.
For Colab, prefer explicit allocate -> execute -> download/export -> acknowledge -> stop
in a finally/cleanup path. A one-shot command that destroys the VM before collecting
artifacts is insufficient. Set the pinned CLI's execution/transport timeout explicitly;
it is distinct from the worker scientific wall-time cap. Inspect its help/defaults
before implementation, rather than inheriting a short interactive timeout.
[CLI execution contract](https://github.com/googlecolab/google-colab-cli/blob/main/docs/05_run_command.md).

The controller is trusted orchestration; the learner is not given credentials, provider
quota, a cloud API, evaluator secrets or an ability to provision more compute.

## Input contract

Use a schema-validated JSON manifest as the canonical interchange format. TOML remains
an authoring format; normalize it once before freezing/hashing. Every unknown/incompatible
schema must reject rather than silently default. Versions here are actual serialization
contracts, not milestone names in reusable implementation modules.

Required fields:

| Field | Meaning |
|---|---|
| schema_version | Request-format compatibility |
| experiment_id / protocol_digest | Frozen scientific protocol identity |
| run_id / trial_id / arm_id | Logical experiment and paired replicate identities |
| operation | train, evaluate, diagnose, continuation_check, or artifact_audit |
| entrypoint | Allowlisted package entry point and declared arguments, no arbitrary shell |
| source | Ontogeny and upstream URLs/full SHAs, package/build identity, content hashes |
| runtime_profile | Python, Torch/CUDA, dependency lock/wheel hashes, numerical policy |
| inputs | Logical artifact names, sizes, SHA-256, trusted lineage, access references |
| initialization | Exactly one of explicit fresh seed or full parent checkpoint |
| world / observations / actions | Definitions and schema hashes; context and action dimensions |
| objective | Algorithm, feedback transformation, optimizer and update settings |
| targets | Scientific hypotheses/criteria references; never privileged correct-action labels |
| split_manifest | Train/dev identity and generator hashes; final-test access separately authorized |
| limits | Interactions, updates, wall time, memory, disk, queries, attempts and cost bounds |
| checkpoint_policy | Safe-point cadence, export deadline, retention, resume compatibility |
| outputs | Required artifact types, logical destinations, manifest contract |

Keep scheduling preferences in a separate envelope so fallback does not modify the
scientific spec/digest. Example, illustrative only:

```json
{
  "job": {
    "schema_version": 1,
    "run_id": "credit-binding-comparison",
    "trial_id": "init-42-terminal-binding",
    "operation": "train",
    "protocol_digest": "sha256:<frozen-protocol>",
    "source": {
      "ontogeny_sha": "<full-published-sha>",
      "upstream_sha": "8567836fb2c8c915d442b7c2fbedb4fd9b30396d"
    },
    "runtime_profile": "torch-cuda-t4-strict",
    "initialization": {"kind": "fresh", "seed": 42},
    "objective": {"kind": "policy_gradient", "config_ref": "sha256:<config>"},
    "targets": {"criteria_ref": "sha256:<qualification-rules>"},
    "limits": {"updates": 2048, "interactions": 6144, "wall_seconds": 1200},
    "outputs": ["metrics", "events", "full_checkpoint", "runtime", "manifest"]
  },
  "schedule": {
    "provider_order": ["colab", "kaggle"],
    "accelerator": "T4",
    "logical_cuda_devices": 1,
    "local_cpu_fallback": false,
    "max_execution_attempts": 2
  }
}
```

Exact resource ceilings and artifact/runtime fields must be filled before submission;
this abbreviated example is not a complete valid production request.

Fresh versus continuing state must be explicit. A checkpoint's optimizer, baseline,
experience, counters, RNG, sampler cursor and environment are one unit. No weights-only
fallback when the caller requested a continuing learner. Evaluation requests identify
the immutable final checkpoint and cannot call learning updates.

## Worker and output contract

One entry point accepts request.json, a verified execution context, staged input root
and output root. Provider adapters map logical paths to /content or /kaggle/working;
training code never hard-codes those paths. Stage allowlisted artifacts by content hash;
never upload an entire home directory/repository with secrets. Private Kaggle inputs
must have an explicitly authorized Colab transport; do not assume its mounts transfer.

Preflight checks source/import identity, GPU model/CUDA availability, memory, visible
GPU count, dependency profile, deterministic settings, observation/action dimensions,
input hashes, trusted checkpoint lineage and external budgets. No model operation until
remote admission succeeds. Do not set a fake KAGGLE environment flag on Colab to bypass
runtime admission. The worker now validates provider-specific execution context and
real environment markers while preserving local refusal.

Every result bundle has:

```text
request.json                 frozen scientific input
execution.json               provider, attempt, lease/session/kernel IDs and timelines
runtime.json                 full Python/Torch/CUDA/driver/device/package inventory
metrics.jsonl                per-arm/seed/split measurements with units and counters
predictions.jsonl            observed inputs, distribution, sampled action, feedback
learning.jsonl               returns, advantages, loss, gradient and update records
checkpoints/                 immutable full learner states and parent hashes
checkpoint_manifest.json     schema, hashes, counters, compatibility and latest ACK
artifacts.json               file hash/size/type index; publish last
result.json                  execution outcome, references and independent verdict
failure.json                 structured failure and traceback when applicable
```

Large traces may be losslessly compressed; schema/count/index references stay explicit.
Every event has sequence number, UTC time, monotonic duration, logical step, attempt,
arm, initialization, episode and world identifiers, where applicable. Record RNG stream
identity; never leak hidden simulator maps into predictions or learner input traces.

Report allocated session-seconds, physical-device-seconds, active training wall time,
setup/export/queue times and measured kernel time if profiled as distinct fields. Unknown
provider billing equivalence stays unknown; two allocated T4s are not automatically
assumed to consume twice the reported notebook quota. Reconcile with provider observations.
Count failed attempts, discarded audit updates and duplicated replay after interruption.

Example result semantics: execution=SUCCEEDED with science=FAIL is valid. A queued kernel,
exit code zero, or completed notebook without complete verified required outputs is not
success. Scientific outcomes are PASS/FAIL/INCONCLUSIVE/NOT_RUN and come from an independent
evaluator. Failure to allocate yields NOT_RUN, not scientific FAIL.

## Scheduling and quota strategy

1. Validate/hash locally, freeze protocol, package inputs and estimate resource envelope.
2. Under a controller lock, reserve a trial and reconcile existing provider sessions.
   Reuse only a healthy compatible lease owned by this run; never seize another task's VM.
3. If Colab is eligible, make one explicit T4 allocation attempt with a bounded timeout
   (proposed 120 s). Verify actual hardware. 503, capacity/quota failure or unsupported
   T4 entitlement immediately opens a circuit and routes this request to Kaggle.
4. Before Kaggle submission, query quota and active sessions. Admission must satisfy
   fresh reported remaining time minus outstanding reservations minus protected reserve
   >= setup + predicted work + export + uncertainty margin. Then reserve atomically.
5. Submit/persist native job identity. Reconcile ambiguous responses before any retry;
   a timeout can mean a job was accepted. Never launch a duplicate as an optimistic hedge.
6. If Kaggle is out of quota/capacity or evidence is stale/unknown, queue. A controller
   may use a conservative reservation ledger only when it has exclusive account visibility;
   otherwise UNKNOWN does not mean available. Do not advertise invented quota API values.
7. Stop/release owned sessions after verified exports. Reconcile usage and release
   reservations; preserve failure traces. No repeated live allocation probe during cooldown.

Cache known Colab unavailability across jobs, rather than repeatedly issuing new-session
requests. Suggested *internal* retry eligibility: 30 minutes for transient capacity,
2 hours for an unknown quota denial, exponential increases capped at 12 hours for
repeated denials. Use provider retry-after when available. These are conservative
controller settings, not a claim about Colab's actual reset schedule. One probe when
eligible; fail once then fall back. Authentication failures require account repair;
validation/numerical failures stop the trial and do not trigger provider shopping.

Default batch concurrency one until reservations/idempotency are qualified. A confirmed
quota reset permits fresh admission; a weekly timer alone does not prove it. No
self-imposed reserve reset simply because provider routing changed.

### Initial weekly envelope (proposal)

If the account reports 30 h available: allocate at most 24 h to ordinary work and
protect 6 h for confirmation, recovery and final evaluation. Suggested 24 h split:
4 h provider/continuation validation, 8 h matched diagnosis, 8 h one admitted repair,
4 h verification. These are ceilings, not usage targets or permission to run all stages.
If observed quota is smaller, scale/defer; reserve release is an explicit controller
budget decision declared before a qualification cohort, not automatic desperation.
Colab provides opportunistic capacity, not hours already available in the budget.

Initially package 10–30 minute jobs; cap a scientific worker segment at 60 minutes.
Avoid one six-hour fragile script. Checkpoint at episode/update boundaries at least every
2 minutes or 32 updates, whichever occurs first; cap pending episode length. Proposed
maximum 5 minutes lost computation only after durable export cadence is measured. Export
and verify in-progress checkpoints independently of final notebook artifact publication.
If a provider cannot durably export mid-run, its shorter job boundary is the durability
boundary; do not claim the loss bound. Group small related tests within one bounded
session to amortize the observed setup overhead; do not leave idle GPUs alive for reuse.

## Interruptions, cooldowns and exact continuation

A lease loss causes reconciliation, then resume from the last hash-verified durable ACK,
with the SAME counters/state and remaining global budget. Updates since that ACK are
uncommitted work; replay may consume extra cost, never extra scientific training budget.
Each segment has a unique attempt and monotonically increasing ownership/fencing token;
only its current owner may publish the next accepted checkpoint. A local journal cannot
stop a remote orphan by itself, so reconcile/cancel it before migration. State UNKNOWN
remains unresolved, not permission for duplicate execution.

Current upstream LearningState snapshots include ALL visible CUDA RNG states and reject
a changed CUDA-device count. Historical Kaggle runs allocated two visible devices;
Colab may expose one. Existing checkpoints therefore cannot be assumed portable or
resumed by dropping an RNG state. Also, matching T4 hardware does not guarantee identical
drivers, kernels or numerical trajectories.

For NEW runs propose a canonical single-visible-device process, configured before Torch
import on both providers. Preserve physical allocation metadata. Admit cross-provider
resume only after checking exact next distribution/action, environment transition,
optimizer step, tensors, gradients, buffers and RNG in a fresh process on each runtime
profile and both directions. If equality fails: default provider-affine learner state;
new independent trials may route anywhere but continuing an arm stays on its qualified
runtime. Historical two-device checkpoints remain provider-affine unless an explicit
GPU-tested reversible migration is separately qualified. Do not reset or replace them.
Behaviorally close continuation is a different experiment with a declared tolerance,
not a renamed deterministic PASS.

Within a paired scientific comparison, initially keep both arms on the same provider/
runtime profile and record provider as a block. Fallback between independent complete
replicates is allowed only by preregistered rules; report per-provider differences.
No result-based rerouting, opportunistic extra seeds or success-driven retries.

## Qualification and implementation order

| Work | Local-safe evidence | Remote evidence needed |
|---|---|---|
| Contract/worker | Invalid fields, hashes, split misuse and incomplete-output rejection | Common worker refuses CPU/wrong GPU/runtime |
| Kaggle adapter | Mock async lifecycle, quota parser incl stale/rounded values | Pinned small job, artifact recovery, real quota reconciliation |
| Colab adapter | Mock allocation error taxonomy, auth/timeouts/cleanup | T4 acquire, same worker, export before stop, bounded release |
| Routing | Single Colab denial -> one Kaggle submit; cooldown persists | Controlled availability failure, no duplicate jobs |
| Persistence | Journal restart, lease conflict, corruption, retry accounting | Mid-episode/pending-buffer interruption and fresh-process continuation |
| Portability | Compatibility profile/migration rejection | Both directions exactness, device-count mismatch rejection |
| Shutdown | Always attempt cleanup; unresolved cancellation reported | No task-owned session left running, exports recoverable |

Build contract/common worker first, move current Kaggle workflow behind it, then add
Colab and routing. Test transport with non-neural artifacts before touching checkpoints.
One bounded GPU canary on each provider precedes production auto-routing. Predeclare
infrastructure validation budget; suggested initial cap 4 reported Kaggle hours and
60 minutes of successful Colab allocation time. Missing auth/quota blocks the respective
qualification rather than causing local inference. Error/failure injection is bounded,
never a quota-exhaustion exercise.

Proposed user surface (not existing commands): `ontogeny train --spec ... --provider auto`,
`status`, `cancel`, `resume`, `artifacts`, and `resources`. Algorithm entry points stay the
same across providers. Output reports selection, rejection reason, checkpoint lineage
and usage; no promise that free infrastructure is always available.

AGENTS and runtime admission now explicitly permit validated remote Colab. The broader
features proposed here still need their own implementation and qualification.
Use functional module names and retain historical import paths/checkpoints. CLI sessions,
outputs and existing serialized classes must not be renamed as cosmetic cleanup.
