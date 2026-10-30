# Common worker qualification

Status at preregistration: IMPLEMENTED, LOCAL CONTRACT TESTS PASS, REMOTE NOT RUN.
This is engineering qualification only. It cannot change the L.1 Learning FAIL verdict.
The research/Colab design drafts remain distinct from this implemented increment.

## Scope and frozen protocol

Implement a strict JSON request/result contract, bounded common worker and a Kaggle
transport. Preserve original PolicyNetwork, PolicyCell, World and optimizer behavior.
No Colab adapter, automatic routing, architecture growth or new scientific cohort.
Upstream exact pin 8567836fb2c8c915d442b7c2fbedb4fd9b30396d; Ontogeny execution SHA is
recorded in the generated request and actual runtime. Publish source before preparing.

One fresh engineering lineage, initialization 42, world seed 91823, episode horizon 10,
initial/max energy 100, other WorldConfig defaults. This previously used safe fixture
ensures any ten-action trajectory survives to its horizon. No original sealed worlds.
Same actual MiniGPT configuration (118,023 parameters) and unchanged policy objective:
REINFORCE, AdamW lr .0005, gamma .97, baseline decay .9, clipping 1.

Kaggle Tesla T4, Torch 2.6.0+cu124, NumPy 2.1.3. One visible CUDA device configured BEFORE
import; physical allocation inventory captured separately. This is a new lineage, not
a migration of historical two-visible-device checkpoints. No cross-provider claim.

Maximum worker task time 120 seconds; Kaggle platform cap 1,200 seconds including setup.
At most 3 physical optimizer steps and 22 physical interactions; actual expected 21.
Output cap 64 MiB; Torch-allocated memory cap 128 MiB. That memory metric excludes driver
context and other processes; no OS hard-memory or filesystem quota is claimed. Worker
monitors disk usage while its task runs and requires publication headroom.

Before submission, fetch fresh Kaggle GPU quota. The initial observed value was 3.06 h
remaining, reset 2026-10-31T00:00:00Z. Retain explicit 2.5 h reserve plus platform cap
and .01 h rounding uncertainty. Insufficient/unknown quota blocks submission. The reserve
is an explicit narrow qualification allocation, not the earlier draft six-hour reserve
assumed to exist despite a smaller real remaining quota. One attempt only, no auto retry
of a submitted/ambiguous run and no local CPU fallback.

## Binding acceptance gates

1. Strict schema rejects unknown fields, bad versions/types, unsafe paths, malformed
   hashes, contradictory initialization, train-on-test and inadequate budgets.
2. Inputs are trusted declared full checkpoints only, matched by exact hash/size before
   load; fresh jobs never silently substitute a checkpoint. Source imports match pins.
3. Common worker refuses local/CPU, wrong provider/runtime/GPU/device count. Context
   is trusted controller metadata, not a security sandbox or cryptographic attestation.
4. Save after nine observed experiences. Continue action ten and one actual update.
   In a fresh process load that pending checkpoint and repeat the same action/update.
   Match distribution/action/transition and deep-copied model/optimizer/scheduler,
   owned buffers/world/metadata/gradients/RNG/training flags exactly.
5. From the same initial checkpoint run the UNCHANGED historical runner and policy for
   ten steps. Its actions and full final learner state must match the instrumented run.
   Both audit copies are discarded; no favorable result/checkpoint choice.
6. Recover the bundle through the Kaggle adapter after native COMPLETE. Verify scientific
   request identity, required files, all indexed bytes/hashes, no unindexed/symlink files,
   checkpoint publication, physical counters and limits. Save collection receipt.
7. Execution result is SUCCEEDED only if complete; scientific result stays NOT_RUN.
   A queued/completed notebook or exit zero without outputs does not qualify.

Local failure tests use fakes/opaque bytes only. Include timeout/cancellation/partial
artifact preservation, corruption, single-attempt idempotency across calls, unknown
quota/refusal and Python -O validation. GPU work is exclusively remote. Remote platform
termination itself is not deliberately induced; recovery evidence here is cold-process
continuation plus collection after completion, not a preemption/durability guarantee.

## Implemented interface

Authoritative request parser: execution/contracts.py; schema_version=1. JSON keys and
nested objects reject extras. Operations are train, evaluate, continuation_check. The
first release supports the fixed historical neural/runtime profile only. New profiles,
algorithms or scientific verdict evaluators require their own explicit extension; these
are current supported experiments, not permanent architectural ceilings.

Request includes run/trial/arm IDs, protocol and split-manifest SHA-256, split, both source
SHAs, runtime profile, explicit fresh seed OR full checkpoint, world recipe, objective,
engineering target, input artifact descriptors, global limits and required outputs.
Train only accesses train split; evaluate development/test; continuation diagnostic.
No arbitrary shell or callable from an input file. Worker dispatch is package-owned.

Worker operations share input/output roots independent of provider native directories.
Kaggle maps mounts into logical inputs by hash/size; explicit dataset references required
for resume/evaluation submission. Initial transport supports unpacked files, not implicit
zip extraction or interactive Drive mounts. The client supplies trusted artifact identities.
Use controller-owned input directories; pickle is never safe for untrusted artifacts.

Train resumes preserve source/protocol/arm/world/objective and the same global budgets;
requesting fresh state is always explicit. Evaluation uses an isolated loaded copy and
publishes the original immutable evaluated checkpoint, never its consumed RNG state as
a promoted learner. It checks model/optimizer/scheduler/gradients/retained buffers and
learning counters for mutation. Global scientific budget and physical audit/replay cost
are separate counters; no implicit reset to evade a cap.

Supported commands are the low-level lifecycle `python -m ontogeny.execution.cli`:
qualification-request, validate, prepare, submit, status, collect and verify. Kaggle native
job names include request digest and successful submission pins native version. A locked
file journal records reservation BEFORE push. Unconfirmed submission stays UNKNOWN;
repeated submit returns its existing record without pushing again. Reconcile status
rather than launching another version. Concurrency is conservatively one per shared
ledger; multiple independent ledgers are not a distributed/account-wide scheduler.

Cancellation: the common worker accepts a controller-owned cancel-file and terminates
its task process group, preserving partial artifacts. Kaggle native remote cancellation
is not exposed in this release; use provider UI for urgent stop, or its finite platform
cap. No automated capacity queue, periodic durable export, resubmission or resume routing
is claimed. Native session-limit errors are surfaced, not converted into silent retries.

Required bundle: request.json, runtime.json, execution.json, metrics.json,
predictions.jsonl, learning.jsonl, checkpoint_manifest.json, full checkpoint files,
result.json and artifacts.json. Failure adds failure.json and task.log. Artifacts publish
immutably; index is last. Result schema v1 separates execution SUCCEEDED/FAILED/CANCELLED
from science NOT_RUN. Adapter collection validates and writes a receipt outside the bundle.
Hashes detect corruption/misassociation, not a malicious provider editing payload and index.

## Reproduction and evidence

Use new directories to retain previous failures. Exact execution SHA/request digest,
Kaggle native ID/version, remote status, receipt, counters and actual gate results will
be appended after execution. Example commands after publishing executable source:

```bash
uv run python -m ontogeny.execution.cli qualification-request --source-sha <execution-sha> --protocol docs/EXECUTION_QUALIFICATION.md --output runs/execution/request.json
uv run python -m ontogeny.execution.cli validate --request runs/execution/request.json
uv run python -m ontogeny.execution.cli prepare --request runs/execution/request.json --job runs/execution/job
uv run python -m ontogeny.execution.cli submit --job runs/execution/job --ledger runs/execution/ledger --reserve-hours 2.5
uv run python -m ontogeny.execution.cli status --request runs/execution/request.json --ledger runs/execution/ledger
uv run python -m ontogeny.execution.cli collect --request runs/execution/request.json --ledger runs/execution/ledger --directory runs/execution/download
uv run python -m ontogeny.execution.cli verify --request runs/execution/request.json --directory runs/execution/download/result
```

A scientific training request is not automatically created from this fixture. No
scientific acquisition or transfer result follows from these engineering gates.

## Actual result

**Engineering qualification PASS for the tested Kaggle continuation task.**
Scientific result remains NOT_RUN; L.1 Learning FAIL is unchanged.

- Tested source: `f8c4bc88cf53ae8b1b82d37fa33fad7a212eeedf`.
- Request digest: `2fc67f1e54ff475818e1b109e1783017ae8a34d7a10c8eeeaabfa8d7bb328b19`.
- [Kaggle run, version 1](https://www.kaggle.com/code/sparklinghydraulics0/ontogeny-exec-2fc67f1e54ff4758): COMPLETE.
- Common worker SUCCEEDED, 13.704 s including its admission; task 7.439 s excluding
  package installation/VM setup. Kernel active time was not separately profiled.
- Python 3.13.15, Torch 2.6.0+cu124, CUDA 12.4, driver 580.178.04, T4. Two physical
  devices allocated, one visible. Full package/version inventory retained in runtime.json.
- All nine full-state groups, sampled action/distribution/transition and the unchanged
  historical runner comparison match exactly. Nine pending experiences restored.
- Three physical optimizer steps / 21 physical interactions, one candidate logical
  update / ten logical interactions. Candidate final energy 82; this is a test fixture,
  not a behavioral qualification metric.
- Peak Torch-allocated memory 77,035,520 bytes <128 MiB. Indexed bundle 9,646,693 bytes
  <64 MiB. Adapter recovered and verified all 21 files plus immutable checkpoint hashes.
- Reported GPU quota 3.06 h before, 3.00 h after. Values are rounded account reports,
  not a precise profiling or billing measurement; protected reserve remains 2.5 h.
- Local final checks: 72 non-neural tests, Ruff check/format, Pyright, compile checks PASS.

Small versioned evidence: experiments/execution_foundation; complete binary bundle and
raw job logs: runs/execution-foundation/download. The versioned evidence directory is
selected records, NOT itself a complete verified result bundle.

The original metrics.json observation_tiles field counts candidate sensory observations
(122), while interactions/updates count physical audit work too. This was clarified after
remote execution: the new physical_observation_tiles metric and budget audit count all
three branches (253). A local regression test and replay of recorded traces verify that
accounting-only addition; the remote state/behavior evidence remains pinned to the tested
source above. No repeat GPU run or scientific change was made for this telemetry addition.

Two local setup issues were preserved: initial build dependency fetch failed under the
sandbox's DNS restriction, then succeeded with authorized package access; formatting/type
issues during development were corrected before publication. There was no failed neural
qualification attempt and no result-driven retry.

### Reproduce the exact published run

The request file contains its immutable protocol/source identities. Use it directly,
not a newly hashed version of this now-extended report. From the final clean checkout:

```bash
uv sync --locked
ontogeny-exec validate --request experiments/execution_foundation/request.json
ontogeny-exec prepare --request experiments/execution_foundation/request.json --job runs/reproduce-worker/job
ontogeny-exec submit --job runs/reproduce-worker/job --ledger runs/reproduce-worker/ledger --reserve-hours 2.5
ontogeny-exec status --request experiments/execution_foundation/request.json --ledger runs/reproduce-worker/ledger
# Collect only after COMPLETE; an ERROR's partial artifacts remain available for diagnosis.
ontogeny-exec collect --request experiments/execution_foundation/request.json --ledger runs/reproduce-worker/ledger --directory runs/reproduce-worker/download
ontogeny-exec verify --request experiments/execution_foundation/request.json --directory runs/reproduce-worker/download/result
```

A new explicit reproduction may create another native version; the original version 1
and selected evidence remain intact. Repeated submit with the SAME ledger never submits
again. Request/artifact-index digests use canonical JSON; individual file SHA-256 values
hash raw bytes. No local checkpoint load is needed for collection or verification.

### Remaining scope and next decision

The common request/worker boundary and Kaggle continuation transport are operational.
Public train/evaluate handlers use the same PolicyCell and have contract/mock coverage;
separate end-to-end uploaded-checkpoint resume/evaluate submissions have NOT yet been
GPU-qualified. The actual test qualified an owned pending checkpoint restored in another
process within one remote job. Abrupt VM loss, periodic off-VM checkpoint ACK, native
Kaggle cancellation and account-wide multi-controller reservations are not qualified.
Split labels protect operation dispatch; the first release uses explicit inline world
recipes, not a general external dataset/sealed-manifest authorization system.

Next: close the public staged resume/evaluate path, then add Colab transport using this
same contract/worker and explicit artifact export-before-stop. Move provider admission
into a validated adapter boundary while preserving current neural import guards and
historical checkpoints. Update AGENTS explicitly when enabling Colab. No fake Kaggle
flag and no CPU fallback. Only after per-provider qualification should routing, cooldown
caching and automatic recovery be enabled. Cross-provider numerical continuation remains
unqualified even with a single visible device. Scientific matched credit/binding work
can proceed separately on qualified Kaggle; do not turn infrastructure PASS into L.1 PASS.

Subsequent provider increment: fresh-job Colab readiness/common-worker execution is
now qualified; see COLAB_ROUTING.md for its different tested source and actual cleanup
recovery. The earlier remaining-scope list is historical. Uploaded-checkpoint and
cross-provider continuation qualification remain outstanding.

Checkpoint follow-up: public upload/resume/evaluation passed the declared fixture on
both providers; Colab→Kaggle exact continuation passed, while Kaggle→Colab was blocked
by Colab allocation refusal. SIGKILL after checkpoint publication passed in both parent
jobs. The input optimizer is still before its first update, so previously trained
optimizer migration remains unqualified. Actual cap/DNS failures and scoped results:
CHECKPOINT_QUALIFICATION.md. Automatic migration remains disabled; overall bidirectional
qualification INCONCLUSIVE, L.1 STILL FAIL.
