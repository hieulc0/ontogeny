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
