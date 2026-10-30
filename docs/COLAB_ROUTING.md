# Readiness-based provider routing

Status: implemented; local tests pass; common-worker Colab GPU qualification pending.
No scientific cohort, new objective, architecture change or old checkpoint migration.

The next increment addresses allocation/readiness semantics, not only HTTP errors.
The installed CLI catches allocation refusal/entitlement errors such as 412/400;
503 is not a sufficient failure detector. It also writes global logs even for help,
so a local permission error must not be presented as GPU quota exhaustion.

## Strategy

1. Freeze/validate scientific request and published source before allocation. Keep
   provider selection outside learner configuration. No CPU fallback.
2. Make ONE Colab T4 create request using a unique task-owned session and private state
   store. Do not use a different person's existing session. Do not assume exit zero,
   the word READY or a cached requested accelerator means the GPU is usable.
3. Confirm an assignment, then run a nonce-bound remote probe. Require actual CUDA,
   a T4 device, a completed GPU allocation/synchronization, a real Colab environment
   marker and successful proof-file download. A known assignment may warm up for a
   bounded 120 seconds (at most two probes); no second create.
4. If allocation is refused/unavailable, or the owned assignment never becomes ready,
   release/reconcile it before Kaggle fallback. Remember unavailability for 30 minutes
   (internal backoff, not a claim about Google's actual cooldown). No Colab polling
   during that period. Recheck on a later eligible request, never in a tight loop.
5. Allocation timeout is ambiguous: server-side resources may exist. Mark UNKNOWN,
   reconcile owned sessions and block blind fallback/new creation. Cleanup uncertainty
   is also UNKNOWN. Do not terminate unrelated sessions.
6. Once the worker starts, a numerical/validation/training failure does NOT cause a
   provider switch or fresh-model retry. Preserve outputs and fail the trial. Only
   availability/readiness failures before starting the trial permit automatic fallback.
7. If Colab is ready, execute the same pinned common worker, download/export the result,
   check artifact hashes/request identity, then stop and reconcile the owned session.
   Exit zero from execution/download/stop alone is not evidence of success. Always
   attempt release on errors; unresolved cleanup blocks automatic resubmission.
8. Otherwise submit through the existing quota-aware Kaggle adapter, retaining the
   explicit reserve. If quota/capacity is inadequate, report blocked/failed admission
   rather than run locally or expand the experiment. Kaggle completion/collection use
   the existing adapter's ledger; routing records preserve the selected native job.

The route ledger is single-process/account-workspace coordination, not a distributed
scheduler. Repeat run for the same request returns its existing record without another
allocation/submission. Per-request journals and CLI credential state are separate.
Never commit private CLI session files/logs; only sanitized outcomes enter research
records. Account authentication is preconfigured; stdin is closed and commands timed
out so interactive login cannot hang an unattended job. No automatic upgrade or paid
accelerator choice; only the requested T4 is accepted.

## Qualification protocol

Readiness-only probe: one real Colab T4 assignment, remote torch.empty(1) and CUDA
synchronization, nonce proof downloaded, owned session stopped. This succeeded before
implementation. It is a transport probe, not training or neural quality evidence.

Common-worker test: one new fresh lineage with the existing ten-step engineering
fixture, three physical optimizer steps maximum (main/resume/historical reference),
22 physical interactions maximum, one visible CUDA device. Same policy, optimizer and
exact upstream8567836fb2c8c915d442b7c2fbedb4fd9b30396d. Worker120s,64MiB output,
128MiB Torch-allocated memory. Allow <=1,050 seconds for pinned setup/execution and
bounded download/cleanup; at most one successful Colab trial. Preserve original inputs,
complete checkpoints, package inventory and all state comparisons. No L.1 PASS claim.

Submit via `ontogeny-exec run --provider auto` only after publishing the exact source.
Colab context validation now uses actual COLAB_RELEASE_TAG; Kaggle uses its own real
marker. Existing serialized model class paths stay intact; no fake KAGGLE flag is set.
Public worker API/source/runtime checks remain the same. Historical provider-affine
checkpoints are not thereby declared cross-provider portable.

Checkpoint upload/resume on Colab remains blocked until separately qualified. Only
fresh requests route through this first Colab transport. Public staged resume/evaluate
qualification from the previous increment remains outstanding; no claim it was completed
by this fresh-state provider test. Mid-run durable export/preemption recovery, account-
wide scheduler, and cross-provider next-update exactness remain separate work.

## Local test coverage

Silent zero exit without assignment; explicit non-503 refusal; READY text without proof;
nonce-bound remote proof; ambiguous timeout; unsafe/archive oversized artifacts; exactly
one fallback submission and idempotency; no provider hopping after the worker starts.
Mock tests never import Torch or create a real session. Sandbox CLI-help logging failed
under read-only home config; actual authorized CLI invocation was used for the probe.
Actual outcomes and source IDs will be appended after qualification.
