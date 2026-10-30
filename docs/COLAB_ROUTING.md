# Readiness-based provider routing

Status: implemented; 86 local-safe tests pass; common-worker Colab GPU engineering PASS.
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
   require ownership-aware manual reconciliation and block blind fallback/new creation.
   The current reconcile command recovers completed Colab bundles; it cannot resolve
   an allocation timeout that never recorded a usable assignment. Cleanup uncertainty
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
exact upstream 8567836fb2c8c915d442b7c2fbedb4fd9b30396d. Worker 120 s, 64 MiB output,
128 MiB Torch-allocated memory. Allow <=1,050 seconds for pinned setup/execution and
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
Actual outcomes and source IDs follow below.

## Actual result

**Colab common-worker engineering PASS, verified export and confirmed cleanup.**
Source `2a005b43900a93b49eda91f7398ace0b3d17a684`; request digest
`ccab269b026e1330ca3a0c00546a68aae0ad537b7695203b7e7fb0f3e286740f`.
Installed Google Colab CLI 0.7.5. The real allocation selected a Tesla T4, one physical/
visible device, Python 3.13.16, driver 580.82.07, Torch 2.6.0+cu124 and CUDA12.4.

The first readiness proof download was absent; a second probe of the SAME assignment
confirmed remote CUDA. No second create. The worker's 8.808 s task completed 3 physical
updates and 21 interactions, exact fresh-process continuation in all nine state groups,
and exact historical-runner comparison. All 21 required/indexed files recovered and
verified before shutdown. Candidate logical state 1 update/10 interactions; physical
sensory cost 253 tiles. No scientific qualification or old checkpoint migration claimed.

Initial release hit a transient DNS failure. The bounded retry actually stopped the
owned session, but the original routing record remained UNKNOWN. A new reconcile command
confirmed server/session absence and reverified the already-exported bundle without
allocation or training; state is now COMPLETE/cleanup confirmed. The failed initial
cleanup reason remains in the evidence as historical context. The first reconciliation
helper had a missing import, caught by local tests and its first invocation; corrected
before successful reconciliation. No repeat neural trial was used to hide those errors.

Current code also automatically accepts a verified completed bundle after a successful
bounded cleanup retry. That post-run controller hardening, submitted-job concurrency
protection and strict cleanup-return checks are covered by local regression tests,
not an extra GPU trial. Unresolved cleanup remains UNKNOWN and blocks a new allocation.

86 local-safe tests, Ruff check/format, Pyright and syntax checks PASS. Real Colab READY
and worker execution are tested. Kaggle availability fallback is mock-tested using the
previously GPU-qualified Kaggle adapter; a live Colab outage was not induced and no
Kaggle quota was spent in this increment. Cross-provider next-update exactness remains
unqualified despite the canonical single-visible-device configuration.

Selected safe evidence: experiments/colab_execution. Complete raw result:
runs/colab-worker/attempt/download/result. CLI session stores/raw CLI traces are private
under ignored runs/ and must NOT be committed or included in exported model artifacts.
The evidence directory holds selected records, not all binary checkpoint files indexed
by artifacts.json. The initial separate readiness-only test also stopped its session.

## Use and next work

```bash
# Request source must identify published code; use a new work directory.
ontogeny-exec run --request request.json --inputs inputs/ --directory runs/new-job --ledger runs/controller --provider auto
# If a completed Colab trial's export/cleanup state needs reconciliation:
ontogeny-exec reconcile --request request.json --directory runs/new-job --ledger runs/controller
```

Colab success returns a verified receipt after release. Kaggle fallback returns its
asynchronous native submission record; use existing status/collect with the nested
`runs/controller/kaggle` ledger and returned native identity. `--provider colab` and
`--provider kaggle` give explicit choices without changing the scientific request.
Do not reuse a request with a new ledger as an automatic retry: that is a new explicit
execution, whose costs/outcomes must be retained. The same shared ledger is idempotent.

Immediate next work: qualify public checkpoint input transport/resume/evaluation on
both providers, then test cross-provider compatibility and interruption recovery from
a durable checkpoint. Fresh-job auto-selection does not yet mean safe mid-training
migration. Keep the matched learning diagnosis on a separate scientific protocol;
L.1 remains Learning FAIL and no L.2/two-cell work is admitted.

## Subsequent checkpoint transport qualification

Explicit-provider upload/resume/evaluation is now implemented and passes its declared
pending-first-update fixture on both providers. Colab→Kaggle exact continuation passes;
Kaggle→Colab was NOT RUN because Colab refused its next assignment. The controller
correctly handled a precondition refusal with no owned assignment, without depending
on HTTP 503, retrying creation or substituting another provider for the missing test.
See CHECKPOINT_QUALIFICATION.md for actual sources, cap/DNS failures, recovery, costs
and fixture limitations. Overall bidirectional portability remains INCONCLUSIVE.
`run --provider auto` still rejects checkpoint inputs; continuing students require an
explicit provider. No previously trained optimizer migration or arbitrary VM-loss
recovery was qualified. These results do not amend the historical fresh-worker report.
