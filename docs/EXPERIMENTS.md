# Append-only experimental history

## 2026-10-29 — Initialization / L.0 v1

PLANNED: deterministic finite world, nonlearning controls, integration contracts; no
local neural execution. Upstream clean source inspected at
8567836fb2c8c915d442b7c2fbedb4fd9b30396d. Existing Ontogeny LICENSE preserved.

EXECUTED locally: 31 explicitly inspected environment/guard/encoding/fake-policy tests;
Ruff checks/format; Pyright; ten development seeds per control, 20 episodes total.
PASSED ENGINEERING CHECKS: all tests; final lint and type checks clean.
L.0 qualification: PASS for the declared v1 engineering invariants and controls.
Raw baseline rows are versioned in experiments/l0_cellworld/baselines.json.

Random mean duration 39.3, collected 0.6, hazard exposures 0.9, final energy 0.
Fixed heuristic mean duration 100, collected 14.5, hazard exposures 0, final energy 40.1.
These development results are not a neural or sealed scientific qualification.

Failed setup/check attempts preserved here: initial uv sync could not write default
cache; temporary writable cache resolved that. Sandboxed PyPI DNS then failed;
authorized dependency installation succeeded. Initial lint found default-constructor,
import-order and long-line issues; corrected. Initial Pyright found missing remote-only
packages, interpreter selection and frozen-field test assignment; remote imports now
lazy, local environment explicit, tests corrected. No failed neural runs occurred.

## 2026-10-29 — L.1 immediate-consequence-v1

PLANNED / PILOT IMPLEMENTED / NOT EXECUTED. One CE objective over real action energy
outcomes, new fresh model lineage; no historical knowledge continuity claimed.
No Kaggle submission or kernel/session result exists. NOT PASSED LEARNING QUALIFICATION.
Admission requires remote fresh-process next-update equivalence, independently audited
evaluator/controls/variations and runtime provenance. See NEURAL_INTEGRATION.md.
L.2–L.5 NOT STARTED. No checkpoint promoted and no upstream edits.

## 2026-10-29 — L.1 REINFORCE v1 preregistration

Supersedes the unexecuted immediate-consequence proposal before any neural outcomes.
Existing CellWorld is frozen. New task-specific action head on actual MiniGPT; one
REINFORCE objective. Kaggle/GitHub access verified, upstream remains clean/pinned.
Protocol, H1/H2/H3 and resource budgets frozen in L1_QUALIFICATION.md. No L.2.

### L.1 remote admission execution

Smoke v1 f7822a9: FAILED_ADMISSION, RNG comparator identity bug, raw outputs preserved.
Smoke v2 6c63f35: COMPLETE / ENGINEERING PASS, exact fresh-process full-state
continuation and genuine neural update on Kaggle T4. No learning qualification implied.
See L1_QUALIFICATION.md and experiments/l1_single_cell/smoke-v1, smoke-v2.

## 2026-10-30 — L.1 actual scientific outcome

EXECUTED: 3 initializations, 768 training episodes, 26,618 interactions and 768 real
training updates. 2,400 evaluation episodes, five controls/policies, two conditions.
H1 FAIL (initialization 44 regresses); H2 FAIL (all paired intervals cross zero,
resource-efficiency mean below random/frozen); H3 PASS (trained full-state exact
fresh-process continuation). Recommendation: L.1 Learning FAIL, no promotion/no L.2.

Original main run adb97ff ERROR at 44 audit after all training and 1,600 evaluations.
Earlier audit did not ensure survival to its assumed horizon. Safe audit energy 100
is used ONLY in continuation testing, no L.0/training/evaluation dynamics changes.
Input-less recovery 2cb499a failed because Kaggle rejected the version-qualified mount.
Private allowlisted hash-pinned dataset recovery 6a0cf71 COMPLETE adds only missing
44 audit/evaluation. One submission was rejected by a transient two-session limit;
retried after the task-owned input-less job errored. No additional training or tuning.

Independent local artifact audit caught missing 44 gradient-history JSON export;
full history is in the final checkpoint. Separate records-only GPU extraction validates
and exports it without forward/update. A preparation attempt also refused the untracked
result directory until the measured scientific outputs were committed; no job ran from
an uncommitted source. Raw failures and traces are preserved, not counted as passes.

Curves, all comparison means, intervals, checkpoint hashes and continuation groups:
experiments/l1_single_cell/results. Full chronological evidence and reproduction:
docs/L1_QUALIFICATION.md. LOCAL: non-neural tests/static/artifact analysis only.
REMOTE: Kaggle CUDA training/inference/checkpoint loading; no Colab or local fallback.

Final records extraction 7f4ba6d COMPLETE, 1.921 s, zero forward/update. Confirmed all
3 checkpoints contain 256 real gradient records and 118,023 parameters (1,799 head).
Full artifact audit PASS: 26,618 training and 111,316 evaluation interactions;
287,242 training / 1,089,140 evaluation sensory tiles. Seven additional discarded
admission/audit updates across attempts are counted separately (775 total updates).
Final local checks: 34 non-neural tests, Ruff, Pyright and syntax PASS. Scientific
verdict remains H1 FAIL, H2 FAIL, H3 PASS => L.1 FAIL. No checkpoint promoted.

Final local inspection found ../adaptive-mini-gpt HEAD at
5bd7b39b3d0b7df7566f136b723eb6fc8669614d with a clean working tree, advanced from the
starting revision during this session. Ontogeny made no upstream edits or reset.
Every actual GPU job cloned/verified 8567836fb2c8c915d442b7c2fbedb4fd9b30396d;
the newer local checkout was not used or claimed compatible. Default source verification
correctly rejects it until a separately qualified integration revision is adopted.

## Learning failure diagnosis — executed

Execution source c09f0cb5e1d6fb5a050a977020d9b65f06580ed0; upstream exact8567836fb2c8c915d442b7c2fbedb4fd9b30396d.
Kaggle sparklinghydraulics0/ontogeny-learning-diagnosis version1 COMPLETE, 155.434 s.
Positive-advantage sign/ownership test PASS. Original checkpoint endpoints probed on
training observations only; 31/31 parameter tensors changed per initialization.
Immediate cue canary PASS: all three 100% held-out. Delayed canary FAIL:15.23/15.63/18.95%.
No repair or original-world rollout after failed delayed prerequisite. L.1 STILL FAIL.
6,145 total updates,12,288 canary training and18,432 evaluation interactions. New
canary checkpoint reload verified; fresh-process next-update equivalence NOT RUN.
Non-neural arithmetic/coverage/artifact audit PASS; curves visually checked. Download
DNS failure retried; offline plot-cache failure resolved by installing pinned plotting
dependency. All raw data and negative outcomes retained. No upstream/code-capacity/world
changes, no L.2. Details:LEARNING_DIAGNOSIS.md; defects:RESEARCH_DEFECTS.md.

## Common execution foundation — engineering qualification

Source f8c4bc88cf53ae8b1b82d37fa33fad7a212eeedf; upstream exact8567836fb2c8c915d442b7c2fbedb4fd9b30396d.
Kaggle sparklinghydraulics0/ontogeny-exec-2fc67f1e54ff4758 version1 COMPLETE.
Request digest2fc67f1e54ff475818e1b109e1783017ae8a34d7a10c8eeeaabfa8d7bb328b19.
Common worker SUCCEEDED; scientific verdict NOT_RUN. Full fresh-process continuation
and independent unchanged historical runner comparisons exact; 3 physical audit/update
steps,21 interactions. Candidate state1 update/10 interactions. All21 bundle files
recovered/verified through new adapter. Quota reports3.06h->3.00h;2.5h reserve retained.
One visible CUDA device, two physical T4s, full runtime inventory captured. No migration
of old two-visible-device checkpoints, no Colab, routing, new scientific trial or L.2.
Original sensory metric122 candidate tiles later clarified with a physical total253
from raw traces; local regression test covers that accounting-only addition. Historical
remote outputs remain unchanged. All72 local-safe tests/static checks pass. Initial
local build fetch failed under restricted DNS, then authorized packaging succeeded.
Details and exact limitations: EXECUTION_QUALIFICATION.md. Evidence: experiments/execution_foundation.

## Colab readiness and fresh-job routing

Source 2a005b43900a93b49eda91f7398ace0b3d17a684; unchanged upstream 8567836fb2c8c915d442b7c2fbedb4fd9b30396d.
One readiness-only T4 probe succeeded/released, then one pinned common-worker Colab
trial SUCCEEDED/science NOT_RUN. Actual proof rather than HTTP 503 controls selection.
One assignment required a second readiness probe; no second create. Common-worker
state/reference equivalence exact; 21 indexed files verified before release. Cleanup
DNS failure required reconciliation; owned session confirmed stopped, no retraining.
An initial reconciliation import error was corrected and regression-tested; automatic
cleanup-retry reconciliation subsequently hardened locally. 86 local-safe tests pass.
Kaggle fallback mocked; no live outage forced. No credentials committed, local neural
execution, source/capacity/learning-objective change or old checkpoint migration.
Details: COLAB_ROUTING.md; safe records: experiments/colab_execution. L.1 remains FAIL.

## Checkpoint transport and interruption qualification

Worker a88d7d386eb502e4ffae5bde7dc504441824c76d; recovery transport/protocol
74bc0e87f428748fe358cd6a6418d8e5d7f6dc15; unchanged upstream
8567836fb2c8c915d442b7c2fbedb4fd9b30396d. Overall INCONCLUSIVE: Colab→Colab,
Kaggle→Kaggle and Colab→Kaggle fixture pairs PASS for upload, exact next-action/
transition/update and zero-update evaluation with byte-identical retained checkpoint.
Kaggle→Colab NOT RUN after a single precondition allocation refusal; no assignment,
retry or provider substitution. Both parents passed SIGKILL-after-publication recovery
and exact fresh-process/historical-reference continuation.

Original Kaggle evaluation exceeded its 300 s platform cap during setup; preserved as
infrastructure FAIL. Explicitly amended recovery batch (600 s cap, unchanged worker and
requests, per-child state gates) completed all three remaining Kaggle operations. Total
reserved Kaggle ceiling stayed 1,500 s across four jobs. First batch download hit DNS
failure; a separate read-only collection recovered the completed job without GPU rerun.
Completed work: eight bundles, nine updates, 105 interactions, 1,217 observation tiles,
38.435 s task time; provider quota 3.00→2.74 h, retaining 2.5 h reserve. Failed attempt
partial neural counters are unknown, not silently zero. 93 inspected local-safe tests
and static checks PASS. All neural execution stayed on remote verified CUDA.

Only one initialization and a pending-before-first-update input were tested; nonempty
input optimizer moments, long rollouts, reverse portability and arbitrary VM-loss/
periodic off-VM-ACK recovery remain unqualified. Automatic checkpoint routing stays
blocked. L.1 remains Learning FAIL; no later stage. Details: CHECKPOINT_QUALIFICATION.md;
selected evidence: experiments/checkpoint_transport; complete raw runs under ignored
runs/checkpoint-transport. Failures and original preregistration preserved.

## Internal engine consolidation

At source a3a26ba630b283333f352b7779ce85b4173d3982, Kaggle job
sparklinghydraulics0/ontogeny-neural-consolidation-a3a26ba630/1 completed: six remote
neural checks passed, independent token training passed and the organism fixture passed
with exact continuation. The checks compared all three position representations against
the historical source and converted a trained checkpoint with existing optimizer state.
The organism bundle's 31 files were recovered and verified locally after bulk-download
DNS failures. One-time reference code is confined to the research experiment; it is not
an active Mini-GPT dependency. A subsequent static AST comparison found no change in the
neural model/optimizer/state operations during archival-boundary cleanup. Detailed scope:
MIGRATION.md. L.1 remains Learning FAIL; no developmental promotion.

## Initial ten-million-parameter student

At source 05ff24b082933b08d7d47545a06cafd281279a94, Kaggle job
sparklinghydraulics0/ontogeny-student-fresh-05ff24b082/1 completed a separately
preregistered fresh independent student: 10,054,016 parameters, two updates, 128 training
input tokens and one discarded audit update. Complete fresh-process continuation matched
all nine state groups. Two full checkpoints and six other files were recovered and
hash-verified. Peak Torch allocation565,831,168 bytes, task wall13.99 seconds; quota
2.67→2.61 h with a 2.5 h reserve. Held-out loss4.24892 on40 tokens is a diagnostic,
not a learning claim. Science NOT_RUN; L.1 remains FAIL.

The first Colab assignment failed after an execution WebSocket closed. Its possible
partial work is unknown; no archive was recovered and its owned session was stopped.
The Kaggle run was an explicitly recorded new fresh lineage, not a claim of continuation.
Full protocol, actual costs, failure, lineage and checkpoint paths:
experiments/neural/initial_student/REPORT.md. Old reports and learner states are preserved.
