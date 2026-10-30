# Unified neural foundation migration

Status: the audited engine at the migration test revision passed six remote neural checks,
independent token training and organism continuation. The 31-file organism bundle was subsequently recovered and verified locally;
subsequent removal of archival references from the active core and the 10M
pilot are separately documented. No scientific acquisition is claimed.
This is an engineering consolidation. It changes no historical scientific verdict.

## Audit and ancestry

Ontogeny starts at 670709ca2d319baf3073be0f7ce709f3f8d664d1. Source repository:
https://github.com/hieulc0/adaptive-mini-gpt, inspected clean revision
87e8928c440fecc4f3a6f6c5ea3c869a6a8537e6. The original sibling is read-only throughout.
Imported component hashes are in decisions/neural_source_inventory.json. Those seven
source files are byte-identical to the former execution pin
8567836fb2c8c915d442b7c2fbedb4fd9b30396d; later source research does not silently alter
this migration's starting numerical implementation. Apache-2.0 LICENSE and NOTICE retain
upstream attribution. Changed files and component ownership are explicit below.

| Class | Original component | Disposition |
|---|---|---|
| A: extract/refactor | model/config.py | mini_gpt/model/config.py; dense fields, validation, original parameter dimensions/names |
| A | model/attention.py | mini_gpt/model/attention.py; causal SDPA, absolute/RoPE/ALiBi and explicit mask intersection; no algorithm change |
| A | dense portions of model/transformer.py | mini_gpt/model/transformer.py; one hidden_states traversal and LM head, same parameter paths/initialization |
| A | training/optimizer_contract.py | mini_gpt/learning/optimizer.py; one shared AdamW grouping/membership contract |
| A | generic parts of training/learning_state.py | mini_gpt/learning/state.py; full joint graph/RNG, rollback and atomic immutable publication; generic backward/step replace concept-only step |
| C: simplify | training/remote_runtime.py, reproducibility.py | mini_gpt/runtime.py; explicit remote CUDA/determinism; experiment code owns seed/stream identity |
| C | trainer.py, checkpoint.py | narrow token trainer + one LearningState persistence implementation; no parallel CheckpointManager/tokenizer manifest framework |
| A/C | Ontogeny state comparison helpers | mini_gpt/checkpoint/comparison.py; shared by independent and organism qualification, no organism dependency |
| B: defer | LoRA/modular adapters, episodic retrieval/replay policy | potentially useful, but not needed by either migrated runtime; remain in source with original tests, not copied speculatively |
| D: omit | expert/depth/skip growth, milestone students, domain routers, capacity migration, skill registries, one-off benchmark families | absent from new core; not qualified by this consolidation |

The source concept path uses a different historical student/backbone and concept heads.
It is NOT silently substituted for this dense engine or claimed compatible. Its reports,
checkpoints and original loaders stay with the original repository at immutable revisions.
Generic external memory/replay graphs remain owned/persisted by LearningState; no unused
retrieval policy or new memory architecture is introduced.

## Boundaries and cleanup

mini_gpt has no imports from ontogeny or adaptive_mini_gpt. Ontogeny's policy head,
observation encoding, environmental feedback and driver remain organism-owned. Shared
optimizer construction, backward/clip/step and checkpoint ownership live in mini_gpt.
Independent token training needs no CellWorld. Concept heads can consume the same
hidden_states interface in separately qualified studies; no new concept solver is added.

Active provider launchers clone only Ontogeny and import the internal package. Request
schema2 declares source {ontogeny_sha, engine: mini_gpt}. Schema1 stays readable for old
artifact hashes but cannot execute in the new worker. There is no --upstream argument,
sys.path sibling injection or installed adaptive_mini_gpt alias in normal operation.

Removed obsolete executable paths: kaggle/{prepare,run_l1,recover_l1,diagnose_learning}.py
and scripts/prepare_diagnosis.py. Their exact historical source is retained at Ontogeny
670709ca2d319baf3073be0f7ce709f3f8d664d1. The unexecuted ConsequenceCell/candidate-action
encoding is also archived there. Existing reports, manifests, checkpoints and result
files are not renamed or deleted. Read-only historical artifact-analysis scripts remain.
The fixed-cohort qualification implementation moved from evaluation/l1.py to
experiments/organism/qualification.py, unchanged, separating a research protocol from
reusable runtime code. Token smoke fixtures live in experiments/neural/token_smoke/.

## Checkpoint contract and explicit conversion

New snapshots use format version2, with the same roots/gradients/RNG graph contract and
new class paths. State-dict parameter keys are retained. Normal load refuses old format
or missing old class paths; it never replaces missing weights or optimizer state.

experiments/integration/import_historical.py is an explicit, remote-only trusted
conversion boundary. It requires a declared SHA-256 and audited source revision, remaps
only five dense config/model/attention classes, validates that no adaptive routing,
selective-token hooks or structural skips are active, and restores the joint graph with
original optimizer moments, gradients, external experience, progress and all RNGs. It
normalizes config/unused attributes without reconstructing/reinitializing parameters.
An added checkpoint_imports record identifies immutable parent hash and format change.
It never overwrites the parent. Unknown historical neural classes fail closed and keep
their supported original-repository loading path.

Conversion alone does not authorize scientific promotion, overwrite source/protocol
metadata or automatically rebind an old execution lineage to a new training request.
Old execution_lineage metadata remains intact; a continuing study must explicitly
preregister its new code/protocol boundary. Mode C consolidation of independent research
weights into an organism is NOT implemented by this same-state format conversion.

## Frozen remote engineering verification protocol

Before submission, commit/publish exact source. One Kaggle T4 job, max720 platform
seconds, fresh quota admission retaining2.5 h reserve. Initial observed quota2.73 h.
No automatic retry or local fallback. Set CUDA_VISIBLE_DEVICES=0 and
CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python; Torch2.6.0+cu124, NumPy2.1.3. Record Python,
driver, GPU inventory, packages, sources, checkpoint hashes and raw pass/failure logs.

Budget: <=32 optimizer updates, <=64 environment interactions, <=120 seconds for each
subprocess, <=300 seconds GPU test phase, <=120 seconds common-worker fixture, <=256 MiB
Torch allocation observed, <=128 MiB complete output. No new learning campaign, final
worlds, benchmark tuning, capacity increase or checkpoint promotion. Source reference
clones exist only in this equivalence job, not in normal unified execution.

Required gates:

1. Three dense absolute/RoPE/ALiBi configurations: identical initial state keys/values,
   logits, aligned loss, gradients, optimizer state and first updated weights. Tolerance
   atol=0/rtol=0. Causal suffix perturbation and all-masked-target refusal.
2. Embedding/tied/no-decay and optimizer ownership; immutable save; model/parameter/
   optimizer aliases survive full restoration and rejected-candidate rollback.
3. Independent token student: two completed updates with nonempty Adam state/gradients,
   scheduler, dropout and RNGs; fresh-process third update matches all state exactly.
4. Real trained historical organism checkpoint (one existing update, not an empty
   optimizer): load it under original source in a separate process; explicitly convert
   a copy and compare preserved pre-state plus a new five-step episode/next update,
   actual probabilities/actions/consequences and complete state. Ignore only the added
   checkpoint_imports annotation, reporting it explicitly. Preserve parent bytes.
5. Independent CLI training smoke and the unchanged ten-step organism common-worker
   continuation/reference/interruption fixture under the internal engine. Save full
   states and results, no behavioral acquisition claim.

The historical input is selected by identity, not quality: the first completed Colab
public-resume checkpoint from the prior infrastructure fixture,
runs/checkpoint-transport/colab/colab-train/download/result/checkpoints/episode-1.pt.
Its frozen SHA-256 is
08a34400dc7aade6bc5e2601ce914b7635613f3d33e324f46ec5b68bdb2d5931,
size 2,105,076 bytes; these are fixed in the package manifest before submission. This is a new
discarded engineering branch, not a promotion or replacement of a developmental student.

Local-safe verification: configuration/import/source-ownership and request-schema tests,
existing environment/fake/hash tests, wheel contents, Ruff, Pyright and compilation.
Torch/NumPy may be installed locally for static typing; no neural construction,
forward/backward, GPU use or real checkpoint load may run locally. Tests with real neural
operations are guarded and selected only in the remote job.

Any GPU gate failure is engineering FAIL/PARTIAL with raw evidence preserved. A missing
GPU is NOT RUN, never inferred from local mocks. Numerical mismatches stop equivalence
claims; no tolerance relaxation after viewing results. Actual outcomes will be appended.


## Initial remote verification

Kaggle native job sparklinghydraulics0/ontogeny-neural-consolidation-a3a26ba630/1
finished COMPLETE. The result record reports six neural tests PASS, independent token
training PASS and organism training/continuation PASS; its organism bundle was verified
remotely with 31 files. The reference comparison included an actual checkpoint with
existing optimizer moments and an exact subsequent interaction/update. Source for that
run is the migration test revision recorded in its output. Two local bulk-download attempts failed during DNS resolution. Eight missing organism
files were then recovered individually; the complete 31-file bundle passed local hash
verification. The original partial files and private logs remain under runs/neural-consolidation/.
No GPU rerun was performed because download failed. Selected non-secret evidence is
in experiments/integration/results; neural test temporary artifacts are not all locally
recovered. The checkpoint input remains immutable.

Following the user's direction, archival source identities and the one-time historical
conversion tool are kept in experiments/integration and migration records, outside the
active mini_gpt package. Normal engine imports never fetch or select an external model.
A training run records its own source for reproducibility; that record is not a choice
of neural engine. The continuing learner's own checkpoint remains its learned state.
The later archival-boundary cleanup and 10M pilot need their own verification.


After the remote test, a local static AST comparison verified that model configuration,
attention, Transformer operations, optimizer grouping and LearningState operations are
unchanged by the archival-boundary cleanup (ignoring docstrings and replacing the old
imported format constant with its same value, 2). The active package no longer contains
a repository name, external revision or Git dependency. The independent CLI uses its
loaded internal engine and checkpoint contract; it does not require a Git checkout.
The 10M pilot separately tests the current source, not the old experimental engine.
