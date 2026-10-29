# Active L.1 integration update

Current protocol: [L1_QUALIFICATION.md](L1_QUALIFICATION.md), l1-reinforce-v1.
Upstream source remains unchanged at the revision recorded below. New PolicyNetwork
wraps actual MiniGPT with a separately initialized Linear(vocab_size,7) head over the
public final-token logit vector. No hidden traversal copy, language-head policy claim
or historical weight continuity. PolicyCell uses upstream LearningState's AdamW and
full trusted checkpoint APIs; concept-specific step() is deliberately not used.
Its episode buffer, baseline, policy RNG, update history, recent experience and driver
world snapshot are owned external state. The earlier consequence adapter stays historical.

Remote admission tests deterministic initialization, finite seven-action output,
backbone/head gradients, nonzero parameter updates, and exact fresh-process full-state
continuation with pending experience. Evaluation checks weights/optimizer/update count
immutability. Source/module-path verification runs before model creation. Exact pins and
results are recorded in L1_QUALIFICATION.md. Executed Kaggle smoke v2 and the completed/recovered multi-seed trial verify actual
imports, finite gradients, 256 updates per student and exact full-state fresh-process
continuation on trained students. Neural engineering works; behavioral qualification
FAILS H1/H2. See L1_QUALIFICATION.md for raw lineage, runtime pins, failures and costs. No upstream source edits.

The following initialization findings and unexecuted pilot limitations are preserved
as historical context; current admission/status is L1_QUALIFICATION.md.

# Verified neural integration findings

Inspected upstream: ../adaptive-mini-gpt, clean HEAD
`8567836fb2c8c915d442b7c2fbedb4fd9b30396d`. No upstream edits.
Read AGENTS.md, README, architecture, CANONICAL_LEARNING, A.12 results, source model,
LearningState, checkpoint publication/load and B.0 training/replay driver interfaces.
A.13/B.0 scripts are present; docs/STAGE_A13.md and docs/STAGE_B0.md are absent and no
matching completed report was found in runs. Do not infer execution or success.

Latest documented completed A.12: 867 remote regression passes, learning FAIL.
Baseline MP-2 mean 62.50%, EB both 43.75%; none qualified stable entity binding.
Sealed A.11 suite remained untouched. Historical acquisition/language results do not
qualify broad continual intelligence. These are upstream reported results, not rerun here.

The active canonical concept path uses TextStudentM103 / IntegratedStudentM104_E and
DynamicEvolvableBackbone.hidden_states, with state CE and attribute BCE. Its LearningState
step(tokens, actions, tags, states=None, attribute_ids=...) is concept-specific, so it
cannot directly consume CellWorld transitions. Existing concept/attribute heads are not
qualified action policies. Legacy memory/replay in external state is caller-declared,
not automatic experience learning. B.0's standalone replay list does not itself prove
full persistent replay ownership.

Verified production backbone: model/transformer.py MiniGPT(ModelConfig), forward(idx,
targets=None, ...) -> (logits, loss). Supplied targets are already aligned next-token
targets; no double shifting. ModelConfig supports configurable dimensions/position modes.
Ontogeny imports this implementation, never copies it. The chosen small model is a NEW
fresh lineage, explicitly no historical language/concept knowledge. Migrating a historical
student into this substrate would need a separate controlled decision and retention audit.

Decision 0001 chooses action-conditioned consequence modeling. Encoding-v1 maps ordered
local tiles, radius, exact bounded energy, recent consequence and candidate action into
token IDs <256, length 13 or 29. Energy >127 rejects: a versioned experiment envelope,
not a permanent representation ceiling. First 24 output logits represent energy deltas
-8..15. Softmax expected delta supplies preferences; fixed epsilon 0.15 explores.
Single CE target is observed delta, not a hand-coded action label. This reinterprets an
existing LM output slice and remains behaviorally unqualified.

training/learning_state.py LearningState(model, lr, weight_decay, external, metadata)
registers AdamW including frozen tensors, excludes embeddings/1-D parameters from decay,
and owns optimizer/scheduler/model/external/metadata. snapshot/restore preserves aliases,
gradients and Python/NumPy/Torch/CUDA RNG. save(path) atomically publishes immutable trusted
full-object snapshots, load(path) reconstructs same-code owner. Cross-version pickle and
untrusted checkpoints are unsupported. Ontogeny reuses this owner, but explicitly updates
its optimizer with the new CE objective rather than calling concept-specific step().
No changes to upstream are required for the pilot. Replay is a bounded retained history,
not an additional learning mechanism; policy RNG and world snapshot live in external.

Local tests cover observable encoding, source hash utility, remote guard and fake learner
continuity only. They do not test real imports, forward compatibility, optimizer ownership
or neural persistence. Those are NOT VERIFIED. Remote pilot is implemented but NOT RUN;
first admit it with full-state fresh-process next-update equivalence and imported-module
path audit. The pilot driver does not yet implement sealed qualification, all independent
controls/variations or full resource accounting; it cannot award a scientific PASS.

Remote preparation pins both source revisions; verify clean upstream SHA before imports,
check actual imported module file is under that checkout, and record checkpoint hashes,
parent/source/environment/config identities, runtime versions, RNG, interactions, update
counts, actual GPU time and final completion status. The source check is implemented;
remote execution of import identity checks and continuation admission remain blockers. No submission is
made until these gates and evaluator are complete. No historical checkpoint dependency
was verified, and none is silently substituted.

Local dependencies are locked with uv.lock. Upstream declares torch>=2.4, numpy>=1.26,
pyyaml>=6. Remote launcher currently installs upstream declared ranges; runtime pinning
and image/version capture must be completed before reproducibility qualification.
Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python. Kaggle CUDA is mandatory; no fallback.
Commit/push executable source, render kaggle/prepare.py from exact SHA, submit private job,
wait for COMPLETE, retrieve all output and failures, then independently audit gates.
