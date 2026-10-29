# L.1 qualification — preregistration and execution record

Protocol **l1-reinforce-v1**, frozen before neural execution. Starting Ontogeny
b0caa6e91d6a2ec44f7b27d3f9a83be5a1524648; upstream unchanged at
8567836fb2c8c915d442b7c2fbedb4fd9b30396d. The execution commit is recorded by the
launcher and runtime.json. No historical checkpoint knowledge is claimed.

## Method and integration

The unexecuted immediate-consequence-v1 proposal remains historical, superseded here
before any results. This is one REINFORCE objective, no world-model auxiliary objective,
no replay training, entropy bonus, planner, expert targets, growth or extra cell.
Model: actual upstream MiniGPT, 2 layers, width 64, 4 heads, FFN 128, context/vocabulary
256, dropout 0. A freshly initialized Linear(256,7) head reads its final-token logits.
This uses the verified forward interface without copying a Transformer or mistaking
language logits for a qualified policy. All backbone and head parameters train.

Encoding is fixed 29 tokens: 5x5 relative tiles, unobserved cells masked to token zero,
radius, energy, consequence and constant readout token. Normal sensing remains 3x3;
inspection reveals 5x5. No global coordinates, seed, full map, future state, heuristic
action or evaluator labels enter the policy. All 7 actions remain legal: no privileged
wall/resource mask. Invalid output shape/nonfinite probabilities fail closed.

Actions sample softmax with an owned Python RNG. After each episode, discounted actual
energy_delta/15 returns (gamma .97) minus an owned timestep-wise EMA baseline (.9 decay)
weight the chosen action log probabilities. One mean policy-gradient objective, AdamW
lr .0005, no weight decay, gradient clipping 1.0. Nonfinite or absent backbone/head
gradients abort. Baseline is updated after the gradient step, never with evaluator data.

## Frozen benchmark, controls and budget

CellWorld v1 dynamics and all L.0 files are unchanged. Initial benchmark is defaults.
Held-out variation changes hazards 12->18 and relocation interval 20->10, leaving resource
count and sensing intact. It is harder but the original heuristic remains feasible.

3 initializations: 42,43,44. Training worlds 10000..10255, one encounter each per init;
256 updates maximum, <=25,600 interactions per init. Development 20000..20015; sealed
30000..30063. Both evaluated under default and hazard_shift. Control action RNG is
model_seed*100000+world_seed and matched between frozen and trained policies. Policies
get neither number. Five arms: rest, random, heuristic, frozen initial neural, final
trained neural. 2,400 total evaluation episodes (3*5*2*80). Final checkpoint is always
update 256, never selected on evaluation. All dev and sealed outcomes are collected
without tuning or checkpoint choice. No later stages begin automatically.

One smoke job then one experiment job, each hard subprocess timeout 4200 s; experiment
cooperative cap 3600 s. No CPU fallback. GPU T4 requested. Torch 2.6.0+cu124, NumPy 2.1.3,
PyYAML 6.0.2, pytest 8.3.5 pinned; Python/CUDA/driver/GPU and full pip freeze captured.
Actual Kaggle runtime is recorded rather than assumed cross-image reproducible.

Store every observed training transition; each checkpoint retains pending experience
(up to 100), previous episode, EMA baseline, update/forward/gradient counters, model,
optimizer, gradients, global and policy RNG and driver world/RNG/last observation.
Full files are immutable upstream LearningState trusted same-code checkpoints.
Snapshots are privileged driver state; policy consumes only observations/experiences.
Save initial, each 64 episodes and final with parent hash. No replay objective.

## Independent hypotheses and decisions

H1 PASS requires nonzero real parameter changes and first-to-last training-quarter
improvement in both mean duration and resource collection for each initialization.
Report gradient norms and separate head/backbone gradient flow. Improvement alone
without updated parameters fails.

H2 PASS requires positive lower95 paired mean differences over random AND frozen for
duration, collection AND collection/action under BOTH world variants. Compute paired
world differences per initialization, then Student t95 interval over the three
initialization means (df=2, critical 4.3026527299). This conservative interval reflects
limited initialization replication, not thousands of independent trained models.
Rest and heuristic remain strong references: report all differences and ceiling cases,
never claim heuristic superiority or broad intelligence merely from beating random.
This prospectively replaces the unexecuted earlier pilot's heuristic-beating gate in
accordance with the L.1 mission's H2 definition; no neural outcomes informed the revision.

H3 PASS requires fresh-process exact equality of the next action distribution, sampled
action, transition, complete model/optimizer, gradients, experience/baseline/history,
counters and all RNG states following the next neural update. Smoke saves with 9
pending experiences and one step remaining; its child process reloads and continues.
Admit full experiment only after smoke PASS. Experiment repeats admission itself.
Additionally repeat fresh-process next-update equivalence on each immutable trained final
checkpoint, using 10 new audit interactions and one discarded audit update per init
(257 total updates per training-plus-audit lineage). Audit worlds are not evaluation
worlds. Never promote the audit copy or count it as extra trained-policy evaluation.

Overall PASS iff H1/H2/H3 pass; failed engineering admission = engineering FAIL and stop.
A finished run failing H1/H2 = Learning FAIL. Incomplete/no remote evidence = INCONCLUSIVE.
No gate changes or tuning on sealed outcomes. Preserve every negative outcome.

## Measurements and execution record

Full per-episode duration, resources, hazard encounters, termination, energy trajectory,
action sequence, resource/action and sensory tiles are logged. Actual forward-example
and gradient-example counts, updates, wall time and package/runtime metadata accompany
checkpoints. The learning arm has extra update cost; do not claim compute matching.
All neural evaluation runs without learn(), checks immutable weights/optimizer/update
counts, and uses separate restored checkpoint owners and per-world RNG. Its changes
are discarded. No evaluation state influences training.

Status at preregistration: IMPLEMENTED / LOCAL CHECKS pending / REMOTE NOT SUBMITTED.
Actual completion status, traces, kernel IDs and measured results will be appended.

### Pre-admission assertion correction

While smoke v1 was running, static review found that loading the resumed owner before
capturing expected global RNG could make RNG equality vacuous. Capture a deep snapshot
before subprocess execution/load, compare immutable external/metadata, and audit trained
finals too. Smoke v1 output is preserved but cannot qualify admission. Rerun required;
no policy objective, environment, learning budget or sealed-world outcome informed this fix.

### Smoke v1 actual negative result

Kernel sparklinghydraulics0/ontogeny-l1-smoke version 1, source f7822a9, ERROR.
Actual forward/action/transition and one neural update executed on Kaggle, but
full-state comparison asserted false. Comparator treated distinct random.Random
objects by identity rather than RNG state. Correct to getstate() equality; the
independent immutable RNG snapshot correction is retained too. No learning experiment
or sealed evaluation was started. Raw trace/checkpoints/logs preserved under
runs/l1-smoke-v1/output. This is a failed admission, not a neural learning result.

### Smoke v2 actual admission result

Kernel sparklinghydraulics0/ontogeny-l1-smoke version 2 COMPLETE at source
6c63f35 (full revision in runtime record), Tesla T4 x2 allocated, CUDA:0 used,
Python 3.13.15, Torch 2.6.0+cu124, CUDA 12.4, NumPy 2.1.3.
Neural smoke wall time 6.667 seconds (excludes environment setup).
Initialization exact; finite action distribution and genuine gradient update executed.
Fresh-process action distribution/transition and all 8 state groups exactly match.
Full experiment is now admitted; H1/H2 are still NOT RUN. Versioned small evidence:
experiments/l1_single_cell/smoke-v2; raw checkpoints/logs: runs/l1-smoke-v2/output.

### Full run v1 interruption and bounded recovery

Kernel sparklinghydraulics0/ontogeny-l1-experiment version 1, source adb97ff,
ERROR after 193.175 seconds of execution (plus setup). All 768 training episodes and
256 updates per initialization completed. Final checkpoints for all 3 are preserved.
1,600 of 2,400 evaluation episodes completed (initializations 42/43); their trained
checkpoint fresh-process continuation audits PASS. The 44 audit fixture exhausted
energy before its assumed tenth step; the harness tried stepping a terminated world.
This is an audit-fixture failure, not a neural-gradient failure or a complete H2 result.
No scientific outcome metrics were inspected to select a repair.

Recovery changes only audit energy capacity/start to 100, provably greater than the
worst 10-step cost (80), within the unchanged observation encoding. Training/evaluation
CellWorld remains defaults/declared hazard variant. Mount immutable experiment v1
outputs, verify EVERY input file against a committed hash manifest before unpickling,
retain all trained models and the 1,600 original evaluation rows, complete only the
44 continuation audit and its missing 800 evaluation episodes. No retraining, no new
model/checkpoint selection, no policy/hyperparameter/gate changes. Recovery is bounded
at 1,800 s (Kaggle cap 4,200 s including installation); one discarded audit update only.
Original failed fixture remains in a separate artifact directory. Historical main
revisions are now accepted for reproducible job preparation when published in origin/main.

Kaggle rejected the version-qualified kernel mount but accepted the job without it.
Recovery v1 (ontogeny-l1-recovery) is therefore an infrastructure failure, no neural
result. Recovery now uses a private dedicated version-1 dataset of allowlisted generated
artifacts only (no repository or credentials), identified in recovery-input.json by
archive SHA-256 and then by every extracted file's committed hash. Loader refuses
missing/wrong input. No new training is authorized by this infrastructure repair.
