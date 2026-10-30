# Learning diagnosis preregistration

Starting tested source ceb52450a3d6fcb074d14ccdae5028b68117ce68; existing uncommitted
style guidance retained and committed separately before executable changes. Historical
policy code, World and consumed sealed benchmark remain unchanged. Upstream exact pin:
8567836fb2c8c915d442b7c2fbedb4fd9b30396d, imported paths verified remotely.

## Confirmed observations before new training

Input is 29 tokens of current local tiles, energy, radius, recent consequence and
readout marker. No observation history, recurrent state or earlier coordinates enter
forward(). Optimizer/experience persistence is not policy context memory. Immediate
resource presence is observable at the center; history is not required to choose
interaction there. Broader navigation may need history; its causal importance is untested.
All seven actions legal; softmax sampling uses owned Python RNG, including evaluation.
No greedy evaluation or privileged wall/resource action mask is used.

Loss is minus selected log-probability times discounted actual energy_delta/15 minus
timestep EMA baseline. Gamma .97, baseline .9 decay, lr .0005, clipping 1. Baseline
updates after gradient. Code sign is conventional; controlled real neural sign/ownership
tests remain required. Mean per-episode loss weights variable episode lengths; it is
not automatically evidence of an implementation defect.

Energy-change reward telescopes: any exhaustion episode's undiscounted return is
-40/15 regardless of resource count. This is a confirmed objective degeneracy; gamma
makes discounted trajectories distinguishable, so it is not proof of unique failure
cause. Training action records are available; historical probabilities, entropy and
advantages were NOT recorded. Initial/final checkpoint measurements are endpoint
probes on original TRAINING observations only, not reconstructed historical quantities.
Parameter changes and moments show updates, not useful representational acquisition.

## Frozen diagnostic sequence and budget

One Kaggle T4 job, pinned Torch 2.6.0+cu124 / NumPy 2.1.3. Maximum 1,200 seconds
execution, 1,800 seconds platform cap including setup. No sealed-world files read,
no original sealed rollout, no capacity change or new neural architecture. Read only
hash-verified original initial/final checkpoints and training interaction history.
Probe up to 256 training observations per initialization with initial/final seven-action
probabilities/entropy; record named parameter differences and optimizer membership.
Controlled positive-advantage test uses fresh matching model, seven-action distribution,
selected action 5 and a single actual AdamW update; selected log probability must rise.
Fail on nonfinite values or optimizer ownership mismatch before canaries.

Canaries use actual MiniGPT and the SAME PolicyNetwork (118,023 parameters), same
AdamW lr, baseline, clipping and discounted policy-gradient objective. Only legal
action logits 0/1 are normalized in two-action tasks; other five head rows remain unused.
This task-interface mask adds no head/capacity and supplies no correct action label.
Fresh initialization declared; no claim of historical learned knowledge continuity.
One persistent owner per task/initialization across all its training episodes.

Immediate cue: seeded binary cue is resource/hazard center tile in the existing
observation encoder. Actions 0/1 have terminal rewards +1/-1 based on hidden cue rule;
policy samples its own actions and receives ONLY consequences. No teacher action.
Delayed cue: 3 actions, no intermediate reward, terminal +/-1 for a cue-conditioned
alternating sequence. Current cue and stage (recent-consequence token) are observable;
no oracle success flag until terminal. This isolates simple delayed credit with sufficient
sensory state, not memory or navigation. Separate diagnostic worlds; CellWorld unchanged.

Each task: initializations 42/43/44, 1,024 training episodes on seeds 70000..71023;
512 held-out evaluation episodes on 80000..80511 per arm. Paired frozen/random/trained
controls, identical per-evaluation action RNG. Frozen checkpoints immutable; no eval
updates or parameter changes permitted. Record every distribution, action, return,
advantage, gradient norm, feedback, owner experience/world/RNG and initial/final checkpoints.
Same limits for delayed task; at most 6,144 canary updates plus one sign-test update.
No search, favorable checkpoint selection or extra seeds.

Per-initialization canary PASS requires >=90% held-out success, last training quarter
success > first quarter by .1, and paired conditional-on-initialization 95% normal
interval lower bound >.15 over BOTH random/frozen (512 paired worlds). Task PASS needs
all three initializations. These canary intervals do not establish CellWorld transfer.
If immediate FAIL: stop, no delayed canary, no navigation repair. If delayed FAIL:
stop before navigation; diagnose credit/representation rather than hide it with curriculum.
If both PASS: CANARY PASS, L.1 STILL FAIL; any original-world intervention needs a new
preregistered controlled protocol justified by these diagnostics, not consumed sealed data.
This job never executes an original-world repair. No L.2 or second cell.
