# Research protocol

Every experiment declares hypothesis, source revisions, initial checkpoint hashes,
world version/configuration, independent splits, seeds, controls, budget, fixed final
checkpoint, primary metric and stop gates before training. Preserve failed attempts.
No selection of favorable checkpoints after seeing development or sealed outcomes.

L.1 hypothesis: a single action-conditioned immediate-energy predictor updated by
cross-entropy on actual observed delta categories improves unseen-world viability over
its unchanged initial policy. This is a minimal model-learning mechanism, not a claim
of long-horizon planning. Default v1 outcomes fit integers [-8,15]. Changes outside
this envelope reject and require explicit encoding/objective migration.

The pilot spec uses model seeds 42/43/44, train worlds 1000..1007, dev 2000..2003,
sealed 3000..3007. Each arm has at most 800 updates and 800 training interactions;
one whole pilot has a 1800-second cooperative time cap plus Kaggle's external job cap.
Maximum model: one 32-wide block, two heads, 64-wide FFN, vocabulary/context 256,
dropout zero. Seven prediction queries per decision, one extra forward/backward per
learning action, gradient norm clipped to 1.0. Store at most 128 experiences in owner;
stream complete history to disk. Stored history is not replay training in this pilot.
These are current experiment limits, never architectural ceilings.

Random, fixed heuristic, frozen neural and learning neural controls must share world
seeds and final evaluation horizon; frozen and learning start from identical checkpoint
and policy RNG state. Report interaction, sensory, forward and gradient costs separately;
training and inference costs are deliberately different, never claim compute matching
without an additional update-cost control. Evaluate prediction MSE separately from
behavior, and record duration, exhaustion/truncation, collection, hazards, final energy,
resource efficiency (collection per action), inspections and tiles observed.

Admission gates: clean pinned sources/import identity, GPU-only preflight, finite loss/
gradients, full trusted checkpoint save/load, fresh-process next-update equivalence,
no reset between episodes and independent evaluator audit. Before sealed execution,
require positive paired dev duration improvement over frozen and random at all three
model seeds, no hazard regression, and prediction improvement over a constant-delta
control. Seal rules/variations and evaluator revision before use.

Final learning qualification: positive paired duration improvement over random, fixed
heuristic and frozen on unseen seeds and preregistered density/hazard-change variations,
with a paired 95% interval lower bound above zero across world/model pairs, no increased
hazard rate and complete cost/provenance audit. A ceiling-tied heuristic result is
INCONCLUSIVE, not a pass. This demanding gate may motivate a revised world under a new
protocol rather than retrospective threshold changes. No L.2 on a pilot result.

Evaluation never calls learn. Freeze final checkpoint; isolate evaluation RNG/state and
restore it before later training. Environment seed lists belong to driver, not policy.
Sealed outcomes cannot drive architecture or hyperparameters; a changed design requires
new untouched seeds. Seed names alone do not prove isolation: audit generation and logs.

Stop on leakage, nonfinite updates, budget exhaustion, ownership mismatch, checkpoint
mismatch or a failed admission gate. Keep raw logs, partial checkpoints and tracebacks;
report FAIL/INCONCLUSIVE/NOT RUN per gate. Roll back rejected candidates to immutable
parents. Ablate updates, initialization/history and observation costs before attributing
benefit to development. L.2 additionally needs history-matched reset controls and prior
competence gates; no silent optimizer resets or replacement specialists.
