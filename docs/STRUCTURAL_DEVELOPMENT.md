# Learning capability before structural development

The objective is to improve future learning while retaining useful knowledge within
explicit resource budgets. Biological development is an inspiration, not evidence that
adding neurons or selecting networks will produce an autonomous organism. Today's
Mini-GPT has persistent learning ownership; reliable concept acquisition remains unqualified.
The next diagnosis is [comparison acquisition](../experiments/neural/concept_acquisition/PROTOCOL.md).
The existing five-update student and unfinished rule-transfer work remain separate.

## Research assessment

[Dohare et al., Nature 2024](https://www.nature.com/articles/s41586-024-07711-7)
show that selective neuron replacement can mitigate plasticity loss in their continual
learning settings. Their replacements use fresh incoming and zero outgoing weights.
For y = v GELU(w x), v=0 initially blocks the incoming-weight gradient, but generally
not the outgoing-weight gradient. Subsequent incoming learning can therefore start.
Dormant input activation, symmetry, disconnected parameters or missing gradients can
still prevent learning. This does not establish retention or benefit in this Transformer.
B.0's weak initial acquisition is not evidence of plasticity lost through long training.

[Net2Net, Chen et al.](https://research.google/pubs/net2net-accelerating-learning-via-knowledge-transfer/)
provides function-preserving transformations for supported networks. Identical unit
copies with balanced outgoing contributions preserve the mathematical function; incoming
noise generally destroys exact preservation. Floating-point operation order, dropout,
normalization and tied parameters require explicit equivalence tests. Transformer FF
width is a narrower first intervention than changing residual width or attention topology.
Equal logits at expansion do not imply equal subsequent AdamW updates or generalization.

[RigL, Evci et al.](https://www.research.google/blog/improving-sparse-training-with-rigl/)
tests dynamic sparse connectivity under a fixed parameter budget. This supports a
fixed-capacity reorganization control before growth, not an assertion of continual
learning or autonomous intelligence. Sparse parameters alone do not guarantee lower
actual GPU latency or memory use.

[PyTorch optimizer loading](https://docs.pytorch.org/docs/2.6/generated/torch.optim.Optimizer.load_state_dict.html)
requires deliberate parameter correspondence. Separate deep copies normally create a
disconnected optimizer copy, rather than mutate parent parameters. Current LearningState
already jointly copies model/optimizer/scheduler/external state and checks membership.
This is useful existing engineering, not a qualified structural tournament or persistent
promotion transaction. Candidate exceptions propagate after rollback. Root handles cannot
be replaced, so arbitrary topology replacement is not admitted by that interface.

## Later admission sequence (inactive)

1. Qualify fixed-capacity acquisition across seeds with absolute held-out competence.
2. Train A→B→C on one persistent state per arm. Compare ordinary training, bounded replay
   and a separately specified retention method with equal total feedback/update costs.
   Require useful absolute competence before interpreting retention percentages.
3. Compare fixed-capacity neuron replacement or rewiring against unchanged training over
   many transitions. Diagnose loss of future acquisition, not just current task accuracy.
4. Test zero-output additions, exact splitting, separately labeled perturbed splitting
   and controlled pruning. Use training-side calibration, centered covariance
   `(H-mean(H)).T@(H-mean(H))/(N-1)` for N>1, gradients, utility and task ablations.
   Low variance alone never authorizes deletion of rare but protected skills.
5. Evaluate isolated structural candidates and unchanged controls from the same complete
   parent, with identical experience and declared short/medium/long adaptation horizons.
   Permit temporary degradation inside candidates; do not weaken committed competence.
6. Only after reliable intervention effects, test learned proposals against fixed/random
   proposals on unseen task families, accounting for proposal-training and search costs.

Every future topology transform needs an explicit parameter/alias map, optimizer-moment,
step-counter, gradient and scheduler treatment, including bias correction. Reject unknown
shapes; do not resize Adam state by generic slicing or median fill. Preserve the immutable
parent; qualify pre/post function and the next update remotely. Changes to normalization
or attention may require a different transformation entirely.

Keep acquisition, retention, future learning-curve area, parameter count, peak memory,
inference latency and adaptation cost separate. Weighted scalar fitness is not Pareto
dominance. Predeclare protected-objective tolerances, uncertainty and a meaningful effect
size; account for multiple candidates using independent confirmation. Noisy improvements
on every metric are unnecessary, but a worthwhile repeatable benefit is necessary.

Promotion must atomically publish the complete state, source/topology metadata, parent
hashes and lineage after qualification. Restore model, optimizer, scheduler, gradients,
all RNGs, sampler/replay/memory and driver history on rollback. Never silently swallow
failures or install independent research weights into an organism. Future experiments
remain finite, interruptible and reversible without a permanent architecture ceiling.
