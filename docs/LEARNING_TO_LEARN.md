# Learning to learn: experimental direction

A model must store information in weights or other state to retain anything. Stored
knowledge and learning to learn are different claims: the latter requires evidence that
prior experience helps the same learner acquire a new task more effectively, not merely
that it recalls previous examples or has lower loss on familiar templates. The first
student's two updates are an engineering pilot, not that evidence.

The starter curriculum provides an auditable data and state-continuation smoke test. Its
held-out examples share templates, domains and operands with training. It cannot establish
learning to learn or wide subject competence. A real comparison requires distinct generated
rule families with newly sampled underlying rules, not just different rows. Reserve unseen
families and examples before training, check content overlap, and prevent final evaluation
material entering training or development artifacts. Data should include comparable tasks
that require feedback and adaptation; next-byte loss alone is insufficient.

A future preregistered pilot should use multiple independent initializations and matched
compute, token and gradient budgets. Compare (1) a continuing student with history A,
(2) a control trained on the same amount of unrelated experience and (3) a fresh or
history-matched student. Evaluate all on previously unseen tasks B: measure accuracy
before feedback, after each fixed number of updates, and the area under the learning curve.
Use separate held-out examples per task, paired task seeds, uncertainty intervals across
initializations and tasks, and retain all outcomes. Evaluate retained A competence before
and after B. Include fixed/no-update controls to distinguish prior generalization from
adaptation. Predeclare the final checkpoints, primary comparison, threshold and stop gates;
report negative transfer and failures. A separate task family C provides independent
confirmation after design decisions are frozen.

Ordinary gradients change weights; they do not themselves demonstrate learning to learn.
Changing width, topology, inputs or outputs is a distinct later hypothesis, not an
assumed benefit or an immediate autonomous permission. Any candidate must preserve an
immutable parent, map or explicitly reinitialize affected state, document optimizer and
representation compatibility, compare matched resources and test retained competence.
No growth is admitted by the present scientific gates. See RESEARCH_PROTOCOL.md.

## Continuation contract

A continuing research student owns one complete LearningState, including optimizer,
RNGs, external sampler cursors, progress and segment lineage. A segment explicitly names
its immutable parent SHA-256, dataset hash, original model configuration and finite
additional update budget. It starts a new output directory and writes `segment-start.pt`
before any update; each update then publishes a full checkpoint. A new dataset starts at
its own cursor; returning to an earlier dataset uses its recorded cursor. Segment resumption
uses the existing segment's fixed data and end step. Loading is only for trusted same-code
checkpoints in a validated remote CUDA worker. This implementation has local contract tests;
its first-update continuation was verified remotely on Colab; a longer interruption/resume comparison is tracked in experiments/neural/continuation/.

```bash
# In a validated GPU worker; config must match the parent checkpoint.
mini-gpt-train --config experiments/neural/initial_student/model.json \
  --data data/generated/starter/train-batches.json --output segment-run \
  --resume step-2.pt --checkpoint-sha256 PARENT_SHA256 \
  --new-segment starter --additional-updates 20 --wall-seconds 300
# To resume an interrupted segment, choose its last checkpoint and its exact SHA-256.
mini-gpt-train --config experiments/neural/initial_student/model.json \
  --data data/generated/starter/train-batches.json --output resumed-run \
  --resume segment-run/step-3.pt --checkpoint-sha256 CHECKPOINT_SHA256 \
  --resume-segment starter --wall-seconds 300
```

The commands describe the interface, not a claimed completed training run. Respect the
remote-provider admission policy and commit the exact source before executing them.

The rule-task generator and paired curve-analysis contracts are documented in
`experiments/neural/rule_transfer/README.md`. The neural four-arm cohort is not yet run.
