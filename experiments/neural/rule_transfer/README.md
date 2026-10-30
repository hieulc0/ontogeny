# Rule transfer study preparation

Status: task generator and paired curve analysis implemented; neural cohort NOT RUN.
The first target is positive forward transfer at fixed architecture, not topology growth.
Reptile, replay, optimizer-moment changes and neuron replacement are separate hypotheses.

A task hides a modular affine rule `y=(a*x+b) mod m` from the student. Experience uses
multipliers 1–4 modulo 11; development uses multipliers 5–8 modulo 11. The confirmation
family uses multipliers 2–5 modulo 13. Each task's shuffled input values are divided into
disjoint support/feedback and query sets. The model sees only `input=...; output=` and
observed feedback, never the parameters or task ID. The generator creates only a requested
family. Task identities are disjoint, although tasks share arithmetic structure, prompt
templates and possible outputs. The confirmation generator is public and therefore not a
sealed test; instantiate a separate, access-controlled confirmation cohort only after
freezing the final protocol. Changing modulus also changes output range and task difficulty,
so it can confound transfer. This is a narrow synthetic diagnostic, not subject competence.

```bash
PYTHONPATH=src python -m mini_gpt.data.rules --family experience --seed 42 --output data/generated/rules-experience
PYTHONPATH=src python -m mini_gpt.data.rules --family development --seed 42 --output data/generated/rules-development
PYTHONPATH=src python -m mini_gpt.evaluation.curves --input curves.json --output report.json
```

The analysis input is `{"updates": [0,1,2,5], "curves": [...]}`. Each curve has
`model_seed`, `task`, `arm` (continuing, unrelated, fresh or frozen), and `accuracy`
(full-answer fractions in update order). Require a complete crossed seed/task/arm cohort;
frozen values must remain fixed and match continuing at update zero. The primary descriptive
statistic is the difference between continuing normalized trapezoidal AULC and the larger
of unrelated or fresh mean AULC. Crossed bootstrap resamples initialization seeds and tasks
independently, preserves pairing, and recomputes that maximum in each replicate. Report its
limitations with small cohorts. Optional `retention` rows contain `model_seed`, `task`,
`before` and `after`; all raw values are retained. A higher AULC is not sufficient by
itself: compare zero-update and adaptation gains, prior competence and retention, report
per-seed/task failures, total interaction, gradient, time and compute cost.

Before scientific execution, implement and locally test the actual four-arm neural driver,
then preregister source, model seeds, task/instance splits, A and control training budgets,
B update schedule, full-answer decoding, checkpoint selection, retention sampling, resource
caps, confidence/effect-size and stop conditions. The current analysis accepts observations;
it does not train agents or prove that input rows came from a valid experiment. Train and
evaluate neural models only remotely on validated CUDA. No final test should influence tuning.

Sources: Nichol et al., [Reptile](https://arxiv.org/abs/1803.02999); Dohare et al.,
[loss of plasticity](https://www.nature.com/articles/s41586-024-07711-7); Javed et al.,
[forgetting and forward transfer](https://arxiv.org/abs/2303.08207). These motivate separate
comparisons; they do not establish performance of this implementation.
