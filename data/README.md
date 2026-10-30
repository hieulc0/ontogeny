# Student data

The first curriculum is generated from code and has no external dataset dependency.
It contains 2,000 short synthetic examples: arithmetic, simple English instructions,
Python integer expressions, constant-speed distance and atom-group counting. The physics
and chemistry examples are idealized arithmetic contexts, not coverage of those subjects.
Do not use this corpus as evidence of broad subject understanding. No personal data is
included; the generated examples are project-authored.

```bash
PYTHONPATH=src python -m mini_gpt.data.curriculum --output data/generated/starter
```

The command refuses to overwrite an existing directory. The output manifest records a
SHA-256 for every file, generation seed, counts, encoding, origin and splitting rule.
`train-batches.json` is the only file intended for the trainer. `development.jsonl` and
`test.jsonl` include prompts and answers for external evaluation; never pass them into a
training job. The files are generated locally and excluded from Git; source and generation
rules are tracked. Keep any experiment's exact manifest and its generated files as
immutable artifacts alongside its checkpoints.

Each example's domain and operand pair determine its split through SHA-256. The same
instance cannot occur in multiple splits; permutations or equivalent equations can still
be related across splits, and all splits use the same templates and number range. This is
an instance-generalization check, not a held-out task-family or contamination-proof
learning-to-learn assessment. The generator does not produce an independently sealed
study: a future protocol must create new task families, archive secret or otherwise
restricted final test inputs and freeze all decisions before evaluating them.

The trainer uses byte tokens. Prompt targets are masked and only answer bytes and newline
are supervised. Each batch contains one variable-length example and fits the student's
512-byte context. Future curated language, science and programming sources require
explicit provenance, usage rights, content quality and split/contamination review before
inclusion. Do not fetch unverified material into the training set by default.
