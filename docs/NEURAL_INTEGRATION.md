# Internal Mini-GPT integration

The current runtime imports mini_gpt from this repository. It has no sibling-repository
runtime dependency. Canonical architecture/training: NEURAL_LEARNING.md. Source inventory,
exact ancestry, state conversion and actual engineering qualification: MIGRATION.md.

Ontogeny's thin factory configures the admitted remote runtime, verifies internal module
paths, builds the same dense brain and organism action head and allocates one persistent
LearningState. Observation encoding and feedback remain organism-owned. The policy does
not access simulator internals. Research students and developmental organisms share code,
not mutable weights/optimizers/history. No historical student is silently reset/replaced.

New execution requests use schema2/source {ontogeny_sha, engine: mini_gpt}; provider
launchers clone only the pinned Ontogeny source and install pinned GPU dependencies.
Historical schema1 remains readable for artifact integrity checks and requires its
original worker/repository for execution. Normal new checkpoint loading is format2.
An explicit remote dense-format1 import can remap admitted class paths without replacing
parameters; unsupported historical models continue using their original source.

The previous external integration findings and launchers are preserved in Ontogeny Git
at 670709ca2d319baf3073be0f7ce709f3f8d664d1 and in immutable experiment records. Their
actual source pin was 8567836fb2c8c915d442b7c2fbedb4fd9b30396d. New source extraction is
from 87e8928c440fecc4f3a6f6c5ea3c869a6a8537e6; selected component bytes are identical.
This does not claim all intervening source research is integrated or qualified.

Existing provider results belong to their tested revisions. The prior first-update
fixture qualified Colab→Colab, Kaggle→Kaggle and Colab→Kaggle; the reverse direction was
blocked by Colab allocation refusal. Its input optimizer was initially empty. The new
migration protocol explicitly tests a trained optimizer/checkpoint and independent
multi-update continuation on Kaggle. No automatic cross-provider migration, arbitrary
VM-loss recovery or historical concept-student consolidation is thereby implied.

All actual neural/checkpoint checks run remotely; local checks are inspected non-neural
contracts, packaging and static analysis. L.1 remains Learning FAIL; no L.2 or multicellular
promotion follows from this engineering consolidation.
