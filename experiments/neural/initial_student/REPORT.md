# Initial student pilot result

**Engineering PASS; scientific learning NOT RUN.** One new independent research student,
not a replacement for any existing organism. Source used by the actual GPU trial:
05ff24b082933b08d7d47545a06cafd281279a94. Its first Colab attempt has no recovered state;
the successful Kaggle run is an explicitly new fresh lineage, seed42.

| Measurement | Actual result |
|---|---:|
| Parameters | 10,054,016 |
| Model configuration | 5 layers, width384, 6 heads, FF1792, RoPE, context512, byte vocabulary256 |
| Continuing updates | 2 |
| Discarded fresh-process audit updates | 1 |
| Training input tokens | 128, plus64 audit tokens |
| Held-out input tokens | 40 |
| Training loss, first / second update | 5.67105 / 4.37176 on different tiny batches |
| Held-out next-byte loss after training | 4.24892 |
| Continuation | Exact all nine model/optimizer/scheduler/external/metadata/gradient/RNG/mode/flag groups |
| Peak Torch allocation across processes | 565,831,168 bytes, approximately540 MiB |
| Task wall time | 13.99 seconds, excluding provider setup/download |
| GPU/runtime | Tesla T4, 1 visible of 2 allocated, Python3.13.15, Torch2.6.0+cu124, CUDA12.4, NumPy2.1.3 |

The two training losses use different sequences and only two updates; they are not an
acquisition curve or evidence of language generalization, transfer or learning to learn.
The held-out loss is a pilot measurement, not a predeclared success threshold. Original
L.1 remains Learning FAIL. No CellWorld or subsequent stage was newly qualified.

The completed native job was
sparklinghydraulics0/ontogeny-student-fresh-05ff24b082/1. One prior Colab T4 assignment
failed when its execution WebSocket closed; no archive was recovered and its possible
partial work is unknown. Its session was stopped. The Kaggle student was separately
preregistered and freshly initialized, with no claim of continuity from that failed
attempt. This was not automatic provider hopping of a continuing student.

Kaggle reported quota2.67 h before / 2.61 h after, with2.5 h reserve. Its 480-second
platform cap includes setup; no CPU fallback occurred. The two complete checkpoints and
six other indexed files were downloaded and independently hash-verified; no checkpoint
was loaded locally. Task peak excludes driver/context overhead and is not an OS memory
limit. The preceding failed attempt has no complete counters and is not counted as zero.

Saved learner state:

- [First update](../../../runs/initial-student-kaggle-verified/student/step-1.pt):
  SHA-256 `3f93ebb0a162304f279081267555bf875965acab3cea4b43279f3e0c00c3a5c6`.
- [Second update](../../../runs/initial-student-kaggle-verified/student/step-2.pt):
  SHA-256 `8e4ea63d5e3c3719b30364614005b631d202dbf09cb666796f89f0fd2b35df7d`.

Each is 162,303,514 bytes and contains the complete trusted LearningState, including model,
optimizer, gradients, progress, batch cursor and RNGs. The second followed the first in
one continuing owner; results/lineage.json records the relationship. The archived
checkpoint files are under ignored runs/ and are not in Git. See results/kaggle-receipt.json
for all verified file identities and results/kaggle-summary.json for full runtime details.
Loading or continuing them requires validated remote CUDA and the internal mini_gpt package.

The current CLI requires its original fixed data/update envelope. Continuing the same
student onto a new subject or beyond the original cap requires an explicit training-segment
interface; it must carry the checkpoint/optimizer forward, preserve prior segment and
sampler lineage, declare the new budget and add retention/held-out evaluation. That is the
next implementation, not a capability implied by this two-update pilot.

## Reproduction

The protocol and checked-in byte batches are in this directory. On a clean source checkout,
the controller tries one verified Colab T4 and uses quota-admitted Kaggle only on confirmed
pre-start unavailability:

```bash
python -m experiments.neural.initial_student.launch --directory runs/new-student-pilot
```

A new run creates a new student; it does not continue the checkpoint above. The separately
preregistered Kaggle recovery command was `python -m
experiments.neural.initial_student.submit_kaggle --directory runs/initial-student-kaggle`.
It was used only after recording the unknown Colab transport failure and admitting a fresh
lineage. Do not rerun an ambiguous submitted job. The exact protocol, configuration,
input batches, update log, complete states and verified checksums accompany the raw output.

Local artifact re-verification reads files and hashes only; it does not deserialize model
weights:

```bash
python -c 'from pathlib import Path; from experiments.neural.initial_student.job import verify; print(verify(Path("runs/initial-student-kaggle-verified"))["continuation_exact"])'
```
