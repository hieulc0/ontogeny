# Initial neural student engineering pilot

Status: preregistered; remote NOT RUN. This is not a language qualification.

Initialize one fresh Mini-GPT student with 5 layers, width 384, 6 heads, feed-forward
width 1792, byte vocabulary 256, RoPE, context 512 and dropout 0.1. Expected parameter
count approximately 10.05 million; report the actual count, not an enforced ceiling.
The finite configuration is an initial hypothesis. No earlier learner is replaced.

Use the two checked-in short text sequences in batches.json, byte shifted by one position.
Run two updates at AdamW learning rate 0.0005 with seed42 and the existing independent
trainer. A fresh-process audit repeats update 2 from checkpoint 1: all nine state groups,
including model, optimizer, external batch cursor, gradients and RNG, must match checkpoint 2.
Also record a held-out next-byte loss on a separate short sentence as an engineering
measurement; no training or decisions use that sentence. Two updates do not demonstrate
language acquisition, transfer or learning to learn.

Budget: at most three physical updates (two continuing, one discarded audit), at most
512 input tokens including the held-out forward, 300 seconds of task time, one T4 with
one visible CUDA device, 3 GiB Torch-allocated peak per process, 1 GiB output. Allow at
most 1,050 seconds for provider setup/execution. Record actual model count, loss, task
wall time, peak allocation, hardware/runtime, source identity and both complete checkpoints.
No automatic retry or CPU fallback. Stop on nonfinite values, cap/lineage mismatch,
continuation inequality, absent GPU or incomplete artifacts. Preserve failures.

Use one Colab T4 allocation and positive readiness proof first. Confirmed unavailable
permits one Kaggle submission with a fresh quota check and at least 2.5 hours reserved;
ambiguous allocation/cleanup blocks provider fallback. The pilot has its own learner and
must not be merged into the developmental organism or represented as historical continuity.
Full checkpoints are trusted objects, archived outside Git. Successful delivery requires
export and verification before releasing Colab; native Kaggle completion alone is not enough.
