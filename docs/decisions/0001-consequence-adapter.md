# 0001: Consequence adapter over upstream MiniGPT

Status: archived unexecuted proposal; superseded for L.1 by decision 0002.

Canonical concept loss expects tag/state supervision, incompatible with environmental
experiences. Reuse MiniGPT and LearningState persistence/optimizer, add no Transformer
copy or hidden survival planner. Predict immediate energy outcome conditioned on local
observation and proposed action, with one online CE update per actual transition.
Action preferences derive from those predictions. Existing LM logits supply a bounded
categorical readout; no new module needed for this first hypothesis.

Tradeoff: immediate outcomes may never learn costly exploration, inspection or memory.
Epsilon exploration and resource caps can bias behavior. Preserve this negative result
if it occurs. Do not add value learning, planning or replay objectives retrospectively.
Fresh model means no historical continuity claim. Upstream remains unchanged. Full
checkpoint continuation and independent unseen-world qualification are required remotely.
