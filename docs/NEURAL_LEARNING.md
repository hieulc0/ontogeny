# Neural engine and learning contracts

## Implemented model

MiniGPT is a pre-norm decoder-only dense Transformer. Token embeddings and optional
absolute positions pass through dropout, repeated causal attention/residual plus
LayerNorm/GELU feed-forward/residual blocks, final LayerNorm and a token-output projection.
One hidden_states method is used by forward and is available to future task readouts.
The organism keeps its existing Linear(vocab_size,7) head over final token logits;
consolidation does not change that experimental representation or objective.

ModelConfig declares vocabulary, width, layer/head counts, feed-forward width, context,
dropout, biases, normalization epsilon, embedding tying and absolute/RoPE/ALiBi position
encoding. Dense parameter/state_dict names and initialization match the selected source.
Causal SDPA uses is_causal on the ordinary path; explicit masks also intersect causality.
RoPE/ALiBi preserve the source operations. Absolute-position sequences reject overflow.
LM targets are already aligned: no implicit second shift. -1/-100 mask unsupervised
positions; all-masked batches fail instead of returning undefined mean loss.

Expert/depth routing, LoRA, structural skip/growth/pruning, legacy concept bottlenecks and
old task registries are not part of this extracted runtime. Their omission is not a
permanent ban; they need a demonstrated use, clear state contract and independent gates.
There is no permanent width, depth, parameter, memory or topology ceiling. Current
settings and supported operations are finite, versioned experimental envelopes.

## Optimizer and learning state

One AdamW grouping contract registers each live parameter once, including temporarily
frozen parameters. Embeddings (including tied aliases) and one-dimensional tensors have
no weight decay. LearningState.backward clears gradients, checks finite loss/gradients
and backpropagates. step clips, updates the registered optimizer, advances an optional
scheduler and increments the update counter. Policy objectives remain caller-owned;
there is no concept-label assumption inside the engine.

Independent train_tokens uses aligned token CE with an external finite update budget.
mini-gpt-train accepts model JSON and an array of {tokens,targets} batches, persists stream
cursor, config/data hash/budget, full state and update logs. --resume requires the complete
checkpoint's hash and unchanged envelope; it does not reseed. It is an experimental
training interface, not evidence of concept generalization or learning-to-learn.

PolicyCell uses observed energy-change returns, its existing per-time baseline and one
REINFORCE loss. The canonical engine executes its backward/optimizer update. Controls
remain independent; no heuristic action becomes a training target.

## State, admission and conversion

The owner jointly copies model/optimizer/scheduler/external/metadata, preserving aliases,
flags, gradients and all used RNG families. Atomic immutable full-object publication is
shared by both modes. Custom hooks and undeclared external resources are unsupported;
transaction rollback covers only owned state. Snapshot shape, trusted loading, version2
and the explicit dense version1 converter are documented in MIGRATION.md.

All real model construction, forward/backward, checkpoint loading and benchmarks require
validated remote CUDA. The qualified runtime profile is Torch2.6.0+cu124/NumPy2.1.3, with
process-start CUBLAS configuration, deterministic algorithms and recorded GPU inventory.
Default local tooling never trains or evaluates a model. Installing tensor packages for
static analysis does not relax that boundary.

## Scientific status

A.13 binding qualification failed. B.0 demonstrated one continuing state and exact
continuation, but not strong initial concept competence, positive forward transfer or
benefit from learned practice selection. Ontogeny L.1 acquisition and held-out behavior
failed; checkpoint continuity passed. The immediate canary passed and delayed task failed;
timing and stage-binding differences remain confounded. No learning status changes from
package migration or integration tests. Source history remains in the original repository.

After engineering equivalence, study reliable acquisition from finite feedback with
independent concept targets and minimal action-consequence tasks. Use fixed final states,
unchanged controls, disjoint held-out evaluation, multiple initializations and matched
observation/update costs. Isolate representation, optimization and credit assignment;
add neither capacity nor an extra cell as a substitute for diagnosis. Mode C knowledge
consolidation requires its own migration/retention experiment, not blind weight merging.

Framework RNGs are process-global. Separate interleaved research arms must use distinct
processes or explicit state snapshot/restore boundaries; the owner does not create an
implicit per-model Torch generator. Current independent and organism workers run separate
processes. Policy/sampler Random instances remain explicit external-state members.
