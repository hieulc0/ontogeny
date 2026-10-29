# Architecture

```mermaid
flowchart LR
  W[CellWorld private simulator] --> O[Immutable partial observation]
  O --> C[Cell policy interface]
  C --> A[Finite action]
  A --> W
  W --> E[Observed consequence and experience]
  E --> U[Bounded learning update]
  U --> S[Upstream LearningState]
  S --> C
  W --> V[External evaluator and provenance]
```

CellWorld owns world generation, transitions, energy and termination. Its private state
and snapshot API belong to the driver/evaluator, never to the policy. Isolation is an
interface convention in one Python process, not a security sandbox against malicious code.
A policy gets Observation and later Experience, never a world reference or seed.
Action is an explicit seven-member enum; invalid values fail before state mutation.

The Cell runner coordinates observations, actions and optional feedback updates. It
accepts any typed Policy, so controls share the same environment contract. It logs each
observed experience through a callback without supplying hidden maps to the learner.
Episode metrics are outside the policy. Internal state is owned by a persistent policy
and, for the neural pilot, upstream LearningState.external; no duplicate state owner.

The active L.1 policy uses the upstream MiniGPT public logits interface plus a fresh
seven-action linear head. A masked 29-token partial observation produces a stochastic
action distribution. Actual energy-change returns weight an episode REINFORCE objective.
One objective only; no predictor/planner or heuristic supervision. The unexecuted original
consequence predictor remains available as historical experimental code, not an active
learning objective. See L1_QUALIFICATION.md for the frozen design.

LearningState owns the upstream model, optimizer, scheduler, RNG snapshot and declared
external graph. Policy RNG, bounded experience history, driver world snapshots and update
counters live there. An immutable parent checkpoint hash links each final checkpoint.
Trusted pickle checkpoints require exact code/runtime and cannot be accepted from unknown
sources. Episode boundaries preserve the same owner within each arm. Evaluation freezes
updates; it must not become a source of training choices.

Snapshot continuation of CellWorld is locally tested. Full learner fresh-process
continuation is an outstanding remote admission gate. Arbitrary cross-version topology
migration is unsupported. Future multi-cell work must define shared versus private state,
communication observations/actions, costs and credit assignment, with no hidden hard-coded
roles and a capacity-matched single-cell control. None is implemented at L.0.
