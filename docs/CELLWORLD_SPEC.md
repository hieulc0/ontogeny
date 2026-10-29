# CellWorld v1

Default grid: 12 × 12, one cell, 24 single-use resources, 12 hazards. A private
random.Random(seed) shuffles row-major positions: first is organism, next resources,
next hazards, all initially disjoint. Defaults live in configs/cellworld.toml.
Energy starts at 40, capacity 60, horizon 100. All parameters are finite integers.

Actions are NORTH/SOUTH/EAST/WEST/INSPECT/INTERACT/REST. Movement changes position by
one tile; boundary moves remain in place and still cost energy. Interact collects a
resource only on the current tile, removes it and adds 15 energy. Rest gives no energy.
Inspect costs 2; every other action costs 1. Hazards cost 6 on every step ending on a
hazard, including rest and inspection. Consequences are atomic: action cost, movement
or interaction, hazard cost, then clamp energy to [0, capacity]. A resource may rescue
low energy within that atomic transition. energy_delta is actual clamped change,
not nominal reward. No additional scalar survival bonus exists.

Energy zero terminates. Reaching horizon truncates unless energy zero already terminated.
Stepping after either fails. At every 20th step, hazards relocate uniformly without
replacement among all non-resource, non-organism tiles using the explicit world RNG.
Relocation occurs after action consequences and before observation, also at terminal
steps. No resource respawn; finite resources and metabolic costs exclude infinite rest.

Observation: immutable relative row-major tile tuples (dx,dy,kind), exact energy,
latest consequence, sensing radius. Normal radius 1 (9 tiles); inspection's returned
observation has radius 2 (25 tiles), lasting until the next action. Tiles are empty,
wall, resource or hazard. No coordinates, seed, RNG, complete map or future state enter
the learner interface. Sensory cost is the action cost plus inspected tile accounting;
there is no free policy request to expand sensing through the runner.

Transition also returns collection/hazard flags and termination/truncation. These are
observed feedback, not hidden truth. Full snapshots include version/config/state/RNG
and are explicitly privileged. Restore deep-copies state. Determinism is defined for
same Python/code version, configuration, seed or snapshot and action sequence.

Validation: disjoint layout; action validation before mutation; bounded energy with
exact ledger; resource depletion; boundary behavior; inspection range/cost; stopping;
seeded relocation; snapshot alias isolation; observation field whitelist; deterministic
replay and controls. Ten baseline seeds are development fixtures, never sealed L.1 tests.
The large heuristic advantage confirms that useful sensor-based behavior is possible;
it does not certify a neural learner or an exploit-proof simulator.
