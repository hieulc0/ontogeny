# Append-only experimental history

## 2026-10-29 — Initialization / L.0 v1

PLANNED: deterministic finite world, nonlearning controls, integration contracts; no
local neural execution. Upstream clean source inspected at
8567836fb2c8c915d442b7c2fbedb4fd9b30396d. Existing Ontogeny LICENSE preserved.

EXECUTED locally: 31 explicitly inspected environment/guard/encoding/fake-policy tests;
Ruff checks/format; Pyright; ten development seeds per control, 20 episodes total.
PASSED ENGINEERING CHECKS: all tests; final lint and type checks clean.
L.0 qualification: PASS for the declared v1 engineering invariants and controls.
Raw baseline rows are versioned in experiments/l0_cellworld/baselines.json.

Random mean duration 39.3, collected 0.6, hazard exposures 0.9, final energy 0.
Fixed heuristic mean duration 100, collected 14.5, hazard exposures 0, final energy 40.1.
These development results are not a neural or sealed scientific qualification.

Failed setup/check attempts preserved here: initial uv sync could not write default
cache; temporary writable cache resolved that. Sandboxed PyPI DNS then failed;
authorized dependency installation succeeded. Initial lint found default-constructor,
import-order and long-line issues; corrected. Initial Pyright found missing remote-only
packages, interpreter selection and frozen-field test assignment; remote imports now
lazy, local environment explicit, tests corrected. No failed neural runs occurred.

## 2026-10-29 — L.1 immediate-consequence-v1

PLANNED / PILOT IMPLEMENTED / NOT EXECUTED. One CE objective over real action energy
outcomes, new fresh model lineage; no historical knowledge continuity claimed.
No Kaggle submission or kernel/session result exists. NOT PASSED LEARNING QUALIFICATION.
Admission requires remote fresh-process next-update equivalence, independently audited
evaluator/controls/variations and runtime provenance. See NEURAL_INTEGRATION.md.
L.2–L.5 NOT STARTED. No checkpoint promoted and no upstream edits.

## 2026-10-29 — L.1 REINFORCE v1 preregistration

Supersedes the unexecuted immediate-consequence proposal before any neural outcomes.
Existing CellWorld is frozen. New task-specific action head on actual MiniGPT; one
REINFORCE objective. Kaggle/GitHub access verified, upstream remains clean/pinned.
Protocol, H1/H2/H3 and resource budgets frozen in L1_QUALIFICATION.md. No L.2.
