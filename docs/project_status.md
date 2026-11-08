# Project status

Updated: 2026-11-08

## Approved scope and completed work

Milestone 1: minimal repository foundation, complete.
Milestone 2: environment audit and benchmark planning, review documents complete
with subsequent host GPU validation complete. User approved read-only checks and
documentation; no installation, data acquisition, or model implementation.

- Foundation committed by user; local HEAD observed as b58e28f.
- GitHub remote: git@github.com:emon109848/Traffic-forecasting-current-research.git.
- Added environment_audit.md, model_compatibility.md, benchmark_proposal.md.
- Updated README and decision log alongside the review and validation records.
- Reviewed primary paper/reference code and documented input/fidelity concerns.
- Confirmed 15-minute SD benchmark cadence from source preparation and config.
- Existing pml imports torch successfully; GPU unavailable within this session.

## Findings and unresolved gates

Conda 26.7.1; WSL2; roughly 11 GiB visible RAM and 701 GiB free project storage.
pml contains Python 3.14.8, torch 2.14.1 (CUDA 13.0 build), NumPy 2.5.4.
Agent-session nvidia-smi reports OS-blocked GPU access. The user subsequently
provided host output confirming an RTX 5060-family GPU (full suffix truncated),
8,151 MiB VRAM, NVIDIA-SMI 615.78.05, KMD 617.42, CUDA UMD 13.4.
Host GPU visibility and PyTorch CUDA execution are confirmed by the subsequent
user-run validation below.

Recommend SD 2019, 716 sensors, 15-minute flow, three-hour history and forecast.
This is a proposal, not a frozen protocol. Missingness, exact provenance, units,
calendar and acquisition path require actual data inspection. Paper reproduction
and shared-protocol adaptation remain separate. Source commits/licenses must be
recorded before incorporating external code. Project license remains undecided.

## Environment validation — complete

User supplied a successful strict host check in pml: CPU values/gradients/SGD,
NumPy interoperability, and CUDA checks all passed. GPU is NVIDIA GeForce RTX
5060 Laptop GPU, 8,518,041,600 bytes VRAM, compute capability 12.0. Versions:
Python 3.14.8, torch 2.14.1+cu130, NumPy 2.5.4. Retain pml for current development
without package changes. See environment_validation.md for user-reported output
and limits. Agent-session GPU restrictions remain; host execution works.
No final dependency lock or minimal environment.yml exists yet.

## Dataset acquisition and local inspection — complete with limitations

User supplied flowsd.npz and sd_meta.csv. Original files unchanged and ignored
by Git. SHA-256 hashes and inspections recorded under docs/datasets/.
Traffic is float64 (35,040,716,1); metadata has 716 unique IDs, valid unique
coordinates and contiguous source ID2 indices 6931–7646. No empty metadata
fields. Traffic contains 563,841 zeros and four all-zero frames (6536–6539).

Available artifacts are structurally consistent, but sensor-column alignment,
original timestamps, exact missingness semantics and hosted data terms are not
independently established. Do not claim these gates were verified. No graph
acquired, preprocessing applied, or experiments run.

## Data protocol decisions — approved

User approved using processed author artifacts with documented assumptions:
metadata row order is paired with NPZ columns; the published 2019/15-minute
setting is assumed rather than independently verified; zero targets are excluded
from loss and metrics. Preserve original files. See benchmark_proposal.md.

## Data pipeline milestone — complete

Implemented src/traffic_benchmark/data/pipeline.py, synthetic tests and
scripts/inspect_pipeline.py. See docs/data_pipeline.md for decisions and usage.
Train-only global scaling includes zeros; zero inputs unchanged. Targets remain
original-unit values with separate exclusion mask. Strict target boundaries,
lazy windows, assumed calendar and metadata row order are explicit.
Real data yields 21,001 / 6,997 / 6,997 windows; source hashes unchanged.
No training, imputation, artifact conversion or dependency installation.

## Masked metrics and persistence milestone — complete

Implemented streaming MAE/RMSE/percentage MAPE and deterministic persistence.
17 tests pass; direct-array checks at leads 1/3/6/12 match on both splits.
Measured results and provenance are in reports/persistence_sd_v1.json and .md.
Pooled test MAE 60.7825, RMSE 94.3350, MAPE 41.7801%, valid targets 96.7400%.
No training or parameter fitting; no packages installed. CLI stores full outputs
in ignored runs/. Reports identify uncommitted code through implementation hashes.

## Proposed next milestone — awaiting approval

Add a training-only seasonal-average baseline, or start the STAEformer paper
walkthrough and architecture specification. Explain the chosen scope before
implementation. Neural training and attention models remain unapproved. Exact
dependency/environment locking remains open before expensive experiments.

## Validation

Documentation links, metadata and scaffold import checked with the standard
library. 17 pipeline/evaluation tests pass. Real-data loading and deterministic baseline
evaluation completed; measured accuracy results are recorded without neural claims.
No packages, repositories or model weights were downloaded by the agent.
The traffic dataset was downloaded by the user and inspected locally.
