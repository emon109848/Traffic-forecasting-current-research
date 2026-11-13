# Project status

Updated: 2026-11-13

## Current checkpoint

PatchSTG seed-2023 training and frozen-checkpoint test evaluation are complete.
Epoch 50: validation MAE 14.9936; full test MAE 17.1943, RMSE 29.1370,
MAPE 11.1706%. See [test report](../reports/patchstg_sd_seed2023_test_v1.md).
All 58 tests pass. CUDA is now accessible in the agent session; the full test
pass ran here. Earlier CUDA/write restrictions below are historical.

Next proposed milestone (not approved): resolve BigST variant and implement/test
it under the same protocol. Short-history/no-LSFE/no-spatial-loss remains a
proposal requiring explicit confirmation and an accurate variant label. Exact
environment recreation and controlled efficiency benchmarking remain open.
Do not tune PatchSTG using its now-observed test results. Do not launch further
training, additional seeds or BigST work without approval.

Earlier sections below record the progression of completed milestones; their
"no training" and "pending" statements describe their dates, not current status.

## Current deadline scope

Compare BigST and PatchSTG on processed SD 2019 by November 24, 2026.
Keep both simple baselines and all shared infrastructure. ASTGCN/STAEformer,
extra datasets and broad scalability work are deferred. See roadmap.md.

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
The pipeline and mask rules below are implemented. Research-model input/history
requirements and training/efficiency settings still need review. Paper reproduction
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
acquired; shared normalization and simple baseline evaluations are complete.

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

## Seasonal-average baseline milestone — complete

Training-only weekday/slot/sensor nonzero means with sensor/global fallback
implemented and evaluated. All 23 tests passed. Independent seasonal grouping
and lead-12 validation/test metrics matched. Same source hashes, windows and
mask counts as persistence. Reports: reports/seasonal_average_sd_v1.json and .md.
Pooled test MAE 28.0045, RMSE 53.3906, MAPE 19.9964%. Four sensors lack nonzero
training values; global training fallback used. Source files unchanged. No
packages installed, neural training, graph preprocessing or model implementation.

## BigST/PatchSTG architecture review — complete, scope choice pending

Reviewed both primary papers and author model/config/preprocessing/trainer files.
Added docs/papers/patchstg.md, docs/papers/bigst.md, docs/model_sources.json and
docs/model_implementation_plan.md. Immutable references identified; file-level
pin verification partial and explicitly recorded. PatchSTG MIT license observed;
BigST root license not found, so no code-reuse permission assumed.

Found PatchSTG reference 288-step similarity segments despite SD's 96 daily
slots; recommend preserving reference behavior with a discrepancy label first.
Recommend BigST short-history core without LSFE/spatial regularizer for deadline,
explicitly labeled as a variant; user confirmation requested and pending.
No architecture code, index maps, model memory measurements or training performed.
Review completed Nov 11, later than original Nov 8–10 target. Keep deadline Nov 24.

## PatchSTG spatial/model milestone — complete

User approved moving to PatchSTG implementation. Added training-only spatial
preparation, integrity/provenance-checked caches, config/model/attention blocks,
CPU check CLI, tests and MIT attribution notice. Code-aligned adaptation, not an
official reproduction. Defaults preserve 288-step similarity and full SD widths.
Stable coordinate/similarity ties are explicit adaptations.

35 tests pass. Real-data full-width batch-one CPU forward/backward and exact
eval checkpoint reload passed. Model: 2,284,844 trainable parameters, 16 patches
of 64 positions, output (1,12,716,1). See docs/patchstg_implementation.md and
reports/patchstg_cpu_check.json. Source arrays unchanged; no packages installed.
No training, optimizer step, GPU model feasibility or forecast accuracy measured.

## GPU feasibility milestone — complete

User-terminal run passed full SD batches 1/2/4 on RTX 5060 Laptop GPU, including
one disposable AdamW step per batch. Peaks allocated/reserved: 168.93/198 MiB,
268.40/282 MiB, 455.00/514 MiB respectively. Finite outputs, gradients and updated
parameters passed. Current implementation hashes match the host report.
Curated report/table: reports/patchstg_gpu_feasibility.json and .md.
Agent CUDA remains unavailable; measurements came from the user’s actual host.
All 42 tests passed at the last code check. No full training experiment or
forecast accuracy for PatchSTG exists. One-off timings are not throughput data.

## Training-harness milestone — implemented and CPU verified

User approved the trainer milestone. Added training package, train_patchstg.py
CLI, capped configs/patchstg_smoke.json and 11 tests. All 53 tests pass. Supports
original-unit masked MAE, empty-batch skip, seeded shuffle, pooled validation,
best/last checkpoints, epoch-boundary resume and source/software/RNG provenance.
No test evaluation in the trainer. See docs/patchstg_training.md.

Real-data full-width CPU diagnostic: two updates, two validation batches,
best-checkpoint reload reproduces diagnostic MAE. Curated subset record in
reports/patchstg_trainer_cpu_diagnostic.json; not comparable to full baselines.
Source arrays unchanged; no new packages. Agent CUDA attempt exited unavailable
without training. Final experiment environment lock remains pending.

## Host training diagnostic — complete

User ran the capped CUDA configuration: 20 updates, no skipped batches, 80
validation windows, no test evaluation. Report/code hashes verified. Training
wall 12.003 s includes a 10.616 s first-step startup; mean after first 5 updates
0.06721 s. Peak allocated/reserved 472.60/540 MiB. Best/last checkpoints produced.
See reports/patchstg_training_smoke_host.json and .md. Subset validation MAE
147.4188 is diagnostic, not comparable to full-split baselines.

## Full-epoch calibration — complete

User completed fresh full-split calibration with batch 4, seed 2023: 5,251
updates, no skips, all 6,997 validation windows. Training 350.97 s, validation
15.11 s (6.10 min combined; excludes setup/checkpoint writes). Full validation
MAE 25.2203, RMSE 36.1422, MAPE 24.5854%. No test evaluation.
Peak allocated/reserved 472.60/584 MiB. Code hashes and baseline validation
valid-target counts checked. Best/last checkpoints saved. See
reports/patchstg_calibration_v1.json and .md.

## PatchSTG 50-total-epoch continuation — complete

Verified completed report, all epochs 1–50, 262,550 optimizer updates, no skips.
Best epoch 50, validation MAE 14.9936 / RMSE 25.4044 / MAPE 9.0571%.
Train plus validation including calibration: 4.6948 hours; setup/checkpoint
writes excluded. Training sources match original recorded hashes. Curated
summary: reports/patchstg_sd_seed2023_training.json. Original run unchanged.

## PatchSTG frozen-checkpoint test evaluation — complete

User explicitly approved the milestone. Added evaluate_patchstg.py, checkpoint
integrity helpers, five focused tests and docs/patchstg_evaluation.md. All 58
tests pass. Evaluated frozen best.pt with saved maps, original train-only
normalization, no gradients/dropout/updates, batch 4, all 6,997 test windows.
Test MAE 17.1943, RMSE 29.1370, MAPE 11.1706%; all 12 horizons recorded.
Dataset hashes, splits and per-horizon target counts match both baselines.
MAE is 38.60% below seasonal average. Reports: reports/patchstg_sd_seed2023_test_v1.*.
Full scoring pass 17.174 s includes batching, transfers and metrics; it is not
isolated inference latency. One seed only; no variability or scalability claim.
The earlier interrupted launch produced no report; process absence was checked
before restarting. No training or test-based tuning took place.

## Validation

Documentation links, metadata and scaffold import checked with the standard
library. 58 pipeline/baseline/evaluation/model/feasibility/training tests pass. Real-data loading and deterministic baseline
evaluation completed; PatchSTG neural test accuracy is now recorded above.
No packages, repositories or model weights were downloaded by the agent.
The traffic dataset was downloaded by the user and inspected locally.

## Learning documentation

Added workflow.md at the user’s request to explain completed milestones, their
purpose, design choices, verification, measured results and remaining limits.
No new implementation milestone started.
