# Deadline scope and roadmap

Updated 2026-11-13. Deadline: November 24, 2026. This replaces the initial
four-model implementation roadmap; the work already completed remains useful.

## Deliverable

A documented comparison of BigST and PatchSTG on processed SD 2019, alongside
existing persistence and seasonal-average baselines. Target results: original-unit
MAE/RMSE, percentage MAPE, horizon plots, parameter counts, training time,
inference timing and peak GPU memory under clearly stated measurement conditions.
Use the existing 716-sensor, assumed 15-minute, 12-to-12 pipeline and zero-target
mask unless an explicitly documented model requirement necessitates an amendment.

Scope excludes additional datasets, ASTGCN, STAEformer, newer-model searches,
large-network scalability claims, extensive tuning, deployment and a web UI.
Ablations are optional only after core experiments and reporting are complete.
The existing framework and both baseline results are retained unchanged.

## Current progress — November 13

PatchSTG implementation, full 50-epoch seed-2023 training and frozen-checkpoint
full test evaluation are complete; all 58 tests pass. Test MAE 17.1943, RMSE
29.1370, MAPE 11.1706%. See [results](../reports/patchstg_sd_seed2023_test_v1.md).
Next proposal: confirm BigST variant and implement/test it. Not yet approved.
Controlled efficiency measurements and final comparison remain open. Historical
updates below retain the state and approval boundaries at their recorded dates;
older "pending" statements are superseded by this paragraph.

## Tentative work budget

These dates are planning targets, not runtime estimates or promises. Work remains
one approved milestone at a time.

| Dates (2026) | Target |
| --- | --- |
| Nov 8–10 | Review both architectures; resolve variants, source licenses, environment specification, comparison settings and GPU feasibility |
| Nov 11–14 | Implement/test PatchSTG and complete a short GPU training run |
| Nov 15–18 | Implement/test BigST and complete a short GPU training run |
| Nov 19–21 | Final accuracy/efficiency runs; additional seeds if measured time permits |
| Nov 22–23 | Analysis, plots, documentation and reproduction check |
| Nov 24 | Submission buffer |

## Architecture review completed November 11; implementation pending

Explain each paper's components, input/output shapes, source-code discrepancies
and required preprocessing. Review author-code licensing and freeze references
before adaptation. Resolve whether BigST includes the long-history extractor.
A short-history variant must be named and cannot be presented as the full method.
If full long history changes eligible origins, match origins across compared
models and rerun baselines as needed; preserve the earlier reports as historical
runs. Do not silently change the current benchmark.

For PatchSTG, resolve patch padding, similarity construction from training data,
index restoration and differences between paper dimensions and author configs.
Identify how to measure a forward/backward smoke test on the 8 GiB GPU. Numerical
GPU model feasibility and training throughput are still unknown.

The review is complete; guides and proposals are in model_implementation_plan.md.
The original Nov 8–10 review target slipped to Nov 11. Next propose PatchSTG
preprocessing/model tests, without implying the remaining dates are guaranteed.
BigST variant confirmation remains pending. No implementation/training approval
is implied by completing this review.
Adaptation of author code is acceptable in principle for the deadline, but code
reuse still requires source/license checks, attribution, tests and clear fidelity
labels. Learning deliverables include equation-to-code mappings and explanations.

## Experiment budget and fallback

Use identical target masks and windows wherever inputs permit. Disclose extra
history, coordinates, spatial priors and auxiliary fitting instead of calling all
input budgets identical. Record full preprocessing/auxiliary costs for BigST and
patch construction costs for PatchSTG. Freeze validation-based checkpoint selection
and a bounded tuning budget before comparative runs; never select using test scores.

Start with a fixed seed and small batch for smoke runs, then decide final seeds
from measured runtimes. Three seeds remain desirable, not a guaranteed deliverable;
report the actual seed count and avoid uncertainty claims from a single run.
Record failed/OOM configurations. No silent width/layer reductions to make a
paper configuration fit: explain and label any adaptation before comparative runs.

Protect Nov 22–24 for reporting. If either implementation is infeasible, explain
the concrete blocker and agree a revised scope; do not fabricate a complete
comparison or add substitute models without approval. Accuracy and correctness
come before optional tuning, ablations and additional plots.

## November 11 progress update

PatchSTG spatial preparation and full-width model implemented; 35 tests and a
real-data CPU forward/backward/reload check pass. See patchstg_implementation.md.
GPU model fit and training throughput remain unknown. Next propose the separately
approved host GPU feasibility milestone, then a bounded trainer. BigST variant
confirmation is still pending. Baseline result tables remain unchanged.

## GPU feasibility tooling update

GPU feasibility milestone approved; command/tests implemented. All 42 tests pass.
Host model run is pending because the agent cannot access CUDA. No memory/runtime
budget has been established. See patchstg_gpu_feasibility.md for the user command.
Training remains a separately approved milestone; BigST variant choice is pending.

## Host GPU results reviewed

Full-width batch 1/2/4 probes passed; batch 4 peak allocated/reserved was
455/514 MiB. See reports/patchstg_gpu_feasibility.md from the repository root.
Use batch 4 as initial harness candidate; measure steady-state throughput
before allocating epochs/seeds. Training-harness milestone awaits approval.

## November 12 training harness update

Trainer implemented with tests and a two-update full-model CPU integration check.
53 tests pass. Host diagnostic capped at 20 training and 20 validation batches
is the next approved action; docs/patchstg_training.md has the command (from the
repository root). Final training epochs/seeds require measured throughput and
a reviewed budget. No final PatchSTG accuracy experiment yet; BigST scope pending.

## Approved 50-total-epoch continuation

Seed-2023 PatchSTG continuation approved on November 12. Resume calibration
last.pt and train epochs 2–50 with identical non-epoch settings. Host launch
pending; no job started in the agent session. Approximate remaining cost five
hours. Test evaluation and BigST scope remain separate decisions.
