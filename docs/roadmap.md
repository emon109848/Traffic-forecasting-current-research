# Deadline scope and roadmap

Updated 2026-11-08. Deadline: November 24, 2026. This replaces the initial
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

## Next proposed milestone: architecture and feasibility review

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

The review is the next proposal, not approval for implementation or training.
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
