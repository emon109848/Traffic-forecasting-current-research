# Experimental protocol outline

Status: general protocol checklist. Processed-SD assumptions and zero-target
masking are approved in benchmark_proposal.md; other settings remain proposed.
No experiments have been run.

## Experiment identity

Separate paper-settings reproduction from shared-protocol comparison. Record
implementation category, paper version, source commit, and all deviations.

## Dataset contract

Specify source, license/access terms, version, checksums, sensor IDs/order,
coordinates/graph provenance, target variable and units, dates, timezone,
sampling interval, and any resampling. Audit missing values and outages; do not
infer that every zero is missing. For processed SD, zero-target exclusion is an
approved benchmark convention because original validity flags are unavailable.
Record available metadata and calendar inputs.

## Temporal protocol and leakage

Specify input history, forecast horizon, stride, chronological split timestamps,
and window eligibility. Assign forecast targets to their split and define
whether earlier observed history can provide context across a boundary.
Fit normalization, imputation statistics, similarity graphs, and other learned
preprocessing using training observations only. Never impute from future data
unavailable at the forecast origin. Preserve a separate target-validity mask.

## Model information and fairness

Record graph/coordinate inputs, calendar features, periodic history, auxiliary
training, preprocessing, and parameter count. Explicitly flag unequal information
budgets. Define validation-based checkpoint selection, tuning budget, early
stopping, optimizer, loss, batch size, seed set, and precision. Do not tune on test
results. Document exceptions rather than silently changing model semantics.

## Accuracy

Specify MAE, RMSE, and MAPE formulas, original-unit inverse transformation,
valid-target masks, near-zero MAPE threshold, and behavior for empty valid sets.
Define aggregation over sensors, samples, and horizons, including whether RMSE
is computed from pooled squared errors. Report per-horizon and aggregate scores,
seed variability, and valid-target counts/coverage.

## Efficiency and scaling

Record hardware, software, batch size, input/output shapes, precision, warm-up,
CUDA synchronization, repetitions, and timing scope. Separate training time per
epoch, total training time, inference latency, throughput, and preprocessing or
auxiliary training costs. Record peak allocated/reserved GPU memory and parameter
counts with explicit counting conventions. Compare fixed workloads and separately
report maximum feasible workloads. Preserve out-of-memory outcomes.

Define sensor subsets and graph/patch reconstruction before scalability trials;
node count alone does not control geography or topology.

## Reproducibility and interpretation

Save resolved configurations, seeds, dependency versions, dataset fingerprints,
code revision and dirty status, checkpoints, and machine-readable metrics.
Report deterministic-mode choices and their performance trade-offs. Use ablations
before attributing accuracy changes to a particular attention mechanism. Make no
claims for datasets, seeds, or workloads that were not evaluated.
