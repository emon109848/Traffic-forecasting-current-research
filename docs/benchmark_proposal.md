# First benchmark proposal

Status: proposed for review; no dataset acquired or protocol frozen.
Reviewed: 2026-11-08.

## Recommended task

Use LargeST-SD, 2019, all 716 sensors, flow forecasting at 15-minute cadence.
Use 12 history steps (three hours) to predict 12 future steps (three hours),
stride 1. Report all horizons, highlighting steps 3, 6, 12 (45, 90, 180 minutes).
This follows the setting described by the [PatchSTG paper](https://arxiv.org/html/2412.09972v2)
and [SD config](https://github.com/LMissher/PatchSTG/blob/main/config/SD.conf).
It is a shared benchmark proposal, not a promise to reproduce published scores.

## Data acquisition and identity — pending

The [LargeST repository](https://github.com/liuxu77/LargeST) links to Kaggle CA
files and describes deriving regional subsets. Prefer an auditable 2019 source
with metadata and graph. Author-provided PatchSTG SD files are an alternative
only after their provenance, sensor order, dates, preprocessing, and usage terms
are verified. Do not download all five years merely to obtain a one-year subset.
Whether selective acquisition is practical remains to be checked.

The [LargeST preparation notebook](https://github.com/liuxu77/LargeST/blob/main/data/ca/process_ca_his.ipynb)
resamples to 15-minute means, rounds, then fills missing values with zero.
Thus the published processed target is a rounded mean of source readings, not a
sum of vehicles over 15 minutes. Preserve source units and explicitly label this
aggregation. Retain missingness before filling whenever source data permits.
Exact bin alignment, timestamp timezone/DST behavior, duplicates and gaps remain
data-audit gates; do not invent timestamps for an unverified processed array.

## Split and window rules — proposed

For exactly T=35,040 verified ordered frames, split by target timestamps at
floor(0.6T)=21,024 and T-floor(0.2T)=28,032. Training targets have indices below
21,024; validation targets are [21,024,28,032); test targets are [28,032,T).
Every target in a 12-step window must fall within one split. Context may use
observations from earlier splits already available at the forecast origin.
Reject target windows crossing a boundary. Store exact timestamp boundaries in
the manifest once real timestamps are checked. This is an explicit project rule;
the [LargeST generation script](https://github.com/liuxu77/LargeST/blob/main/data/generate_data_for_training.py)
splits origin indices instead, so matching a 60/20/20 ratio is not exact equivalence.

## Normalization, missing data, and metrics — proposed

Fit one global scalar mean/std on valid training flow observations, normalize
only the traffic channel, and guard against zero variance. Validity excludes
nonfinite or source-flagged missing readings; zero alone is not evidence of
missingness. Define input imputation after the audit; prefer causal forward fill
with a training-only fallback and retain an input mask for diagnostics. No
interpolation using observations after a forecast origin. If source validity
cannot be recovered, state that limitation before choosing a masking policy.
These rules may differ from author preprocessing and must be labeled accordingly.

Compute original-unit MAE and pooled RMSE on valid targets, including valid zero
flow. MAPE: 100 * mean(abs(pred-target)/abs(target)) over valid targets with
abs(target)>=1 source flow unit. Report its coverage; this threshold is a project
proposal, not an inferred paper rule. Empty valid sets produce an explicit
undefined result, not zero. Aggregate by pooled valid observations and separately
report each horizon. If reporting mean-of-horizon metrics, label them separately
from pooled metrics. Never hide nonfinite predictions through masking.

## Training and efficiency — proposed

First validate persistence and training-only seasonal averages. Implement
STAEformer as the first attention model; see the compatibility review for why.
Use validation MAE to select checkpoints. Initial neural smoke runs use seed
2023; final comparison uses seeds 2023, 2024, 2025 with per-seed values and
mean/std. Fix a tuning budget before comparative runs. Optimizer, loss,
epoch budget and model configs are not selected by this planning milestone.

Start GPU checks with batch 1, FP32, no compilation. Compare latency at batch 1
and throughput at a common batch size that all models support. Separately report
largest feasible batches. Measure synchronized forward-only device timing and
end-to-end timing separately, using 20 warm-up iterations and 100 timed iterations
as an initial measurement plan. Report median/p95 latency, sample throughput,
peak allocated/reserved GPU memory, trainable/total parameters, epoch and total
training time, plus preprocessing/auxiliary costs. Hardware feasibility is pending.

## Storage and tests to implement later

Keep a source manifest with checksums, source URLs/versions, timestamps, sensor
IDs, graph/coordinate order, preprocessing revision, split indices, and mask
semantics. Store arrays once and generate windows lazily; no materialized
sliding-window copies. Add tests for target split isolation, known window values,
training-only scaler/imputation/graphs, valid zero targets, MAPE threshold and
empty masks, pooled metric arithmetic, and metadata/sensor alignment.

## Approval gates

Before data acquisition: confirm source accessibility, licensing, and expected
archive size. Before benchmark freeze: inspect actual data, missing-value semantics,
calendar, resampling and exact split counts. Before training: verify GPU access,
environment compatibility, baseline tests and model fidelity records.
The next major milestone should resolve the environment first; dataset work
remains separately approved.
