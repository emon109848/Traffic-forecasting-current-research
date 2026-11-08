# First benchmark proposal

Status: processed-data assumptions and zero-target policy approved; remaining
pipeline and experiment settings are proposals. Local artifacts acquired.
Reviewed: 2026-11-08.

## Recommended task

Use LargeST-SD, 2019, all 716 sensors, flow forecasting at 15-minute cadence.
Use 12 history steps (three hours) to predict 12 future steps (three hours),
stride 1. Report all horizons, highlighting steps 3, 6, 12 (45, 90, 180 minutes).
This follows the setting described by the [PatchSTG paper](https://arxiv.org/html/2412.09972v2)
and [SD config](https://github.com/LMissher/PatchSTG/blob/main/config/SD.conf).
It is a shared benchmark proposal, not a promise to reproduce published scores.

## Approved processed-data assumptions

The user approved continuing with author-provided flowsd.npz and sd_meta.csv
rather than obtaining original timestamp-bearing data. Local inspections and
checksums are recorded in docs/datasets/. Preserve metadata row order and assume
it corresponds to NPZ sensor columns, following the author artifact pairing.
The NPZ has no sensor IDs, so independent alignment verification is unavailable.

Use the published 2019 / 15-minute setting as an explicit temporal assumption.
The NPZ has no timestamps; exact start time, gaps, timezone and DST cannot be
independently checked. Any reconstructed calendar must be marked assumed, not
observed. Exact calendar indexing will be specified before implementation.

The [LargeST notebook](https://github.com/liuxu77/LargeST/blob/main/data/ca/process_ca_his.ipynb)
averages source readings into 15-minute bins, rounds and fills NaNs with zero.
Preserve the downloaded artifacts. No source-validity mask can be recovered from
this NPZ. Hosted data terms remain unresolved; data stays outside Git.

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

## Approved zero-target policy

Exclude zero targets from training loss and evaluation metrics, following the
reference author evaluation convention. This revises the earlier proposal to
include genuine zero flow: the processed NPZ cannot distinguish genuine zeros
from missing readings replaced by zero. This is an operational benchmark mask,
not a claim that every zero is missing. Preserve original values and report the
excluded-target fraction. The current array is finite and nonnegative.

Reject or explicitly handle nonfinite targets/predictions; never hide invalid
predictions by extending the target mask. Empty valid sets must produce a clear
undefined/skip outcome, not a perfect zero score. Define training-batch behavior
for empty masks during implementation planning.

## Remaining preprocessing and metric proposals

Normalization statistics must use training observations only. Whether statistics
include zero inputs, and whether/how input zeros are imputed, are not selected
by the zero-target masking decision. No automatic forward filling or deletion
of all-zero frames is approved. Resolve these details before the data pipeline.

Propose original-unit MAE, pooled RMSE and percentage MAPE on the nonzero valid
target mask. Exact near-zero handling, aggregation, scaling guards and tests
remain to be specified. Report each horizon and mask coverage; label a mean of
horizon metrics separately from pooled metrics.

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
training-only scaler/imputation/graphs, approved zero-target exclusion, MAPE threshold and
empty masks, pooled metric arithmetic, and metadata/sensor alignment.

## Approval gates

Before data acquisition: confirm source accessibility, licensing, and expected
archive size. Before benchmark freeze: inspect actual data, missing-value semantics,
calendar, resampling and exact split counts. Before training: verify GPU access,
environment compatibility, baseline tests and model fidelity records.
Environment validation and local dataset inspection are complete. Data pipeline
implementation remains a separate milestone requiring approval.
