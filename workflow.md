# Project workflow: what we built and why

This is a learning guide to the completed work, not a plan to build every remaining
phase at once. Current scope is BigST vs PatchSTG on SD 2019, due November 24,
2026; ASTGCN/STAEformer and extra datasets are deferred. It explains the current implementation and the evidence behind
our decisions. Update it as approved milestones are completed.

## 1. Clarify the research question before implementing models

The goal is to compare attention-based traffic forecasting models for accuracy,
efficiency on SD while understanding their papers. Scalability was part of the
original ambition and is now deferred. We began by
reviewing ASTGCN, STAEformer, BigST, and PatchSTG rather than assuming their
published scores could be compared directly.

Their inputs and experimental settings differ. Full ASTGCN uses recent, daily,
and weekly histories, while its author's PyTorch release contains the recent
branch only. BigST includes a long-history feature extractor. PatchSTG requires
coordinates and spatial patch management. STAEformer uses adaptive embeddings
and attention along temporal and spatial axes.

**Why:** changing inputs or removing components changes what is being evaluated.
A controlled comparison on SD is an adapted evaluation, not automatically a
reproduction of a paper's original experiment. Our code so far is independently
written infrastructure and educational baselines; no research model has been
implemented or reproduced yet.

See [model compatibility review](docs/model_compatibility.md).

## 2. Establish a small repository foundation

We created a Python package under `src/traffic_benchmark/`, package metadata,
ignore rules, and documentation. Each area was added when its milestone needed
it rather than creating a large framework in advance.

| Location | Purpose |
| --- | --- |
| `src/traffic_benchmark/data/` | Loading, normalization, forecast windows |
| `src/traffic_benchmark/baselines/` | Persistence and seasonal-average predictors |
| `src/traffic_benchmark/evaluation/` | Shared streaming metrics |
| `scripts/` | Commands that use the reusable package |
| `tests/` | Synthetic behavioral tests |
| `docs/` | Decisions, protocol, audits, status and explanations |
| `reports/` | Curated measured results and provenance |
| `data/` | Local dataset files; ignored by Git |
| `runs/` | Full local run outputs; ignored by Git |

**Why:** reusable logic belongs in the package so tests, command-line scripts,
and future notebooks use the same behavior. Raw data and generated artifacts
should not inflate the Git repository.

[AGENTS.md](AGENTS.md) records the incremental workflow for future coding sessions.
We explain significant changes, complete one approved milestone, verify it,
update records, and wait for approval before the next major milestone.

## 3. Check the actual development environment

You selected Conda. We inspected the existing `pml` environment and created
[scripts/check_environment.py](scripts/check_environment.py). It checks known
matrix calculations, gradients, a small optimizer update, NumPy interoperability,
and CUDA execution when required.

CPU checks passed inside the agent session. GPU access was blocked there, but
you ran the same check in your terminal and supplied a successful CUDA result:

- Python 3.14.8; PyTorch 2.14.1+cu130; NumPy 2.5.4.
- NVIDIA GeForce RTX 5060 Laptop GPU, approximately 8 GiB VRAM.
- Compute capability 12.0.

We retained `pml` without installing or replacing packages. An earlier suggestion
to consider Python 3.11 remained an option, not a downgrade requirement.

**Why:** a restriction in the agent session does not prove your host GPU is
broken. Testing computation is more useful than inferring compatibility from
an installed toolkit or `nvidia-smi` alone. These checks establish basic operation,
not compatibility with every future model. An exact environment lock is still
unfinished; the environment includes packages unrelated to this project.

See [environment validation](docs/environment_validation.md).

## 4. Acquire and inspect SD before changing its values

You downloaded `flowsd.npz` from the PatchSTG author's linked data folder and
`sd_meta.csv` from the author's GitHub repository. Agent-side network restrictions
prevented downloading these files directly. We inspected your local copies and
recorded SHA-256 checksums, which identify their exact contents.

| Check | Observation |
| --- | --- |
| Traffic shape | `(35040, 716, 1)` |
| Meaning of axes | Time frames, sensors, traffic channel |
| Stored type | `float64` |
| Invalid numeric values | No NaN, infinity, or negative readings |
| Zero readings | 563,841, approximately 2.25% |
| Entirely zero frames | Four consecutive frames: indices 6536–6539 |
| Metadata | 716 unique sensor IDs, no empty fields |
| Coordinates | Valid, with no duplicate coordinate pairs |
| Region | District 11, San Diego |

**Why:** checking dimensions alone would miss invalid values, missing information,
or metadata problems. We kept the originals unchanged and recorded findings
before making preprocessing decisions.

See the [acquisition report](docs/datasets/sd_acquisition.md),
[array audit](docs/datasets/sd_array_audit.json), and
[metadata audit](docs/datasets/sd_metadata_audit.json).

## 5. Make missing information explicit

The NPZ contains traffic values only. It has no timestamps, sensor IDs, or original
missing-value flags. Therefore, we agreed to proceed with these assumptions:

1. Metadata row order corresponds to traffic column order. The matching sensor
   counts support structural consistency, but do not independently prove alignment.
2. The data follows the published 2019 setting with regular 15-minute intervals.
   Calendar features are reconstructed from assumed January 1, 2019 at midnight,
   Monday encoded as 0, without timezone or DST adjustment.
3. Zero targets are excluded from loss/evaluation by convention. Genuine zero flow
   cannot be distinguished from missing readings that were filled with zero.

**Why:** the processed dataset is a practical starting point, but we must not
present assumptions as verified observations. The published preparation averages
readings into 15-minute bins; the forecast target is not a sum over those bins.

Our initial draft proposed including genuine zero targets. We revised that policy
explicitly when inspection showed the validity information was unavailable.
Inputs still retain zeros. No imputation or removal of all-zero frames occurs.
A differentiable training loss has not yet been implemented.

Hosted dataset usage terms remain unresolved. Keeping data outside Git does not
resolve its licensing; we have not established redistribution permission.

## 6. Build a pipeline that respects forecasting time

[The pipeline](src/traffic_benchmark/data/pipeline.py) loads and validates both
files, preserves sensor order, and constructs windows on demand.

Each sample uses the previous 12 steps to predict the next 12. At the assumed
15-minute cadence, that means three hours of history and three hours of forecasts.

For example, with first target index `t`, inputs are `[t-12, t)` and targets are
`[t, t+12)`. Every input precedes the first target.

| Split | Target frame interval | Eligible forecast windows |
| --- | --- | ---: |
| Training | `[0, 21024)` | 21,001 |
| Validation | `[21024, 28032)` | 6,997 |
| Test | `[28032, 35040)` | 6,997 |

Intervals include the left boundary and exclude the right. Every target in a
window must belong to one split. Validation/test inputs may use earlier observed
frames as context, but target windows cannot cross a split boundary.

**Why:** random splitting could expose future traffic patterns to training.
Strict target boundaries also prevent the same realized target timestamp from
belonging to two splits. This rule is explicit and may differ from a reference
repository that splits forecast origins instead.

The scaler computes one global mean and population standard deviation from
training frames only, including zeros. This follows the reference loader's
normalization convention. Validation/test values cannot change those statistics.
Zero variance is rejected.

Samples return normalized `x`, original-unit `y`, a separate boolean
`target_mask`, calendar indices, and the first target index. Sample arrays are
float32; scaler statistics are computed with float64 precision. Future models
will adapt their input encodings without changing the shared sample definition.

**Why:** keeping targets in original units makes metric interpretation clear.
A separate mask avoids confusing a normalized value of zero with a missing target.
Lazy windows avoid storing thousands of overlapping copies; the underlying NPZ
is still loaded into RAM, so this is not disk-streaming infrastructure.

See [pipeline explanation and commands](docs/data_pipeline.md).

## 7. Verify evaluation with known answers

We implemented MAE, RMSE, and MAPE in
[the streaming metric accumulator](src/traffic_benchmark/evaluation/metrics.py).
For valid nonzero targets, with error equal to prediction minus target:

- MAE: mean absolute error.
- RMSE: square root of mean squared error.
- MAPE: mean absolute error divided by target, multiplied by 100.

No epsilon or extra threshold is applied to MAPE. Scores use original flow units,
except MAPE, which is a percentage. Predictions are not clipped.

The accumulator adds error sums and valid counts across batches, then divides
at the end. Overall RMSE is computed from pooled squared errors, not by averaging
batch or horizon RMSE values. All 12 horizons are also reported individually.

**Why:** averaging batch scores gives the wrong weighting when batch sizes or
valid-target counts differ. Pooled sums/counts avoid that problem.

Nonfinite predictions fail even at zero targets. A completely empty valid set
returns `null` in JSON, not a perfect score of zero. Each report includes target
counts and coverage so the amount excluded is visible.

See [evaluation rules](docs/evaluation.md).

## 8. Establish persistence as the simplest reference

[Persistence](src/traffic_benchmark/baselines/persistence.py) repeats each sensor's
latest observed reading across all 12 future steps. It retains a latest reading
of zero. It has no fitted parameters and uses no randomness.

**Why:** this answers whether recent conditions alone are enough, and exercises
forecast indexing and metrics before adding learned models. The evaluation reads
the latest raw value using the pipeline's window indices, avoiding unnecessary
normalization/inversion rounding.

We evaluated validation and test sets without tuning on them. Independent direct
array calculations matched the reported scores at steps 1, 3, 6, and 12.

See [persistence results](reports/persistence_sd_v1.md).

## 9. Establish a seasonal-average reference

[SeasonalAverage](src/traffic_benchmark/baselines/seasonal_average.py) stores the
average nonzero training flow for each weekday, 15-minute slot, and sensor.
For a future calendar slot, it looks up that mean rather than using future traffic.

When a group has no nonzero training readings, it uses the sensor's nonzero
training mean. If that sensor has none either, it uses the global nonzero training
mean. All-zero training data is rejected because no fallback can be estimated.

**Why:** this baseline tests how much recurring calendar structure explains.
It complements persistence, which follows recent conditions. Excluding zero
readings during seasonal fitting is a documented baseline choice; it does not
change the scaler's include-zero policy.

Four SD sensors had no nonzero training readings. The global fallback handled
them without consulting validation or test values. No online updating, gradient
optimization, or hyperparameter tuning was performed.

An independent weekday/slot grouping calculation matched the horizon-12 scores.
Both baselines used identical source hashes, forecast windows, and target masks.

## 10. Record measured results and their limits

These are measured pooled test scores under our project protocol:

| Baseline | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Persistence | 60.7825 | 94.3350 | 41.7801 |
| Seasonal average | 28.0045 | 53.3906 | 19.9964 |

Approximately 96.74% of test targets are scored after excluding zeros. Overlapping
windows score the same observation at different forecast leads; those entries
are not independent samples for uncertainty estimation.

Seasonal averages perform better on these pooled metrics. That does not isolate
why persistence errors are large, prove generalization to another period, or
say anything yet about attention architectures. These are not published-paper
reproductions.

Reports record configuration, software versions, dataset checksums, code revision,
dirty Git status, implementation hashes, and all horizon scores. The runs used
uncommitted implementation files, so hashes supplement their recorded Git revision.
Seed is explicitly absent because neither baseline uses stochastic operations.

Fit/evaluation wall times are recorded with their scope. They are not controlled
inference-speed or GPU-memory benchmarks.

See [the comparison and full provenance](reports/seasonal_average_sd_v1.md).

## 11. Test behavior before trusting results

The current suite has 23 passing tests using synthetic data. It checks:

- Split boundaries, known windows, previous-split context, and array shapes.
- Training-only scaling and rejection of malformed inputs or metadata.
- Zero-target masks, unchanged zero inputs, and calendar rollover.
- Known metric values, pooled aggregation, empty masks, and invalid predictions.
- Persistence's latest-value behavior.
- Seasonal means, fallback cases, and invariance to changes in held-out data.

**Why:** small examples have answers we can calculate independently. They reveal
errors that realistic-looking dataset scores could hide. Real-file smoke checks
and independent score calculations add evidence beyond those unit tests.

Run the suite from the project root:

```bash
conda activate pml
PYTHONPATH=src python -m unittest discover -s tests -v
```

The standard-library test runner avoids installing another dependency.

## 12. Reproduce the completed checks

```bash
conda activate pml
python scripts/check_environment.py --require-cuda
PYTHONPATH=src python scripts/inspect_pipeline.py
PYTHONPATH=src python scripts/evaluate_persistence.py --output runs/persistence_repeat/results.json
PYTHONPATH=src python scripts/evaluate_seasonal_average.py --output runs/seasonal_repeat/results.json
```

Choose fresh output paths if those reports already exist. Evaluation scripts
refuse to overwrite them. GPU validation may fail inside the restricted agent
session even though it passed in your terminal. Baseline evaluations run on CPU.

Git staging/commit attempts inside the agent session failed because `.git` was
read-only. You ran commit/push commands locally. The agent did not successfully
commit or push those changes; check Git status for current pending work.

## 13. What remains before attention experiments

The proposed next milestone is a focused BigST/PatchSTG architecture and
feasibility review. This replaces the earlier STAEformer-first plan. Propose
PatchSTG implementation first, then BigST, after resolving its long-history
variant. See [deadline roadmap](docs/roadmap.md). Implementation and training
remain independently approved milestones.
No attention model, differentiable training loss, neural training loop, or
GPU performance benchmark exists yet.

Before expensive experiments, we still need an exact environment specification,
model fidelity records, training/checkpoint/tuning rules, and reproducible run
configuration. Dataset assumptions and source-terms limitations remain documented
rather than silently treated as resolved.

Consult [project status](docs/project_status.md) to resume work, and
[the decision log](docs/decisions.md) for the history behind current choices.

## 14. Narrow the scope for the November 24 deadline

After completing the shared pipeline and baselines, we limited research-model
work to BigST and PatchSTG. Everything already built supports that comparison;
no ASTGCN/STAEformer model code existed to discard. We updated the active roadmap
and preserved historical notes as history.

**Why:** two architectures on one audited dataset give a focused question and
leave time for implementation checks, measured experiments and clear reporting.
Large-network scalability, extra models/datasets, broad tuning and optional
ablations are deferred. The tentative schedule reserves Nov 22–24 for writing,
reproduction checks and submission. It is not a training-runtime guarantee.

Use author references carefully and label adaptations. Full BigST long-history
costs and fair origin eligibility still need review. No complete research-model
comparison or GPU memory feasibility is claimed yet.
