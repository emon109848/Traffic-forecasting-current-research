# Persistence baseline — SD v1

Scope note: this completed baseline remains a reference for the BigST/PatchSTG
SD comparison due November 24, 2026. Recorded scores and original run provenance
are unchanged. See [roadmap](../docs/roadmap.md).

Measured locally on 2026-11-08. Independent educational baseline; no training.

Persistence repeats the latest observed flow for all 12 future steps, including
zero inputs. Evaluation excludes zero targets. MAE/RMSE use original source
flow units; MAPE is a percentage with no epsilon or additional threshold.
All finite positive targets are included. The processed data contains integer
values, so its smallest positive denominator is at least one.

| Split | MAE | Pooled RMSE | MAPE (%) | Valid targets (%) |
| --- | ---: | ---: | ---: | ---: |
| val | 64.1336 | 99.3846 | 41.5436 | 97.0882 |
| test | 60.7825 | 94.3350 | 41.7801 | 96.7400 |

| Split | Step | Assumed lead (minutes) | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 3 | 45 | 35.3556 | 53.0339 | 20.5501 |
| val | 6 | 90 | 61.0580 | 89.1405 | 37.4695 |
| val | 12 | 180 | 107.3122 | 148.0311 | 76.3545 |
| test | 3 | 45 | 33.6057 | 50.9561 | 20.7282 |
| test | 6 | 90 | 57.7801 | 84.8702 | 37.6462 |
| test | 12 | 180 | 101.7560 | 140.0991 | 76.6705 |

Overall scores pool all valid forecast-origin/lead/sensor observations, not
batch means or horizon RMSE values. Overlapping windows intentionally score the
same realized observation at different forecast origins/leads. These are not
independent samples for uncertainty estimates. Empty masks yield JSON null.

Reproduce from the project root:

```bash
conda activate pml
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/evaluate_persistence.py --output runs/persistence_sd_repeat/results.json
```

The script refuses to overwrite output. It uses the pipeline loader and window
indices, but reads the last raw observation directly to avoid normalization
round-trip errors. It does not inspect future traffic to make predictions.
No parameters are fitted or hyperparameters selected from test results.

[Full measured report](persistence_sd_v1.json) includes all 12 horizons, source
checksums, exact config, Python/NumPy/PyTorch versions, platform, Git revision
and dirty status, plus hashes of implementation files. This run used uncommitted
implementation code; the hashes identify those files. No experiment lock exists
yet, and CPU model information may be empty when platform detection is limited.
No randomness is used; seed is explicitly null.

Recorded wall times cover CPU indexing, prediction and metric accumulation.
They exclude loading/setup and are not hardware-controlled inference benchmarks.
No GPU timings, memory benchmark or training-time measurements are claimed.

Sensor pairing and calendar are author-artifact assumptions. Original validity
flags are unavailable: zero masking may exclude genuine zero flow. Latest zero
inputs are retained and can hurt persistence accuracy when they encode missing
readings. Do not directly compare this table to a paper using a different split,
history, mask or preprocessing. No architecture conclusions follow from this run.
