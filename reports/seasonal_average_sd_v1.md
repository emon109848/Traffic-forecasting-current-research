# Seasonal-average baseline — SD v1

Scope note: these completed baselines remain references for the BigST/PatchSTG
SD comparison due November 24, 2026. Recorded scores and original run provenance
are unchanged. See [roadmap](../docs/roadmap.md).

Measured locally on 2026-11-08. Independent educational baseline. Closed-form
training statistics only; no gradient optimization or test-set tuning.

For each weekday/15-minute slot/sensor, average nonzero readings from training
frames [0,21024). Unseen/all-zero groups fall back to the sensor nonzero training
mean; sensors with no nonzero training readings fall back to the global nonzero
training mean. Zero exclusion during fitting is a documented baseline choice,
distinct from the pipeline scaler, which includes zeros. All means remain in
original flow units. No smoothing, clipping, rounding or online updates.

Calendar assumes 2019-01-01 00:00, 15-minute regular samples, Monday=0, no timezone.
Future calendar information is known under that assumption; no future traffic
enters prediction. Sensor row/column pairing remains assumed.

| Baseline | Split | MAE | Pooled RMSE | MAPE (%) |
| --- | --- | ---: | ---: | ---: |
| Persistence | val | 64.1336 | 99.3846 | 41.5436 |
| Persistence | test | 60.7825 | 94.3350 | 41.7801 |
| Seasonal average | val | 19.7442 | 36.4470 | 12.4536 |
| Seasonal average | test | 28.0045 | 53.3906 | 19.9964 |

Both runs have the same data hashes, windows and scored target counts. Metrics
pool nonzero targets in original units; MAPE is percent without epsilon. Test
coverage is 96.7400%; validation coverage is 97.0882%. These are measurements
under this project protocol, not reproductions of published tables.

| Split | Step | Assumed minutes | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 3 | 45 | 19.7410 | 36.4439 | 12.4600 |
| val | 6 | 90 | 19.7424 | 36.4454 | 12.4548 |
| val | 12 | 180 | 19.7499 | 36.4516 | 12.4428 |
| test | 3 | 45 | 27.9958 | 53.3866 | 19.9909 |
| test | 6 | 90 | 28.0048 | 53.3912 | 19.9966 |
| test | 12 | 180 | 28.0167 | 53.3961 | 20.0049 |

## Fit diagnostics and interpretation

The fitted table has 481,152 weekday/slot/sensor cells; 2,719 use a fallback.
Four sensors have no nonzero readings anywhere in the training split and use
the global training mean. These sensors were not removed, and no held-out data
was used to fill their statistics. This does not establish the cause of zeros.

Seasonal averages outperform persistence on pooled validation/test metrics in
this run. They exploit recurring calendar structure over the training history;
persistence uses only the most recent reading. The comparison does not isolate
attention effects or establish performance on another dataset or time period.
Large persistence errors can reflect long leads, changing traffic and zero inputs;
their relative contributions have not been measured by this milestone.

## Reproduce

```bash
conda activate pml
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/evaluate_seasonal_average.py --output runs/seasonal_average_sd_repeat/results.json
```

[Full results and provenance](seasonal_average_sd_v1.json) contain all 12 horizons,
source and implementation hashes, configuration, software versions and Git dirty
status. The run used uncommitted implementation files; hashes identify them.
Fit wall time is recorded separately from evaluation. Neither time is a controlled
inference benchmark. No CUDA use, stochastic optimization, or seed-dependent
training occurred; seed is explicitly null. Exact environment locking is pending.

All 23 pipeline/baseline/evaluation tests passed. An independent weekday/slot
grouping calculation matched validation/test horizon-12 metrics. Source hashes,
window counts and valid-target counts matched the saved persistence run.

Source files remain unchanged. Missingness flags and original timestamps remain
unavailable. Calendar-dependent results rely on the approved temporal assumption.
