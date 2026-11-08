# Masked evaluation and persistence

Current deadline scope: BigST vs PatchSTG on SD 2019, November 24, 2026. Both
simple baselines remain reference results. Model-specific training and controlled
GPU timing/memory measurements are pending; see [roadmap](roadmap.md).

Independent infrastructure: no author code copied. Metrics accept finite arrays
of shape (batch,horizon,sensors,channels), cast to float64 and accumulate error
sums and valid counts. Targets must be nonnegative. Zero targets are excluded;
no mask is inferred from prediction values. Nonfinite predictions are rejected
even at excluded targets. Negative finite predictions are scored as supplied;
no clipping or smoothing is performed.

MAE = sum(abs(error))/count. RMSE = sqrt(sum(error squared)/count).
MAPE (%) = 100*sum(abs(error)/target)/count for nonzero targets. No epsilon or
extra near-zero threshold. This implements the accepted author-style nonzero
mask; the older proposed threshold is superseded. Exact small positive targets
are scored, including in synthetic tests. Empty counts yield None/JSON null,
not zero. These are evaluation metrics, not differentiable training losses.

The accumulator reports each horizon and pooled overall metrics, along with
valid/total counts and valid fractions. Batch partitioning cannot reweight the
result; float64 reductions may introduce tiny roundoff differences.

Persistence takes original-unit history and repeats its last value across all
leads. It has no learned parameters and preserves zero inputs. The CLI uses
TrafficWindows target indices and raw last readings to avoid z-score inversion
noise. It evaluates validation/test without tuning. Seasonal averages are implemented in the subsequent approved baseline milestone;
no attention model exists yet.

See [measured baseline report](../reports/persistence_sd_v1.md) and
[data pipeline](data_pipeline.md). Tests verify known values, masked zeros,
empty/no-update states, batch aggregation, nonfinite rejection, small positive
MAPE denominators and latest-reading persistence. Real-data horizon checks were
also independently recomputed directly from the source array.

## Seasonal-average baseline

SeasonalAverage fits nonzero training flow by weekday, slot and sensor. It reads
only values/calendar before train_end. Missing group means use sensor nonzero
training means, then the global nonzero training mean for sensors with no valid
training values. All-zero training data raises an error. No observations after
the training boundary can alter the table or its fallbacks. predict accepts only
forecast calendar; it cannot inspect validation/test traffic.

Run scripts/evaluate_seasonal_average.py with --output pointing to a new report.
The CLI uses the same pipeline window indices and metrics as persistence, saves
fit diagnostics and provenance, and refuses output overwrite. Tests cover
closed-form means, zero exclusion, unseen slots/sensors, global fallback and
held-out changes. See [measured comparison](../reports/seasonal_average_sd_v1.md).
