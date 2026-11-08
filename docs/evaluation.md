# Masked evaluation and persistence

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
noise. It evaluates validation/test without tuning. No seasonal baseline or
attention model is implemented in this milestone.

See [measured baseline report](../reports/persistence_sd_v1.md) and
[data pipeline](data_pipeline.md). Tests verify known values, masked zeros,
empty/no-update states, batch aggregation, nonfinite rejection, small positive
MAPE denominators and latest-reading persistence. Real-data horizon checks were
also independently recomputed directly from the source array.
