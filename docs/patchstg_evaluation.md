# Frozen PatchSTG test evaluation

This milestone evaluates the validation-selected model from the completed seed
2023, 50-total-epoch run. It does not train, tune, refit on validation, select a
checkpoint using test scores, or modify training artifacts. This remains a
code-aligned adaptation of PatchSTG under our shared protocol, not an official
paper reproduction.

```bash
conda activate pml
PYTHONPATH=src python scripts/evaluate_patchstg.py \
  --run-dir runs/patchstg_sd_seed2023_50ep \
  --output runs/patchstg_sd_seed2023_test_v1/report.json
```

The CLI defaults to CUDA, refuses an existing output, and has no diagnostic
subset option. It loads `best.pt` with `weights_only=True` on CPU, verifies the
completed non-diagnostic training report and first minimum validation MAE,
and checks dataset hashes, splits, training-only scaler, original training
implementation hashes, and spatial-cache provenance/checksum. Spatial buffers
must match the saved maps before strict state loading. Model parameters are
frozen and dropout is disabled with `eval()`; inference runs without gradients.
Only the full test dataset is passed to the shared evaluation loop.

The test interval is [28032,35040), with 6,997 stride-one windows, 12 input
frames and 12 forecast frames. The last partial batch is included. Predictions
are inverse-scaled once into source flow units. The existing float64 streaming
metrics exclude zero targets, score finite negative predictions without clipping,
and compute pooled MAE, pooled RMSE and MAPE percent without an epsilon.
Per-horizon results cover 15 through 180 minutes. Dataset calendar, sensor-order,
and missingness assumptions remain those in the approved data protocol.

The JSON records checkpoint/report/map checksums, original training identity,
evaluation implementation checksums, code revision/dirty status, software,
hardware, seed, configuration, parameter count and counts per horizon. Timing
covers the entire scoring pass including data loading, transfers and metrics;
it excludes setup and is not an isolated inference-speed benchmark. Peak GPU
allocation/reservation is recorded for this pass. Single-seed accuracy does not
establish variability, and SD efficiency does not demonstrate scalability.

Curated results and comparison: [test report](../reports/patchstg_sd_seed2023_test_v1.md).
Keep the selected architecture/training configuration fixed after examining test
scores. BigST implementation remains a separate milestone requiring approval.
