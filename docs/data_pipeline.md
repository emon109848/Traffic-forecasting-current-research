# Processed SD data pipeline

This shared pipeline is retained for the BigST/PatchSTG comparison on SD 2019
(deadline November 24, 2026). BigST long-history support and common-origin changes
are not implemented; review them before changing current windows. See
[roadmap](roadmap.md).

This is independently written project infrastructure, not copied author code.
No forecasting baseline, training loss or metric implementation is included yet.

## Decisions implemented

- Load flowsd.npz with allow_pickle=False; require (35040,716,1), finite,
  nonnegative data. Reject malformed data instead of silently imputing.
- Preserve metadata row order. Require unique IDs, matching sensor count,
  District 11 and valid coordinates. Metadata pairing remains an author-artifact
  assumption; checks cannot establish column identity without embedded IDs.
- Target intervals: train [0,21024), validation [21024,28032), test [28032,35040).
  Every forecast window's targets belong to exactly one split. Earlier split
  observations may supply context. No shuffle occurs in inspection.
- History=12, horizon=12, stride=1. The first target follows its history. Windows
  are generated on demand, not stored as one expanded sliding-window array.
- Fit one float64 population mean/std on all training frames, including zeros,
  matching the reference loader's global-statistic convention. Reject zero std.
- Preserve input zeros and all-zero frames: no filling, deletion, or interpolation.
- Normalize only x, casting windows to float32. Keep y in original units (float32).
  Provide target_mask=(y != 0), separately from inputs. This is a benchmark mask,
  not recovered source validity. Future losses/metrics must consume the mask.
- Calendar assumes a regular sequence from 2019-01-01 00:00 every 15 minutes,
  with no timezone or DST correction. Return slot-of-day 0–95 and weekday
  Monday=0. These are reconstructed assumptions, not verified timestamps.
  This weekday origin differs from reference row-counter calendars that start at
  zero; it is an explicit project adaptation. No future traffic enters features.

## Sample contract

TrafficWindows returns x (history,N,1), y (horizon,N,1), boolean target_mask,
x_calendar (history,2), y_calendar (horizon,2), and target_start (first target's
array index). PyTorch DataLoader collates these into tensors with a batch axis.
Calendar features stay separate so each model can adapt its expected encoding.
Future y_calendar describes known calendar only; it is not an extra traffic input.
Samples are copies so edits cannot mutate source observations or other windows.
The loader marks the original traffic array read-only. Arrays are loaded once
per caller and shared across datasets; lazy windows do not imply disk streaming.
The NPZ uses about 191 MiB in memory. Training shuffling/seeds belong to a later
training milestone. No worker count, GPU transfer or persistent cache is imposed.

## Run in pml

```bash
conda activate pml
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/inspect_pipeline.py
```

Optionally save a report with --output runs/pipeline_check.json; existing output
files are not overwritten. Source files are never written. The committed
[real-data check](datasets/sd_pipeline_check.json) records hashes, scaler values,
splits and shapes; these are data checks, not forecasting results.

Observed windows: train 21,001; validation 6,997; test 6,997. Scaler mean
247.19353447084683, population std 184.27468320611501. All original hashes match
the acquisition audits. Tests use synthetic observations and require no dataset.
CPU checks passed in pml; no new dependencies were installed. Package metadata
now declares NumPy/PyTorch; an exact reproducible environment lock is still pending.

## Remaining limits

Hosted data terms, true timestamps, source validity and independent sensor-order
verification remain unavailable. No graph or model-specific preprocessing exists.
Masked losses/metrics must later reject invalid predictions and explicitly handle
empty masks; they must never interpret an empty valid set as perfect accuracy.
