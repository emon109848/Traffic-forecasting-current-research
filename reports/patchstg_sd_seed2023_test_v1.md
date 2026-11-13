# PatchSTG SD test results — seed 2023

Completed November 13, 2026. Actual full test evaluation of frozen `best.pt`.
Code-aligned adaptation under our shared protocol, not an official reproduction.

## Accuracy

| Model | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Persistence | 60.7825 | 94.3350 | 41.7801 |
| Seasonal average | 28.0045 | 53.3906 | 19.9964 |
| PatchSTG, seed 2023 | 17.1943 | 29.1370 | 11.1706 |

PatchSTG reduces pooled MAE by 38.60% relative to seasonal average.
Both baseline reports match dataset hashes, split boundaries, window counts and
all per-horizon valid/total target counts. No baseline was rerun or retuned.

## Forecast horizons

| Lead | Minutes | MAE | RMSE | MAPE (%) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15 | 11.8080 | 18.9561 | 7.3839 |
| 2 | 30 | 13.4866 | 22.2640 | 8.3001 |
| 3 | 45 | 14.6465 | 24.4130 | 9.0353 |
| 4 | 60 | 15.6450 | 26.1569 | 9.7127 |
| 5 | 75 | 16.5619 | 27.7292 | 10.3570 |
| 6 | 90 | 17.3399 | 28.9936 | 10.9371 |
| 7 | 105 | 18.0584 | 30.2283 | 11.4706 |
| 8 | 120 | 18.6190 | 31.2163 | 11.9152 |
| 9 | 135 | 19.2063 | 32.2355 | 12.5871 |
| 10 | 150 | 19.8116 | 33.2276 | 13.5110 |
| 11 | 165 | 20.3081 | 34.1732 | 14.1655 |
| 12 | 180 | 20.8403 | 35.2660 | 14.6722 |

## Selection and reproducibility

Epoch 50 was selected by lowest full validation pooled MAE (14.9936), before
examining the test scores. Training continued the full calibration checkpoint
for 50 total epochs, batch 4, seed 2023. Training code and configuration were
not modified. This evaluation did not perform optimizer updates or clipping.

All 6,997 test windows were scored: 58,158,396 valid targets
out of 60,118,224 (96.7400%). Zeros are excluded;
MAPE is percent with no epsilon. History/forecast are 12 frames each (3 hours).
Dataset calendar and sensor-order assumptions remain documented limitations.

Train plus validation across all 50 epochs: 4.6948 hours, excluding
setup/checkpoint writes. Parameters: 2,284,844. GPU: RTX 5060 Laptop GPU.

Test scoring pass: 17.174 seconds. Peak allocated/reserved:
71.02/92.00 MiB.
These describe one pass including batching/transfers/metrics, with no warmup;
do not compare them to isolated model latency or differently scoped baseline timings.

Only one seed was trained, so variability is unknown. These results establish
SD accuracy for this adaptation, not full paper reproduction, BigST superiority,
or scalability. Do not retune this configuration using its test results.

- [Curated test metrics and provenance](patchstg_sd_seed2023_test_v1.json)
- [Training completion summary](patchstg_sd_seed2023_training.json)
- [Evaluation command and integrity checks](../docs/patchstg_evaluation.md)
- Full local artifacts: `runs/patchstg_sd_seed2023_test_v1/report.json` and
  `runs/patchstg_sd_seed2023_50ep/` (ignored by Git).
