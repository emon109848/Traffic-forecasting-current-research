# PatchSTG one-epoch calibration

Completed November 12, 2026 on the user's RTX 5060 Laptop GPU. Fresh seeded model,
full training/validation splits, batch 4, seed 2023, FP32. This is one completed
epoch, not a final multi-epoch model or a published-paper reproduction.

| Measurement | Actual result |
| --- | ---: |
| Training updates | 5251 |
| Empty batches skipped | 0 |
| Training wall time | 350.97 s |
| Full validation windows | 6997 |
| Validation wall time | 15.11 s |
| Training plus validation | 366.09 s (6.10 min) |
| Validation MAE | 25.2203 |
| Validation RMSE | 36.1422 |
| Validation MAPE (%) | 24.5854 |
| Peak allocated memory | 472.60 MiB |
| Peak reserved memory | 584.00 MiB |

Training MAE 32.8170 is online across changing weights, not a fixed-model
training-set evaluation. Validation scores cover the full split with the same
valid-target count as the saved baselines. Zero masking and assumed sensor/calendar
provenance remain unchanged. No test data was evaluated. Both best.pt and last.pt
were produced; source-code hashes match the report.

The measured training-plus-validation time excludes preprocessing, checkpoint
writes and command setup. At similar sustained conditions, 49 more epochs would
require approximately 4.98 hours of training/validation, plus those
costs. This is an extrapolation from one full epoch, not a promised runtime or
an assertion that 50 epochs is optimal. Final epoch/seed budget needs approval.

The saved last.pt can support an approved continuation with larger total epochs
and otherwise identical configuration/identity. Do not restart the capped smoke
model. No continuation has been launched.

[Full report and provenance](patchstg_calibration_v1.json). Current result is a
validation-stage calibration; keep final test evaluation separate from model
selection and do not advertise this as the final BigST/PatchSTG comparison.
