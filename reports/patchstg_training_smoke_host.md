# PatchSTG host training diagnostic

Completed November 12, 2026 on RTX 5060 Laptop GPU. Capped one-epoch diagnostic
with 20 training updates and 20 validation batches, batch 4, seed 2023. No empty
training batches skipped. Best/last checkpoint files were produced. No test data
evaluated. Implementation hashes match the current code.

| Measurement | Observed |
| --- | ---: |
| Training windows sampled | 80 |
| Validation windows evaluated | 80 |
| Training wall time | 12.003 s |
| First training step | 10.616 s |
| Mean step time after excluding first 5 updates | 0.06721 s |
| Validation wall time | 0.150 s |
| Peak allocated GPU memory | 472.60 MiB |
| Peak reserved GPU memory | 540.00 MiB |

Subset train MAE 267.2340; subset validation MAE
147.4188. These scores are diagnostic only and cannot be
compared against full-split baseline scores. Training MAE is online across
changing weights. This run establishes successful supervised updates and
validation execution, not convergence or final forecast quality.

## Preliminary runtime planning

The full training split has 21,001 windows, or 5,251 batches at batch 4.
Multiplying by the observed post-warm-up step mean gives 5.88 minutes
of step work. This excludes loading overhead, validation, checkpoint saving,
preprocessing and sustained thermal/resource changes. Only 15 post-warm-up steps
were timed, so this is a rough extrapolation, not a measured epoch duration.

A tentative planning range is 6–10 minutes per full training/validation epoch
under similar conditions, or roughly 5–8.5 hours for 50 epochs. These estimates
are not a promised runtime. Measure one full epoch before selecting a final
epoch/seed budget. A full-epoch pilot and final experiments require further
approval. The first-step startup cost should not be multiplied by every batch.

[Full diagnostic/provenance](patchstg_training_smoke_host.json) includes resolved
configuration, source/code hashes, installed versions, GPU, checkpoint selection
and step timings. The earlier exact checkpoint-resume test was CPU-based; this
host run does not independently verify bit-exact GPU resume.
