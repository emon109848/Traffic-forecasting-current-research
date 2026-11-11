# PatchSTG host GPU feasibility results

Measured on the user’s RTX 5060 Laptop GPU on November 11, 2026. All requested
batch sizes passed full-width FP32 forward/backward and one disposable AdamW
update. This establishes execution for this probe, not forecast accuracy.

| Batch | Peak allocated (MiB) | Peak reserved (MiB) | Result |
| ---: | ---: | ---: | --- |
| 1 | 168.93 | 198.00 | Passed |
| 2 | 268.40 | 282.00 | Passed |
| 4 | 455.00 | 514.00 | Passed |

Model: 2,284,844 parameters, five layers, width 192, 16 patches of 64 positions.
Seed 2023; no AMP/compilation; TF32 disabled. Each batch ran in a fresh process.
Finite predictions, gradients and updated parameters were verified.

Use batch 4 as a conservative initial training-harness candidate because it was
tested. Larger batches remain untested. The later harness should measure warmed-up
step/epoch throughput before choosing its final batch and experiment budget.

Reported one-off wall times include setup and have no warm-up. Their differences
across batches are not evidence that larger batches are faster; do not estimate
epoch runtime or inference latency from them. Allocator peaks exclude non-PyTorch
driver/context allocations and do not guarantee whole-run memory headroom.

[Full report](patchstg_gpu_feasibility.json) records configuration, software, GPU,
source/implementation hashes and dirty Git status. Current implementation hashes
were checked against the report before curating it. The artificial objective
used no forecast targets; no checkpoint or trained result was produced.

See [GPU command and measurement scope](../docs/patchstg_gpu_feasibility.md).
