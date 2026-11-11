# PatchSTG host GPU feasibility check

Status: host GPU feasibility complete; batches 1, 2 and 4 passed. See
[measured results](../reports/patchstg_gpu_feasibility.md).
Milestone approved November 11, 2026. No training experiment is authorized here.

## Run from your WSL terminal

From the repository root:

```bash
conda activate pml
PYTHONPATH=src python scripts/check_patchstg_gpu.py --output-dir runs/patchstg_gpu_host_v1
```

Choose a new output directory if that path already exists. No package install is
needed. The first host report has been reviewed. For additional probes, share the
new report before changing the training batch-size recommendation. A spatial cache from
scripts/check_patchstg.py can optionally be passed with --cache; source/config
identity must match. Without --cache, training-only spatial maps are recomputed.

## What the command does

1. Verify CUDA availability. If unavailable, write an explicit unattempted report
   and exit 2 before data preparation. Never silently substitute CPU results.
2. Prepare the first four training windows and provenance-checked spatial maps
   once on CPU. Only x and input calendar are supplied to the GPU probe.
3. Launch a fresh worker process for batch sizes 1, 2 and 4, in that order. Every
   worker initializes a new full-width model, seed 2023, and its own CUDA context.
4. Run finite forward/backward checks and exactly one AdamW step per passing
   batch, then verify finite updated parameters and a changed checked parameter.
5. Stop on the first OOM, CUDA unavailability or other failure. Remaining sizes
   are explicitly not_attempted; generic errors are not relabeled as OOM.

The objective is mean squared normalized output, an artificial autograd probe
without targets. lr=0.002, weight_decay=0.0001, foreach=False. These are probe
settings, not the final training configuration. Probe weights and optimizer
state are discarded when the worker exits; no trained checkpoint is produced.
No forecast accuracy, validation tuning or full training loop is involved.

## Memory and timing scope

Workers use FP32, no AMP or compilation, with TF32 disabled and cuDNN benchmark
mode off. After CUDA context setup, reset peak counters before model/input
transfer. The recorded peak therefore includes model tensors, inputs, gradients,
attention/MLP intermediates and first-step AdamW state allocation. Fresh workers
keep a smaller batch's allocator/optimizer state out of the next batch's process.
Model-level finite checks are included in the measured workload.

Report both PyTorch [peak allocated memory](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html)
and [peak reserved memory](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_reserved.html).
Allocated memory represents tensor allocations; reserved memory includes the
allocator's managed pool. These are not total device usage or driver/context
memory. Device free/total memory is separately sampled after context setup.
[Peak counters](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.reset_peak_memory_stats.html)
are reset in each worker. Other processes may still affect available VRAM.

Timing uses CUDA synchronization around a single transfer/setup/probe sequence.
It has no warm-up or repeated steady-state samples and is not inference latency,
throughput, an epoch-time estimate, or evidence of training convergence. Passing
batch 4 means only this probe fits at that batch; it is not a maximum-batch search
or a guarantee of headroom for a later training harness.

## Artifacts and exit status

All outputs live under ignored runs/:

- report.json: aggregate status, batch summaries, full config, software/platform,
  Git revision/dirty status, implementation hashes, source hashes via spatial
  provenance, objective, precision and measurement scope.
- batch_1.json, batch_2.json, batch_4.json and logs: attempted worker reports.
- prepared.json, spatial_maps.npz, probe_inputs.npz: checksummed CPU inputs/maps.

GPU name, total VRAM and compute capability are recorded by attempted workers.
Exit 0 means all requested batches passed; 1 means partial success or another
failure; 2 means CUDA unavailable in the main process (or invalid CLI usage).
Failed workers record stage/error and memory counters when available. Optimizer
step count is unknown/null if a failure prevents confirming completion. Artifacts
are not overwritten; failed runs can leave partial files for inspection.

## Checks completed in this session

All 42 unit tests pass, including seven new feasibility tests. They cover one
CPU optimizer step, rejection of invalid outputs/missing gradients, OOM stopping,
malformed reports, unavailable-CUDA reporting and overwrite refusal. Existing
model/metric/data tests still pass. Real SD preparation and cache identity were
also checked independently on CPU.

The agent run at runs/patchstg_gpu_agent_v1/report.json exited 2 and reported
CUDA unavailable: no batches attempted and no memory measurements. The host's
earlier generic CUDA check passed, and the subsequent host report completed this model-specific milestone.
Implementation/source hashes were verified against the successful report. No packages or original source data changed.

## Host outcome

User ran the command and supplied the saved report. All three requested batch
sizes passed; curated results and exact provenance are under reports/. Batch 4
is the largest tested size and a conservative starting candidate for a later
trainer, not a maximum or throughput-optimized batch. No full training experiment
has run. The next proposed milestone is a bounded training harness with masked
loss, validation checkpointing and reproducible configuration.
