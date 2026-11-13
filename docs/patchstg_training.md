# PatchSTG training harness

Current status (November 13): 50-total-epoch seed-2023 training is complete.
Best epoch 50 has validation MAE 14.9936. Frozen test evaluation is also complete:
MAE 17.1943, RMSE 29.1370, MAPE 11.1706%. See
[test results](../reports/patchstg_sd_seed2023_test_v1.md) and
[evaluation command](patchstg_evaluation.md). Earlier diagnostic/calibration
sections below are historical; pending-launch statements are superseded.


Implemented November 12, 2026. Harness tests and a two-batch real-data CPU
diagnostic passed. Host GPU capped training diagnostic also passed; see
[measured diagnostic](../reports/patchstg_training_smoke_host.md). This is not a
completed accuracy experiment or a reproduction of published PatchSTG results.

## Run the approved short diagnostic

From the project root in your own WSL terminal:

```bash
conda activate pml
PYTHONPATH=src python scripts/train_patchstg.py --config configs/patchstg_smoke.json --output-dir runs/patchstg_train_smoke_host_v1
```

The JSON configuration caps execution at **one diagnostic epoch, 20 training
batches and 20 validation batches**, batch 4, seed 2023. This samples at most 80
training and 80 validation windows. It does not process full epochs and never
evaluates test data. New output directories are required; CUDA unavailability
exits 2 with a report, without silently using CPU. Optional --cache accepts an
existing provenance-matching spatial cache; otherwise maps are reconstructed.

Share the epoch summary or report.json after the host run. We will use actual
post-warm-up step timings to plan the next experiment. No full-run configuration
or automatic final experiment is supplied by this milestone.

## Training and validation rules

| Component | Behavior |
| --- | --- |
| Architecture | Existing full-width SD PatchSTG; config defaults resolved into artifacts |
| Input | Existing normalized x and assumed input calendar; training data only for updates |
| Targets | Original flow units; never inverse-transform targets a second time |
| Loss | Inverse-transform model outputs, then mean absolute error on nonzero targets |
| Empty training mask | Clear gradients and skip update, including AdamW weight decay; count skips |
| Invalid numeric values | Fail on nonfinite predictions, targets, loss, gradients or updated weights |
| Optimizer | AdamW, lr 0.002, weight decay 0.0001, foreach=False |
| Gradient clipping | Global gradient norm 5, rejecting nonfinite norms |
| Scheduler | Epoch-end MultiStepLR at 1,35,40, gamma 0.5, after validation |
| Selection | Lowest pooled validation MAE; ties keep the earlier best |
| Metrics | Shared original-unit MAE/RMSE/percentage MAPE and valid-target counts |
| Train ordering | Seeded shuffled windows; num_workers=0; incomplete final batch retained |
| Validation ordering | Sequential windows; no gradient recording, parameter updates or shuffling |
| Precision | FP32; deterministic algorithms, TF32/cuDNN benchmark disabled; no AMP/compilation |

Optimizer, clipping and scheduler settings follow the reviewed
[pinned author entry point](https://github.com/LMissher/PatchSTG/blob/8b298b230227a304a082e05241c69dca3740a73d/main.py).
The diagnostic caps, shared split/calendar rules, explicit empty-mask handling,
pooled training logging and foreach=False are project choices. No author training
runner was copied. The first epoch uses lr 0.002; with this schedule the second
uses 0.001. Changing epoch caps to final experiment settings requires a reviewed
configuration and separate experiment approval, not an undocumented edit.

Training MAE is accumulated with valid-count weighting while parameters change
across updates. It is an online training diagnostic, not a fixed-model training-set
evaluation. Validation MAE pools all evaluated valid targets using the existing
metric accumulator. An entirely invalid training epoch or validation subset fails
rather than producing zero loss or selecting a checkpoint. Limits are implemented
without fetching an extra batch beyond the requested cap.

## Saved artifacts and recovery

All weights, caches and full logs stay in ignored runs/.

- resolved.json: model/run identity and resolved training configuration.
- provenance.json: seed, original config hash and explicit overrides, core software,
  observed installed Python packages, hardware/platform, Git revision and dirty
  status, parent checkpoint checksum when resuming. Written before epoch execution.
- report.json: completed/failed/interrupted status, diagnostic flag, epoch history,
  train/validation counts and metrics, timings and CUDA allocator peaks when used.
- best.pt: best-validation model state, epoch, validation MAE, scaler and identity;
  intended for later evaluation, not optimizer resume.
- last.pt: epoch-boundary model, optimizer, scheduler, best model, history, and
  Python/NumPy/PyTorch/CUDA/shuffle-generator RNG states.
- spatial_maps.npz: source/split/config-bound maps, saved after successful completion.

Source arrays and implementation files are SHA-256 identified. Caches retain
source/ordering/scaler/segment provenance. Observed package inventory is **not**
an exact environment lock and includes packages unrelated to this project.
A clean recreation specification remains pending before final experiments.

Checkpoint/report writes use a temporary file and atomic replacement inside the
new run directory. A failure preserves the last completed checkpoint if one exists.
It does not save a mid-batch state. If interrupted before the first checkpoint,
start a fresh run. Do not assume a report history entry alone guarantees a fully
written checkpoint; resume only from an actual last.pt file.

Resume uses --resume pointing to local last.pt, a new output directory, and a
config with a larger **total** epoch budget. All other training settings, source/
model/scaler/code identity and core software/device must match. For example,
resuming an epoch-one diagnostic requires the same diagnostic batch caps; it
cannot be silently promoted into full-dataset training. Best weights are retained
across resume even if no later epoch improves. Only trusted local checkpoints
should be loaded; the code uses weights_only=True and built-in/tensor states.

Exact CPU continuation was tested with dropout and shuffling. This is not a
promise of bitwise equality across hardware/software versions. Deterministic CUDA
operations and the CUBLAS_WORKSPACE_CONFIG setting may affect speed or raise an
explicit error if the environment cannot support the requested execution.

## Timing and resource interpretation

Each train-step timing starts after the DataLoader delivers the batch and covers
transfer, prediction, loss, backward, finite checks, clipping and optimizer update.
CUDA is synchronized around timed work. The report excludes the first five
successful updates from a separate mean step-time summary. With fewer than six
updates that summary is null. Epoch training wall time additionally includes
loading/iteration overhead; validation time is separate. Checkpoint writing and
preprocessing are not included in train-step timings.

A short subset can expose startup costs but does not establish full-epoch runtime,
convergence or inference latency. CUDA peaks cover the run's model/gradient/
optimizer allocations and validation; they are allocator counters, not whole-device
VRAM. Different loss paths can use more memory than the earlier artificial GPU
probe. Measure before increasing batch size or committing to final epochs/seeds.

## Verification completed

All **53 tests passed**, including 11 trainer tests. These verify known loss values
and inverse-scaling gradients, empty masks, skipped updates/weight decay, pooled
validation without updates, best-vs-last selection, exact epoch-boundary CPU
resume, identity/config rejection and failure metadata. Existing tests still pass.

Real SD CPU integration used --train-batches 2 --val-batches 2 with the same full
model and batch 4. It completed two updates and eight-window validation, saved
best/last states, and reloading best.pt reproduced diagnostic validation MAE.
[Curated diagnostic record](../reports/patchstg_trainer_cpu_diagnostic.json) is
explicitly a subset check and must not be compared to full-split baseline tables.
No test set was evaluated. GPU training here exited as unavailable without updates;
GPU feasibility from the earlier milestone remains valid, but the host training
command above must still be measured.

Run tests:

```bash
conda activate pml
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Host diagnostic reviewed

The approved 20-batch host run passed. Current next proposal is one full-split
epoch to measure actual epoch time; this requires approval before execution.
Do not rerun the diagnostic merely to tune its subset score. Final experiment
budgets and BigST variant remain unresolved.

## Approved one-full-epoch calibration (November 12)

The user approved one full training/validation epoch to calibrate runtime.
configs/patchstg_calibration.json preserves the smoke optimizer, scheduler,
batch 4, seed and architecture but sets both batch caps to null. Epochs remains
exactly 1. It starts a fresh seeded model; do not resume from the capped diagnostic
checkpoint because that used a different sampling budget.

```bash
conda activate pml
PYTHONPATH=src python scripts/train_patchstg.py --config configs/patchstg_calibration.json --output-dir runs/patchstg_calibration_host_v1
```

Expected traversal: 21,001 training windows (5,251 batches) and all 6,997
validation windows (1,750 batches). Empty-mask skipping remains active. The
training command prints its epoch summary after training and validation finish,
so a quiet terminal during the epoch is expected. Keep it open until completion.
The earlier 6–10-minute range is only a rough estimate; actual runtime may differ.

The run saves best/last checkpoints and measured timings. diagnostic_subset is
false because entire splits are traversed; this does not make a one-epoch model
a final experiment. No test data is evaluated. Review the report before setting
a final epoch/seed budget. A later approved continuation could use last.pt with
a larger total epoch budget and all other identity/config settings unchanged.
No such continuation is started automatically.

## Full-epoch outcome reviewed

The approved calibration completed 5,251 updates and full 6,997-window validation.
Training plus validation took about 6.10 minutes, excluding setup/checkpoints.
See [measured calibration](../reports/patchstg_calibration_v1.md). Multi-epoch
continuation is the next proposal and has not started. Use the full calibration's
last.pt if an extension is approved; the capped smoke run remains incompatible
with full-split settings. No test set has been evaluated for PatchSTG.

## Approved 50-total-epoch continuation

The user approved seed 2023 training through epoch 50.
configs/patchstg_sd_50epochs.json differs from the full calibration config only
in total epochs. The saved epoch-one last.pt was checked for configuration and
code identity compatibility. Run from the project root in the host pml terminal:

```bash
conda activate pml
PYTHONPATH=src python scripts/train_patchstg.py \
  --config configs/patchstg_sd_50epochs.json \
  --resume runs/patchstg_calibration_host_v1/last.pt \
  --output-dir runs/patchstg_sd_seed2023_50ep
```

This performs epochs 2–50: 49 additional epochs, not another 50. It preserves
model, optimizer/scheduler, best checkpoint and RNG/shuffle state. The first
continued epoch uses the restored scheduler learning rate. Rough remaining
runtime is five hours plus setup/checkpoint overhead, based on the measured
calibration. Keep the terminal open and the machine awake during execution.
An epoch summary is printed after each full training/validation pass.

The output directory must be new. best.pt can come from an earlier epoch; last.pt
is the resume point. If interrupted, resume from the new run's last.pt into
another new directory with the same total epoch budget of 50; already completed
epochs are not repeated. A change to code identity/config will be rejected.
No test evaluation, extra seed, final comparison claim or BigST training is
authorized by this command. Review the completed report before the next task.
