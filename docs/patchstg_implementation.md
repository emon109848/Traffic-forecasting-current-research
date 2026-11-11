# PatchSTG implementation milestone

Completed November 11, 2026. Code-aligned adapted implementation for the shared
SD protocol, not an official reproduction. No trained forecast results yet.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| [spatial.py](../src/traffic_benchmark/data/spatial.py) | Training-only similarity, coordinate partitions, padding, gather/restore maps and cache validation |
| [patchstg.py](../src/traffic_benchmark/models/patchstg.py) | SD model configuration, one-head attention, depth/breadth blocks and forecast projection |
| [check_patchstg.py](../scripts/check_patchstg.py) | Real-data CPU check, source/config provenance and spatial cache |
| [test_patchstg.py](../tests/test_patchstg.py) | Synthetic numerical, axis, leakage, cache and checkpoint tests |
| [third-party notices](../THIRD_PARTY_NOTICES.md) | Pinned PatchSTG reference and preserved MIT notice |

## Behavior implemented

Similarity uses the exact training interval and checks that supplied scaler
statistics match that interval. It uses normalized float64 training profiles
averaged over complete 288-frame segments, cosine similarity and the reviewed
exponential transform. Trailing incomplete segments would be excluded. At SD's
training boundary there are exactly 73 segments. Zero-norm profiles follow the
zero-cosine convention; zero cosine variance fails explicitly instead of inventing
a graph. Cosine is clipped to [-1,1] for roundoff. No held-out traffic is read.

The coordinate partition alternates latitude/longitude with balanced splits.
Padding uses the strongest similarity to a leaf member. Coordinate/score ties
are resolved by sensor index; this specifies previously unspecified reference
tie behavior. Each original sensor has one restoration position; duplicated
padding positions never become extra prediction targets.

Spatial caches are ignored local NPZ artifacts, loaded without pickle. Identity
includes both source hashes, training boundary, scaler, segment length, spatial
settings, tie policy, NumPy version and preprocessing-code hash. A payload hash
also protects maps, dtypes and patch dimensions. Loading with a different identity
fails; output/cache saves refuse overwrite. Cache checks are integrity checks,
not a verification of the unavailable original sensor-ID pairing.

The model implements the one-temporal-patch SD path only: history and horizon
are equal; it does not silently generalize to unequal horizons or multiple
patches in time. Config defaults use five blocks and width 192. Tests may use
tiny configurations, but the real-data check uses unchanged full SD dimensions.
All index maps are state-dict buffers that move with the model.

Model inputs are finite float32 normalized traffic and int64 calendar on the
same device. Shared (B,H,2) calendar is expanded across sensors; (B,H,N,2) is also
accepted. Output is normalized flow (B,H,N,1); inverse scaling belongs at the
original-unit loss/evaluation boundary. No y or future calendar is used.

## Attention replacement and fidelity limits

Explicit one-head scaled dot-product attention replaces the timm dependency.
The implementation preserves qkv/output biases, dropout locations, LayerNorm,
residual structure and GELU MLP ratio one. Defaults were checked against the
[timm v1.0.12 Attention source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.12/timm/models/vision_transformer.py)
and [MLP source](https://github.com/huggingface/pytorch-image-models/blob/v1.0.12/timm/layers/mlp.py).
No timm package or source was installed/vendored. It uses the explicit attention
path, not an automatically fused kernel; numerical roundoff and performance can
differ. Fixed-weight eval attention matches torch.nn.MultiheadAttention. This
is not a numerical comparison against the complete upstream model/checkpoints.
Initialization order is also not asserted to match upstream seed-for-seed.

Keep the reference 288-frame similarity convention and the code-aligned dimensions
as documented in [the architecture guide](papers/patchstg.md). Other adaptations:
shared calendar origin, stable ties, explicit guards, checkpointed maps and
integrity-aware caches. Preserve the original files; do not call this a full
paper-settings reproduction. Dataset terms remain unresolved.

## Measured CPU checks

All **35 tests passed** (existing tests plus 12 PatchSTG tests). Coverage includes
known index maps, train-only statistics, zero-norm profiles, degenerate inputs,
cache mismatch/corruption, attention axes, known embedding projection, numerical
attention parity, dropout, finite gradients and exact checkpoint reload.

The [real SD check report](../reports/patchstg_cpu_check.json) records:

- 716 originals, 1,024 padded positions; 16 patches of 64 positions.
- 308 padding positions; 73 similarity segments.
- 2,284,844 parameters, all trainable.
- Batch-one output shape (1,12,716,1).
- Finite predictions and parameter gradients; finite inverse scaling.
- Bit-exact evaluation output after local state-dict reload on the same CPU.

The scalar used for backward was an artificial squared-output probe. No optimizer
step, fitted model, training loss, forecast metric, CUDA execution or memory/speed
benchmark was performed. Tests do not establish forecast quality or GPU fit.
The report identifies dirty code through implementation hashes. Maps/checkpoints
are local artifacts; only the small check report is curated into Git.

## Reproduce

```bash
conda activate pml
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/check_patchstg.py --output-dir runs/patchstg_cpu_repeat
```

Optional reuse of an existing validated spatial cache:

```bash
PYTHONPATH=src python scripts/check_patchstg.py --output-dir runs/patchstg_cpu_cached --cache runs/patchstg_cpu_repeat/spatial_maps.npz
```

Use new output directories; failures may leave partial artifacts for inspection.
No packages were installed for this milestone.

## Next proposal — separate approval

Add a host GPU feasibility command for full-width batches 1, 2 and 4, including
forward/backward and one optimizer-step memory check. Record allocated/reserved
peaks and hardware, then scope a training harness from measured resource usage.
Do not begin full training before that step. BigST variant confirmation is still
pending; this PatchSTG implementation does not select the BigST method variant.
