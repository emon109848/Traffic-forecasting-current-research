# BigST architecture and variant review

Reviewed 2026-11-11. Documentation only. The proposed deadline variant needs user
confirmation; full BigST is not interchangeable with the short-history core.

## Intuition and paper mapping

The [VLDB paper](https://www.vldb.org/pvldb/vol17/p1081-han.pdf), sections 3.2–3.3,
combines a long-sequence feature extractor (LSFE) with linearized global spatial
convolution. LSFE models temporal context and periodic history; its auxiliary
training/cached representations add a distinct pipeline. The spatial core uses
positive random features to avoid explicitly storing dense all-sensor attention.
A compact kernel description is Z = Q′(K′ᵀV)/(Q′ sum(K′)), with row-wise division.
This is an approximate kernel computation, not exact softmax attention.

```mermaid
flowchart LR
    A[Recent traffic and calendar] --> B[Signal and node-time embeddings]
    B --> C[Linearized global spatial blocks]
    C --> D[Concatenate initial and final features]
    L[Optional long-history extractor and periodic features] --> E[Forecast head]
    D --> E
```

## Short-history core: source-derived shape trace

The [model reference](https://github.com/usail-hkust/BigST/blob/main/model.py) uses
node and last-input calendar embeddings for spatial query/key projections.
With candidate widths 32 and three blocks, signal and three embeddings concatenate
to 128 channels. Initial/final features concatenate to 256 before the short-history
head. Each spatial block gates projected values with sigmoid, applies dropout,
then random-feature kernel mixing. Residuals are optional; use_bn actually selects
LayerNorm. A proposed SD adapter is:

| Boundary | Shape/encoding |
| --- | --- |
| Pipeline traffic + calendar | (B,12,716,1) and (B,12,2) |
| Core input | (B,716,12,3): normalized flow, slot/96, weekday integer |
| Embedded values; random query/key features | (B,128,716,1); (B,716,1,64) |
| Core output; benchmark output | (B,716,12); (B,12,716,1) |

The code couples input-projection width to output length; our equal H/F avoids
that mismatch. Its random matrix is regenerated from a query-derived seed and
resets global RNG; changing to a fixed projection changes behavior. Numerics,
RNG isolation and projection policy require explicit tests before implementation.

## Candidate settings, output units and optional paths

The [pinned runner](https://github.com/usail-hkust/BigST/blob/830cb529354c0cda75c1660d1a0af2ffc7e66abb/run.py)
sets hidden/node/time widths 32, random-feature width 64, temperature 0.25 and
three layers. The README's short-history command enables residuals and use_bn;
use_long and use_spatial are optional flags. For SD propose 716 nodes and 96
slots/day. Long mode loads a separately trained extractor and adds sampled
periodic features. Its [pretraining entry point](https://github.com/usail-hkust/BigST/blob/main/preprocess/preprocess.py)
defaults to 2016 history steps: at 15 minutes, this would span 21 days rather than
the original five-minute week's duration. Never transplant this number blindly.

The [trainer](https://github.com/usail-hkust/BigST/blob/main/trainer.py) uses Adam,
original-unit masked MAE after inverse scaling, gradient clipping and an optional
spatial-prior penalty. Our pipeline already supplies original-unit targets:
invert model outputs only, not targets a second time. Keep pooled project
metrics rather than copying the runner's averages of batch metrics.

## Recommended bounded comparison — pending approval

Label the candidate **BigST short-history variant (no LSFE, no spatial-prior
regularizer)**. Retain its linearized spatial core and calendar/node embeddings,
use the shared 12-step history, and leave the existing forecast origins intact.
This studies spatial mechanisms under a bounded input history; it cannot support
claims about the complete BigST method. PatchSTG still receives coordinates for
partitioning, so disclose this unequal metadata access.

If full BigST is selected instead, first scope LSFE pretraining, history duration,
causal feature caches, spatial prior provenance, additional cost and matched
origins. Re-evaluate baselines on revised origins if needed. No such change is
approved or implemented by this review.

## Source reuse constraint

No LICENSE file was found in the reviewed root listing; the direct LICENSE URL
returned 404. Record code-reuse terms as unresolved, not permissively licensed.
Propose an independent paper implementation rather than vendoring repository
code without established reuse terms. This does not block understanding the
published method. See [source record](../model_sources.json) for what was read
and which immutable-file checks could not be completed.
