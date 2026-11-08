# Model compatibility: active BigST/PatchSTG scope

Reviewed: 2026-11-08. Deadline: November 24, 2026.

Active models are BigST and PatchSTG. ASTGCN/STAEformer notes below are archived
context, not implementation priorities. See [roadmap](roadmap.md).

 This is a planning review, not completed equation-by-equation
verification. No author code was copied or executed. References are moving branch
URLs unless a paper version is specified; freeze source commits before adapting.
Licenses for all incorporated code and dataset artifacts must be checked then.

| Model | Core mechanism | Inputs / prerequisites | Implication for SD comparison |
| --- | --- | --- | --- |
| ASTGCN (2019) | Spatial/temporal attention with Chebyshev graph convolution and temporal convolution | Graph; full paper uses recent, daily, weekly histories | Deferred beyond deadline; full and recent-only variants differ |
| STAEformer (2023) | Adaptive spatio-temporal embeddings and temporal/spatial dot-product attention | Traffic, time-of-day/day-of-week; learned embeddings indexed by sensor and history position; no predefined adjacency required | Deferred beyond deadline; SD would require an adapted evaluation |
| BigST (2024) | Long-sequence feature extraction plus linearized global spatial convolution | Short and long/periodic histories; preprocessing/training for long features; spatial prior in paper | SD flow differs from original speed datasets; full and short-history versions are distinct experiments |
| PatchSTG (KDD 2025) | Irregular spatial patches and alternating depth/breadth attention | Traffic/calendar/node embeddings, coordinates, similarity-based padding and sensor index mappings | Best initial protocol anchor because author SD configuration exists; padding and index restoration need dedicated tests |

## Archived: ASTGCN fidelity issue (deferred)

The [paper](https://guoshnbjtu.github.io/pdfs/AAAI2019-GuoS.2690.pdf) models recent,
daily, and weekly dependencies. The [author PyTorch README](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch)
explicitly says the release contains only the recent component. Its PEMS04/08
setup and default zero masking must not silently become our full-ASTGCN claim.
Full-history comparisons require common eligible forecast origins, accounting
for the longest lookback in every model. The graph construction and Laplacian
conventions must also be recorded.

## Archived: STAEformer review (deferred)

The [paper, arXiv v5](https://arxiv.org/html/2308.10425v5) describes feature,
periodic, and adaptive embeddings with attention along both axes. The
[author model](https://github.com/XDZhelheim/STAEformer/blob/main/model/STAEformer.py)
provides a compact reference for tracing shapes and attention behavior.
The earlier recommendation to implement it first is superseded by the
BigST/PatchSTG deadline scope. No STAEformer implementation was started. An independent implementation on SD is an adapted evaluation, not an
official reproduction. Existing sensor embeddings also mean cross-network
transfer must not be assumed without retraining or a documented adaptation.

## BigST cost and information budget

The [VLDB paper](https://www.vldb.org/pvldb/vol17/p1081-han.pdf) evaluates five-minute
speed series on California/Beijing with 7:1:2 splits and a one-hour output. It
includes a long-sequence extractor and precomputed representations. The
[author repository](https://github.com/usail-hkust/BigST) exposes an optional
use_long path and describes itself as refactored. Treat long-history features,
auxiliary training, and cache generation as explicit costs; verify that features
at each origin use only available past observations. A 12-step-only variant does
not reproduce the full method. The README contains an impossible June 31 date,
so exact original dates require artifact inspection, not literal copying.

## PatchSTG protocol and code discrepancies

The [paper, arXiv v2](https://arxiv.org/html/2412.09972v2) reports 2019 data,
35,040 frames, 716 SD sensors, chronological 6:2:2 splits and 12-to-12 prediction.
The [SD configuration](https://github.com/LMissher/PatchSTG/blob/main/config/SD.conf)
sets 96 daily slots, batch 16, five layers, and seed 4096. This gives a
15-minute cadence and three-hour horizon, not a five-minute one-hour forecast.
Paper embedding dimensions and SD config values differ; resolve these in the
model milestone rather than silently picking one.

The [loader](https://github.com/LMissher/PatchSTG/blob/main/lib/data_loader.py)
uses a global training mean/std, permits previous-split history as context, and
constructs calendar indices from row numbers. Our NPZ lacks timestamps, so the approved pipeline reconstructs an explicitly
assumed 2019 calendar with Monday=0; this origin is a documented adaptation.
The [utilities](https://github.com/LMissher/PatchSTG/blob/main/lib/utils.py) use
training data to construct similarity if no cached adjacency exists, but reuse
an existing cache without checking its provenance. Cache identity must include
training split and sensor order. Coordinate partitioning uses static metadata;
similarity fitting must not use validation/test traffic.

## Deadline comparison plan

Compare BigST and PatchSTG on SD with the existing baselines. Resolve the BigST
variant before implementation. If short-history only, label it explicitly; if
using the full long-history extractor, disclose extra information and cost,
match eligible origins and revisit baseline reports when necessary. Do not
commit to separate reproduction tracks or extra datasets before the deadline.

Proposed implementation order: PatchSTG first, then BigST. This replaces the
STAEformer-first recommendation. The next milestone is a focused architecture
and feasibility review, with implementation/training separately approved. See
[deadline roadmap](roadmap.md). No large-network scalability claims from SD alone.
