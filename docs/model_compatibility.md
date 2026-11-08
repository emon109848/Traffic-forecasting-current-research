# Initial model compatibility review

Reviewed: 2026-11-08. This is a planning review, not completed equation-by-equation
verification. No author code was copied or executed. References are moving branch
URLs unless a paper version is specified; freeze source commits before adapting.
Licenses for all incorporated code and dataset artifacts must be checked then.

| Model | Core mechanism | Inputs / prerequisites | Implication for SD comparison |
| --- | --- | --- | --- |
| ASTGCN (2019) | Spatial/temporal attention with Chebyshev graph convolution and temporal convolution | Graph; full paper uses recent, daily, weekly histories | Full ASTGCN has more history than a recent-only benchmark; label ASTGCN-r explicitly if used |
| STAEformer (2023) | Adaptive spatio-temporal embeddings and temporal/spatial dot-product attention | Traffic, time-of-day/day-of-week; learned embeddings indexed by sensor and history position; no predefined adjacency required | A useful first research implementation; SD is a new setting and dense spatial attention needs memory measurement |
| BigST (2024) | Long-sequence feature extraction plus linearized global spatial convolution | Short and long/periodic histories; preprocessing/training for long features; spatial prior in paper | SD flow differs from original speed datasets; full and short-history versions are distinct experiments |
| PatchSTG (KDD 2025) | Irregular spatial patches and alternating depth/breadth attention | Traffic/calendar/node embeddings, coordinates, similarity-based padding and sensor index mappings | Best initial protocol anchor because author SD configuration exists; padding and index restoration need dedicated tests |

## ASTGCN fidelity issue

The [paper](https://guoshnbjtu.github.io/pdfs/AAAI2019-GuoS.2690.pdf) models recent,
daily, and weekly dependencies. The [author PyTorch README](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch)
explicitly says the release contains only the recent component. Its PEMS04/08
setup and default zero masking must not silently become our full-ASTGCN claim.
Full-history comparisons require common eligible forecast origins, accounting
for the longest lookback in every model. The graph construction and Laplacian
conventions must also be recorded.

## STAEformer implementation priority

The [paper, arXiv v5](https://arxiv.org/html/2308.10425v5) describes feature,
periodic, and adaptive embeddings with attention along both axes. The
[author model](https://github.com/XDZhelheim/STAEformer/blob/main/model/STAEformer.py)
provides a compact reference for tracing shapes and attention behavior.
Recommend implementing it first after persistence/seasonal baselines and metric
verification: it introduces attention without graph convolution or patch-index
machinery. An independent implementation on SD is an adapted evaluation, not an
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
constructs calendar indices from row numbers. Our calendar features should be
based on validated timestamps; a different weekday origin is a documented change.
The [utilities](https://github.com/LMissher/PatchSTG/blob/main/lib/utils.py) use
training data to construct similarity if no cached adjacency exists, but reuse
an existing cache without checking its provenance. Cache identity must include
training split and sensor order. Coordinate partitioning uses static metadata;
similarity fitting must not use validation/test traffic.

## Proposed comparison tracks

1. Shared short-history benchmark: same origins, 12-to-12 task, same calendar
   availability and metrics; retain required graph/coordinate inputs and disclose
   their unequal information. Label ASTGCN-r and BigST variants accurately.
2. Architecture-complete track: retain full periodic/long-history components,
   match eligible origins across models, and report added history and costs.
3. Paper-settings reproduction: separate runs with original dataset/protocol,
   only when feasible. Never compare their scores as a single controlled table.

Model selection is provisional. Add newer papers only after this pipeline works.
