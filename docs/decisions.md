# Decision log

Entries preserve historical choices; later entries supersede earlier proposals.
Current scope is BigST/PatchSTG on SD 2019 by November 24, 2026 (roadmap.md).

## 2026-11-08 — Minimal foundation only

Status: accepted for milestone 1.

Use a `src/traffic_benchmark` package and `pyproject.toml` with setuptools.
Declare no runtime dependencies until environment compatibility is reviewed.
Do not create empty model/data/training abstractions before their requirements
are known. Directories will be added as their milestones need them.

Keep reusable logic in the package and future notebooks focused on exploration.
Keep raw data and full run artifacts out of Git; track curated reports later.

## 2026-11-08 — Reproduction and controlled comparison are separate

Status: methodological convention; exact protocols pending.

A paper-settings reproduction and a shared-protocol adaptation answer different
questions. Label each experiment and document changed inputs, preprocessing,
architecture, training settings, and evaluation rules. Do not assume published
scores across papers are directly comparable.

## 2026-11-08 — No environment or dataset commitment yet

Status: deferred.

LargeST-SD and the four named models are candidates. Python 3.14.7 is an
observation, not a chosen training version. Choose versions and benchmark scope
after hardware and primary-source compatibility review.

## 2026-11-08 — Use Conda for the project environment

Status: accepted; explicitly requested by the user.

Use a dedicated Conda environment. Choose Python and PyTorch/CUDA versions after
the environment audit, then record the environment specification and resolved
dependencies. Do not infer that the shell's Python version is the training choice.
No environment creation or package installation has been performed.

## 2026-11-08 — Milestone 2 review recommendations

Status: proposed; not permission to install or acquire data.

Recommend SD 2019 at 15-minute cadence, 12-to-12 forecasting and explicit target
split boundaries. Recommend STAEformer as the first attention implementation
after simple baselines. Keep full-history variants separate from short-history
comparisons. Recommend a dedicated Conda environment with Python 3.11 as a
candidate, retaining existing pml unchanged; exact PyTorch/CUDA selection awaits
GPU/driver details. See benchmark_proposal.md and environment_audit.md.

## 2026-11-08 — Validate existing pml before changing environments

Status: validation in progress; final environment choice pending.

The user approved environment validation. Keep existing pml unchanged and run
a numerical CPU/CUDA smoke check before deciding whether a new environment is
needed. CPU passed; agent GPU visibility remains blocked. Require a user-terminal
CUDA result and avoid pinning unrelated pml packages as project dependencies.

## 2026-11-08 — Retain pml after successful host validation

Status: accepted for current development.

User-terminal CPU and CUDA smoke checks passed with Python 3.14.8,
torch 2.14.1+cu130 and NumPy 2.5.4 on RTX 5060 Laptop GPU (compute capability
12.0). No environment replacement is currently necessary. Reassess compatibility
as data/model dependencies are introduced; this is not a final dependency lock.

## 2026-11-08 — Approved processed SD assumptions and zero-target exclusion

Status: accepted explicitly by user. Supersedes the initial include-zero-target
proposal for this processed dataset only.

Proceed with local author-processed SD data; preserve metadata row order and
assume author pairing with traffic columns. Treat published 2019 / 15-minute
calendar as an assumption because timestamps are absent. Exclude zero targets
from loss and metrics following the reference evaluation convention, documenting
that genuine zero flow cannot be distinguished from filled missing values.
Preserve originals. Input imputation, scaler zero inclusion, exact calendar
indexing and pipeline implementation are not approved by this decision.

## 2026-11-08 — Data pipeline implementation

Status: implemented within the approved data-pipeline milestone.

Preserve zero inputs; include them in global training-only scaler statistics,
following the reference loader. No imputation or frame deletion. Use strict target
split boundaries and lazy 12-to-12 windows. Reconstruct a timezone-naive regular
calendar from assumed 2019-01-01, Monday=0, explicitly adapting reference calendar
indexing. Return original-unit targets with separate zero-exclusion mask.
Use unittest with installed NumPy/PyTorch; no test dependency installation needed.

## 2026-11-08 — Masked evaluation and persistence

Status: implemented and evaluated within the approved milestone.

Use float64 error sums/counts, nonzero-target masking and original-unit metrics.
MAPE is percent without epsilon/additional threshold. Empty masks yield null;
nonfinite predictions fail even at masked positions. Persistence repeats raw
latest values, including zero, with no fitting. Evaluate validation/test without
tuning, and store implementation/source hashes and dirty Git status alongside
results. Full reports and curated tables are under reports/. No inference-speed
claim is derived from evaluation wall time.

## 2026-11-08 — Seasonal-average educational baseline

Status: implemented and evaluated in approved milestone.

Fit weekday/15-minute-slot/sensor means from nonzero training readings only.
Fallback is sensor nonzero training mean, then global nonzero training mean;
reject all-zero training data. No online updates or test tuning. Use the approved
assumed calendar and identical persistence windows/metrics. Source and run
provenance, fallback diagnostics and measured comparisons are stored in reports/.
This fitting rule does not change pipeline input-zero or normalization policies.

## 2026-11-08 — Narrow deadline scope to BigST and PatchSTG

Status: accepted; user requested all documentation reflect the narrowed project.

Deadline November 24, 2026. Retain processed SD 2019, shared pipeline, masks,
metrics, tests and measured persistence/seasonal baselines. Defer ASTGCN and
STAEformer, new model searches, extra datasets, broad tuning and scalability
trials. This supersedes STAEformer-first and multi-track model plans.

Next proposed milestone: review BigST/PatchSTG architecture and GPU feasibility,
resolve BigST long-history variant and common origins, and scope implementation.
Propose PatchSTG first, then BigST. Exact author-code adaptation is not yet
selected; license/source review and fidelity labels remain required.

Report SD accuracy/efficiency without larger-network scalability claims. Use
measured runtimes to freeze the seed/experiment budget; three seeds desirable
but not promised. Preserve Nov 22–24 for documentation and submission.
No neural implementation or training authorized by this documentation update.

## 2026-11-11 — Architecture review findings and pending proposals

Status: review completed; implementation/variant proposals await approval.

Recommend code-aligned PatchSTG SD dimensions and preserve 288-step similarity
segments initially, explicitly documenting the cadence discrepancy. Recommend
BigST short-history core without LSFE/spatial regularizer for the deadline,
with a variant label; full BigST remains a user choice. No silent method reduction.
Partial immutable-source checks and license observations recorded in
model_sources.json. PatchSTG MIT observed; BigST code reuse terms unresolved.
No vendored source, package installs, architecture code or hardware measurements.

## 2026-11-11 — PatchSTG implementation and CPU verification

Status: completed within explicitly approved milestone.

Implement the code-aligned SD path (one temporal patch, H=F), explicit one-head
attention/MLP and full reference SD widths. Preserve 288-frame similarity profiles;
use training-only scaler verification and integrity/provenance-checked maps.
Specify stable sensor-index tie handling. Keep registered index buffers and
normalized outputs. MIT notice retained in THIRD_PARTY_NOTICES.md.

35 tests pass, including numerical one-head attention checks, axis behavior,
map/cache invariants and state-dict reload. Real-data CPU check passed with
2,284,844 trainable parameters. No optimizer updates, training, GPU measurements
or forecast accuracy claimed. BigST scope remains pending; GPU feasibility next
requires a separately approved milestone.

## 2026-11-11 — Approved host GPU feasibility tooling

Status: implementation and CPU checks complete; host GPU run pending.

Use separate CUDA worker processes for batches 1/2/4; stop at first failed/OOM
batch. Record peak allocated/reserved memory including forward/backward and
initial AdamW state allocation. Disposable model, artificial squared-output
objective, no checkpoint/forecast metric or training run. Preserve source/config
hashes, precision, optimizer flags and failure outcomes. No invented GPU results
when agent visibility is blocked. No training-harness approval implied.

## 2026-11-11 — Host GPU feasibility passed

Status: complete; actual saved user-terminal report reviewed and hashes verified.

Full-width PatchSTG passes batch 1/2/4 including backward and one AdamW update.
Recommend tested batch 4 as a conservative trainer starting point. Larger sizes
and steady-state throughput remain unmeasured; no epoch-runtime inference from
one-off setup timings. Curate host JSON/table under reports/. No full training
is authorized by this result; the next harness milestone needs approval.

## 2026-11-12 — Bounded training harness and diagnostic configuration

Status: implemented/CPU verified within approved milestone; host diagnostic pending.

Use original-unit masked MAE; skip empty masks without AdamW updates, reject
invalid values, clip gradient norm 5. AdamW .002/.0001, foreach=False, epoch-end
schedule [1,35,40] gamma .5. Select by pooled validation MAE. No test evaluation.
Default supplied config is diagnostic: 1 epoch, batch 4, 20 train and 20 val
batches, seed 2023. No final experiment config or accuracy claim.

Record seed/software/source/config/code/hardware identity and installed version
inventory; the latter is not a full environment lock. Atomic best/last states
include epoch-boundary optimizer/scheduler/RNG/shuffle resume. Config identity
must match except extending total epochs. 53 tests pass and a two-batch full-model
CPU integration/reload check passed. Host timing is next; do not infer runtime
from earlier GPU setup probes. No dependencies installed.

## 2026-11-12 — Host supervised diagnostic passed

Status: capped host training validation complete.

20 updates/20 validation batches passed with batch 4, no skips and no test use.
Use post-warm-up timing (.06721 s/update) rather than first-step startup for
rough planning. Full epoch estimates remain extrapolations. Recommend a single
full-epoch calibration milestone before final budget selection; not yet approved.
Store subset results separately and never compare them to full baseline scores.

## 2026-11-12 — Approved one-full-epoch calibration

Status: configuration prepared; host execution pending.

Preserve model, optimizer/scheduler, batch 4 and seed 2023. Remove train/val
batch caps while keeping epochs=1. Start fresh, without resuming the capped
smoke model. Traverse full training/validation splits, save checkpoints and
measure actual runtime; no test evaluation or automatic continuation. Full
experiment budget remains a later reviewed decision. No source code changed.

## 2026-11-12 — Full-epoch calibration completed

Status: actual user-host report reviewed; code hashes verified.

5,251 updates, full validation, no skips, no test use. Training/validation time
350.97/15.11 s; actual combined epoch cost about 6.10 min excluding setup and
checkpoints. Validation MAE 25.2203 after one epoch, not final model accuracy.
Saved last.pt is eligible for a later approved epoch-budget extension; source
identity and non-epoch settings must remain unchanged. No continuation started.

## 2026-11-12 — Approved seed-2023 PatchSTG continuation to 50 total epochs

Status: config/checkpoint compatibility verified; host launch pending.

Resume the full calibration last.pt after epoch one, run epochs 2–50, batch 4
and all other settings unchanged. Best checkpoint chosen by full validation MAE;
no test evaluation or additional seed. Estimated remaining train/validation
time approximately five hours, not guaranteed. No training source changes or
packages installed. Agent did not launch a GPU job in its restricted session.

## 2026-11-13 — Frozen PatchSTG test evaluation completed

Status: approved implementation, all 58 tests passing, actual full CUDA test pass.

Verified 50-epoch training completion and validation-selected epoch-50 best.pt.
Preserved the training implementation and artifacts. Test evaluation verifies
original code/data/scaler/split identity and saved spatial maps before loading
frozen weights; shared pooled metrics score all windows and the partial batch.
No optimizer updates or test-based selection/tuning. Dataset/split/per-horizon
mask counts match both baselines. Reported single-seed SD adaptation accuracy:
MAE 17.1943, RMSE 29.1370, MAPE 11.1706%. Configuration stays fixed after test
inspection. No repeated run was used to select a score. A server interruption
left no saved report; no process remained before evaluation was restarted.

Scoring wall time includes batching, transfers and metrics and is not isolated
model latency. SD-only results do not demonstrate scalability. BigST variant and
implementation remain a separate, unapproved next proposal. No packages installed.
