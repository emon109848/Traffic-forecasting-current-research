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
