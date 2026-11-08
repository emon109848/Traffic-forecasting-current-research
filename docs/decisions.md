# Decision log

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
