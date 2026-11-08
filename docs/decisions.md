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
