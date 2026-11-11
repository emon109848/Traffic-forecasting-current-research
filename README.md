# BigST vs PatchSTG: Traffic Forecasting on LargeST-SD

A practical research and learning project comparing BigST (2024) and PatchSTG
(KDD 2025) for traffic forecasting on processed LargeST-SD 2019. We study their
architectures, prediction accuracy and computational efficiency on one fixed
network. Submission deadline: **November 24, 2026**.

The research question is how linearized spatial modeling and spatially patched
attention compare under a documented SD protocol. One network size supports an
efficiency comparison on SD; it does not establish large-network scalability.

## Current status

Foundation, compatibility review, environment validation, dataset inspection and
the reusable data pipeline are complete. Local SD data is excluded from Git.
Tested persistence and seasonal-average baselines with measured results are available. PatchSTG spatial/model components pass full-width CPU checks; no trained neural
results or training loop exist yet.

## Scope and roadmap

Active research models: **BigST and PatchSTG**. Keep the existing persistence and
seasonal-average baselines as reference results. ASTGCN, STAEformer, new model
searches, extra datasets, broad scalability trials and extensive ablations are
deferred beyond the deadline.

1. Completed: foundation, Conda/CUDA checks, SD audit, data pipeline and baselines.
2. Completed: focused architecture review; BigST variant confirmation is pending.
   PatchSTG preprocessing/model tests and real-data CPU check are complete.
   Next proposed: host GPU feasibility before a training harness.
3. Verify PatchSTG GPU feasibility; scope training and BigST implementation
   separately, subject to approval.
4. Run consistent accuracy and SD efficiency experiments within a measured budget.
5. Write the comparison, plots, limitations and reproduction instructions.

See the [deadline roadmap](docs/roadmap.md). Each major milestone requires approval;
this scope change does not authorize the next implementation or training phase.
Careful author-code adaptation with attribution is an option after license/source
review; exact implementation categories and deviations must be documented.

## Development

The source package uses a `src/` layout with NumPy and PyTorch dependencies.
Use Conda environment `pml`: the pipeline was checked with Python 3.14.8,
PyTorch 2.14.1+cu130 and NumPy 2.5.4. An exact environment lock remains pending.

A dependency-free scaffold check from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -c "import traffic_benchmark"
```

Use the existing `pml` Conda environment for current development. No packages
were changed during validation. A minimal `environment.yml` remains pending
a complete dependency-locking step; installed NumPy/PyTorch were reused.
Installation and training instructions will be added when validated. Git is initialized and a GitHub remote is configured. CPU and CUDA smoke checks passed in the user’s `pml` environment on an RTX
5060 Laptop GPU. This agent session has separate GPU access restrictions.

## Project records

Start with [the workflow learning guide](workflow.md) for what was built and why.

- [PatchSTG implementation and CPU checks](docs/patchstg_implementation.md)
- [Third-party attribution](THIRD_PARTY_NOTICES.md)
- [Architecture review and implementation plan](docs/model_implementation_plan.md)
- [PatchSTG architecture guide](docs/papers/patchstg.md)
- [BigST architecture guide](docs/papers/bigst.md)
- [Deadline scope and roadmap](docs/roadmap.md)
- [Seasonal-average comparison](reports/seasonal_average_sd_v1.md)
- [Persistence baseline results](reports/persistence_sd_v1.md)
- [Evaluation rules](docs/evaluation.md)
- [Data pipeline and test commands](docs/data_pipeline.md)
- [Conda validation instructions](docs/environment_validation.md)
- [Environment audit](docs/environment_audit.md)
- [Model compatibility review](docs/model_compatibility.md)
- [First benchmark proposal](docs/benchmark_proposal.md)
- [Current status and next task](docs/project_status.md)
- [Experimental protocol outline](docs/protocol.md)
- [Decision log](docs/decisions.md)
- [Paper implementation record template](docs/papers/TEMPLATE.md)
- [Coding-agent workflow](AGENTS.md)

## Reference starting points

- [BigST paper](https://www.vldb.org/pvldb/vol17/p1081-han.pdf)
  and [author repository](https://github.com/usail-hkust/BigST)
- [PatchSTG paper](https://arxiv.org/abs/2412.09972)
  and [author repository](https://github.com/LMissher/PatchSTG)
- [LargeST paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ee57cd73a76bd927ffca3dda1dc3b9d4-Paper-Datasets_and_Benchmarks.pdf)

Reference code has not been copied. Review licenses and record source commits
before adapting any external code. A project license remains to be chosen.
