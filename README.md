# Comparative Study of Attention-Based Traffic Forecasting Models

A practical research and learning project comparing attention-based models for
spatio-temporal traffic forecasting. The aim is to understand research papers,
build traceable implementations, and evaluate accuracy, computational efficiency,
and scalability under explicit experimental conditions.

## Current status

Foundation, compatibility review, environment validation, dataset inspection and
the reusable data pipeline are complete. Local SD data is excluded from Git.
No forecasting models, training loop, or accuracy results exist yet.

Initial model candidates are ASTGCN (2019), STAEformer (2023), BigST (2024), and
PatchSTG (KDD 2025). Inclusion and implementation order remain provisional.
LargeST-SD is a dataset candidate, not a confirmed benchmark choice.

## Roadmap

1. Project foundation and development conventions.
2. Paper/protocol review, dataset selection, and reusable data preparation.
3. Simple forecasting baselines to validate the pipeline.
4. Individually reviewed attention-based model implementations.
5. Unified experiments with documented model-specific requirements.
6. Comparative analysis, ablations, and visualizations.
7. Portfolio documentation and reproducibility instructions.

Each major milestone requires approval before work starts.

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

- [ASTGCN paper](https://guoshnbjtu.github.io/pdfs/AAAI2019-GuoS.2690.pdf)
  and [author repository](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch)
- [STAEformer author repository](https://github.com/XDZhelheim/STAEformer)
- [BigST author repository](https://github.com/usail-hkust/BigST)
- [PatchSTG paper](https://arxiv.org/abs/2412.09972)
  and [author repository](https://github.com/LMissher/PatchSTG)
- [LargeST paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ee57cd73a76bd927ffca3dda1dc3b9d4-Paper-Datasets_and_Benchmarks.pdf)

Reference code has not been copied. Review licenses and record source commits
before adapting any external code. A project license remains to be chosen.
