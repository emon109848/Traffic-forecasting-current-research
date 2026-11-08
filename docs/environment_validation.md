# Conda environment validation

Status: CPU and host PyTorch CUDA smoke checks passed. Updated 2026-11-08.

## Candidate environment

The existing `pml` environment is being evaluated without modification. It has
Python 3.14.8, torch 2.14.1+cu130 (CUDA build 13.0), and NumPy 2.5.4. It also
contains unrelated packages, so this is not a minimal project environment or a
final dependency specification. No package or Conda environment was changed.
The earlier Python 3.11 suggestion remains an alternative, not a required downgrade.

## Reusable check

From the project root in the user's WSL terminal:

```bash
conda activate pml
python scripts/check_environment.py --require-cuda
```

The check verifies known matrix-product values and gradients, a tiny SGD update,
and NumPy/PyTorch array interoperability. It runs numerical/optimizer checks on
CPU and, when visible, CUDA device 0. It reports software versions, GPU name,
memory and compute capability. It uses tiny synthetic tensors, not project data.
No external downloads are required.

With `--require-cuda`, unavailable CUDA or a failing check produces exit status 1.
Without it, unavailable CUDA is reported but does not fail a successful CPU check.
A CUDA execution error always fails. Import failures also exit unsuccessfully.
This is an environment smoke check, not a model benchmark or full compatibility
test for future data packages and research implementations.

## Agent-session result

- CPU known-value matrix product and gradient: passed.
- CPU optimizer update: passed.
- NumPy interoperability: passed.
- CUDA availability: false; CUDA numerical checks not executed.
- Strict command exit status: 1, correctly reflecting unavailable CUDA.

The user separately demonstrated host GPU visibility with nvidia-smi. Session
restrictions prevent completing host CUDA validation from here. User-run output
from the command above is needed before declaring GPU readiness. No environment
lock or environment.yml has been generated: freezing all unrelated pml packages
would misrepresent project requirements, and a new installation is not needed to
resolve this check. Record the host result before choosing whether to retain pml
or create a dedicated environment.

## User-terminal result — validation complete

On 2026-11-08 the user ran the strict check in activated pml and supplied:

```json
{
  "python": "3.14.8",
  "torch": "2.14.1+cu130",
  "numpy": "2.5.4",
  "torch_cuda_build": "13.0",
  "cpu_check": "passed",
  "numpy_interop": "passed",
  "cuda_check": "passed",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "gpu_total_memory_bytes": 8518041600,
  "gpu_compute_capability": [12, 0]
}
```

The host check resolves the pending CUDA gate; the earlier failure reflects
agent-session visibility. Retain pml as the current development environment
without changing packages. These versions are an observed working smoke-check
combination, not a full lock or proof of future model compatibility. A minimal
environment specification should be added when data-pipeline dependencies are
selected and checked; do not export unrelated packages as project requirements.
