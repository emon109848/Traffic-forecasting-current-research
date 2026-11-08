"""Check the active Python/PyTorch environment without installing anything."""

import argparse
import json
import platform
import sys

import numpy as np
import torch


def check_device(device):
    """Check known values and gradients, followed by a tiny optimizer update."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
    product = x @ x.T
    expected = torch.tensor([[5.0, 11.0], [11.0, 25.0]], device=device)
    torch.testing.assert_close(product, expected)
    product.sum().backward()
    torch.testing.assert_close(
        x.grad, torch.tensor([[8.0, 12.0], [8.0, 12.0]], device=device)
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0], device=device))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    optimizer.zero_grad()
    parameter.square().sum().backward()
    optimizer.step()
    torch.testing.assert_close(parameter.detach(), torch.tensor([0.8], device=device))
    if device == "cuda":
        torch.cuda.synchronize()
    return "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cpu_check": "not_run",
        "numpy_interop": "not_run",
        "cuda_check": "not_run",
    }
    failed = False
    try:
        report["cpu_check"] = check_device("cpu")
        array = np.array([1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(torch.from_numpy(array).numpy(), array)
        report["numpy_interop"] = "passed"
    except Exception as exc:
        report["cpu_error"] = f"{type(exc).__name__}: {exc}"
        failed = True
    try:
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            properties = torch.cuda.get_device_properties(0)
            report["gpu_name"] = properties.name
            report["gpu_total_memory_bytes"] = properties.total_memory
            report["gpu_compute_capability"] = list(torch.cuda.get_device_capability(0))
            report["cuda_check"] = check_device("cuda")
        else:
            report["cuda_check"] = "unavailable"
            failed = failed or args.require_cuda
    except Exception as exc:
        report["cuda_check"] = "failed"
        report["cuda_error"] = f"{type(exc).__name__}: {exc}"
        failed = True
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
