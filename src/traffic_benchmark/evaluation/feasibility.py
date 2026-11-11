"""One-step execution probes and bounded batch sweeps; not model training."""

import torch


def optimizer_probe(model, x, calendar, optimizer):
    """Probe finite forward/backward and one update using artificial output energy.

    This intentionally has no target, forecast loss, or reusable trained weights.
    Callers own synchronization, device, memory measurement, and model lifetime.
    """
    model.train()
    first = next(model.parameters())
    before = first.detach().cpu().clone()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x, calendar)
    if not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite prediction in feasibility probe.")
    objective = prediction.square().mean()
    if not torch.isfinite(objective):
        raise ValueError("Nonfinite artificial objective.")
    objective.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise ValueError(f"Missing/nonfinite gradient for {name}.")
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"Nonfinite parameter after optimizer step: {name}.")
    changed = not torch.equal(before, first.detach().cpu())
    if not changed:
        raise ValueError("Probe did not change the checked parameter.")
    return {"output_shape": list(prediction.shape), "finite_prediction": True,
            "finite_gradients": True, "finite_updated_parameters": True,
            "checked_parameter_changed": True, "optimizer_steps": 1,
            "objective": "mean squared normalized model output; no forecast targets"}


def sweep_batches(run_one, batches=(1, 2, 4)):
    """Stop after first unsuccessful worker; retain explicit unattempted entries."""
    if (not batches or any(not isinstance(b, int) or isinstance(b, bool) or b < 1 for b in batches)
            or list(batches) != sorted(set(batches))):
        raise ValueError("Batch sizes must be increasing unique positive integers.")
    results = []
    stopped = False
    for batch in batches:
        if stopped:
            results.append({"batch_size": batch, "status": "not_attempted", "reason": "earlier batch did not pass"})
            continue
        result = run_one(batch)
        if result.get("batch_size") != batch or result.get("status") not in {"passed", "oom", "failed", "cuda_unavailable"}:
            raise ValueError("Malformed feasibility worker report.")
        results.append(result)
        stopped = result["status"] != "passed"
    passed = [row["batch_size"] for row in results if row["status"] == "passed"]
    return {"results": results, "largest_passed_batch": max(passed) if passed else None,
            "all_requested_passed": not stopped}
