"""Persistence baseline; no fitting, imputation, or zero replacement."""

import numpy as np


def persistence(history, horizon):
    """Repeat the latest original-unit observation, including zero, at each lead."""
    history = np.asarray(history)
    if history.ndim != 4 or min(history.shape) < 1 or not np.isfinite(history).all():
        raise ValueError("Expected finite nonempty (batch,history,sensors,channels).")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    return np.repeat(history[:, -1:], horizon, axis=1)
