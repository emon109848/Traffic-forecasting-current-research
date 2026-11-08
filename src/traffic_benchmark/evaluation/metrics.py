"""Accumulate sums/counts rather than averaging batch metrics."""

import numpy as np


class MaskedForecastMetrics:
    """MAE, pooled RMSE, percentage MAPE for positive traffic targets.

    Shape is (batch, horizon, sensors, channels). All predictions/targets must be
    finite, even at excluded targets. Zero targets alone are excluded. Empty
    valid sets yield None (JSON null) instead of a misleading zero score.
    """

    def __init__(self, horizon):
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        self.horizon = horizon
        self.count = np.zeros(horizon, dtype=np.int64)
        self.total = np.zeros(horizon, dtype=np.int64)
        self.absolute = np.zeros(horizon, dtype=np.float64)
        self.squared = np.zeros(horizon, dtype=np.float64)
        self.percentage = np.zeros(horizon, dtype=np.float64)

    def update(self, prediction, target):
        prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if prediction.shape != target.shape or target.ndim != 4 or target.shape[1] != self.horizon or min(target.shape) < 1:
            raise ValueError("Expected matching nonempty (batch,horizon,sensors,channels) arrays.")
        if not (np.isfinite(prediction).all() and np.isfinite(target).all()):
            raise ValueError("Predictions and targets must all be finite, including masked positions.")
        if (target < 0).any():
            raise ValueError("Traffic targets must be nonnegative.")
        mask = target != 0
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error = np.where(mask, prediction - target, 0.0)
            absolute = np.abs(error)
            squared = np.square(error)
            percentage = np.divide(absolute, target, out=np.zeros_like(error), where=mask)
            axes = (0, 2, 3)
            sums = [x.sum(axis=axes) for x in (absolute, squared, percentage)]
        if not all(np.isfinite(x).all() for x in sums):
            raise ValueError("Metric accumulation overflowed.")
        # Validate everything before mutating accumulator state.
        new_sums = [a + b for a, b in zip((self.absolute, self.squared, self.percentage), sums)]
        if not all(np.isfinite(x).all() for x in new_sums):
            raise ValueError("Metric state overflowed.")
        self.absolute, self.squared, self.percentage = new_sums
        self.count += mask.sum(axis=axes)
        self.total += target.shape[0] * target.shape[2] * target.shape[3]

    @staticmethod
    def _scores(absolute, squared, percentage, count, total):
        return {"mae": float(absolute / count) if count else None,
                "rmse": float(np.sqrt(squared / count)) if count else None,
                "mape_percent": float(100 * percentage / count) if count else None,
                "valid_targets": int(count), "total_targets": int(total),
                "valid_fraction": float(count / total) if total else None}

    def compute(self):
        per_horizon = []
        for i in range(self.horizon):
            per_horizon.append({"step": i + 1, **self._scores(
                self.absolute[i], self.squared[i], self.percentage[i], self.count[i], self.total[i])})
        return {"overall_pooled": self._scores(self.absolute.sum(), self.squared.sum(),
                self.percentage.sum(), self.count.sum(), self.total.sum()), "per_horizon": per_horizon}
