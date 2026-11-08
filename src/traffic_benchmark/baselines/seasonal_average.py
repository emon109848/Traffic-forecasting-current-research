"""Training-only weekday/slot means; independently written educational baseline."""

import numpy as np


class SeasonalAverage:
    """Mean nonzero training flow per weekday, slot and sensor.

    Fallback: sensor nonzero training mean, then global nonzero training mean.
    Calendar is supplied explicitly; no timestamps are inferred by this class.
    """

    def __init__(self, slots_per_day=96):
        if not isinstance(slots_per_day, int) or isinstance(slots_per_day, bool) or slots_per_day < 1:
            raise ValueError("slots_per_day must be a positive integer.")
        self.slots_per_day = slots_per_day
        self.table = None

    def _calendar_keys(self, calendar):
        calendar = np.asarray(calendar)
        if calendar.ndim < 2 or calendar.shape[-1] != 2 or not np.issubdtype(calendar.dtype, np.integer):
            raise ValueError("Calendar must have integer columns (slot, weekday).")
        slot, weekday = calendar[..., 0], calendar[..., 1]
        if ((slot < 0) | (slot >= self.slots_per_day) | (weekday < 0) | (weekday > 6)).any():
            raise ValueError("Calendar index out of range.")
        return weekday * self.slots_per_day + slot

    def fit(self, values, calendar, train_end):
        """Read only values/calendar before the exclusive training boundary."""
        values = np.asarray(values)
        if values.ndim != 3 or min(values.shape) < 1 or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError("Expected real numeric (time,sensors,channels).")
        if not isinstance(train_end, int) or isinstance(train_end, bool) or not 0 < train_end <= len(values):
            raise ValueError("Invalid training boundary.")
        if np.asarray(calendar).shape != (len(values), 2):
            raise ValueError("Calendar must align with traffic frames.")
        train = values[:train_end].astype(np.float64)
        if not np.isfinite(train).all() or (train < 0).any():
            raise ValueError("Training flow must be finite and nonnegative.")
        keys = self._calendar_keys(np.asarray(calendar)[:train_end])
        valid = train != 0
        if not valid.any():
            raise ValueError("No nonzero training values; fallback cannot be fitted.")
        counts = np.zeros((7 * self.slots_per_day, *train.shape[1:]), dtype=np.int64)
        sums = np.zeros_like(counts, dtype=np.float64)
        np.add.at(counts, keys, valid)
        np.add.at(sums, keys, np.where(valid, train, 0))
        global_mean = train[valid].mean()
        sensor_counts = valid.sum(axis=0)
        sensor_mean = np.divide(train.sum(axis=0), sensor_counts,
                                out=np.full(train.shape[1:], global_mean), where=sensor_counts != 0)
        table = np.divide(sums, counts, out=np.broadcast_to(sensor_mean, sums.shape).copy(), where=counts != 0)
        if not np.isfinite(table).all():
            raise ValueError("Fitted seasonal table is not finite.")
        self.counts, self.sensor_mean, self.global_mean = counts, sensor_mean, float(global_mean)
        self.table = table
        self.table.setflags(write=False)
        return self

    def predict(self, forecast_calendar):
        """Predict (batch,horizon,sensors,channels) from known future calendar."""
        if self.table is None:
            raise ValueError("Fit the seasonal baseline before prediction.")
        calendar = np.asarray(forecast_calendar)
        if calendar.ndim != 3 or min(calendar.shape) < 1:
            raise ValueError("Expected nonempty (batch,horizon,2) calendar.")
        return self.table[self._calendar_keys(calendar)].copy()
