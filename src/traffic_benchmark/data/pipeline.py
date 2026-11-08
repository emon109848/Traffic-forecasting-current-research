"""Processed-SD pipeline; sensor pairing and calendar are explicit assumptions."""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def _check_values(values):
    if values.ndim != 3 or values.shape[-1] != 1 or min(values.shape) < 1:
        raise ValueError("Traffic must have nonempty shape (time, sensors, 1).")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("Traffic must be real numeric values.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Traffic must be finite and nonnegative; no silent imputation.")


def load_processed_sd(directory):
    """Load originals without reordering. Row/column pairing is not verifiable."""
    directory = Path(directory)
    with np.load(directory / "flowsd.npz", allow_pickle=False) as archive:
        if archive.files != ["data"]:
            raise ValueError("Expected exactly one NPZ array named 'data'.")
        values = archive["data"]
    _check_values(values)
    if values.shape != (35040, 716, 1):
        raise ValueError(f"Expected SD shape (35040, 716, 1), got {values.shape}.")
    with (directory / "sd_meta.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ID", "ID2", "Lat", "Lng", "District"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Metadata lacks required ID/coordinate fields.")
        metadata = list(reader)
    if len(metadata) != values.shape[1]:
        raise ValueError("Metadata row count differs from sensor count.")
    for field in ("ID", "ID2"):
        entries = [row[field] for row in metadata]
        if any(not value for value in entries) or len(set(entries)) != len(entries):
            raise ValueError(f"Metadata {field} must be present and unique.")
    for row in metadata:
        lat, lng = float(row["Lat"]), float(row["Lng"])
        if not (np.isfinite(lat) and np.isfinite(lng) and -90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError("Invalid metadata coordinates.")
        if int(row["District"]) != 11:
            raise ValueError("Expected SD District 11 metadata.")
    values.setflags(write=False)
    return values, metadata


def make_splits(length, train_ratio=0.6, test_ratio=0.2):
    """Return half-open target intervals; validation receives the remainder."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer.")
    if not (0 < train_ratio < 1 and 0 < test_ratio < 1 and train_ratio + test_ratio < 1):
        raise ValueError("Ratios must leave positive train/validation/test fractions.")
    train_end = int(length * train_ratio)
    test_start = length - int(length * test_ratio)
    if not (0 < train_end < test_start < length):
        raise ValueError("Every split must contain at least one frame.")
    return {"train": (0, train_end), "val": (train_end, test_start), "test": (test_start, length)}


@dataclass(frozen=True)
class GlobalStandardScaler:
    """Scalar z-score fitted on all training values, including zeros."""

    mean: float
    std: float

    def __post_init__(self):
        if not (np.isfinite(self.mean) and np.isfinite(self.std) and self.std > 0):
            raise ValueError("Scaler requires finite mean and strictly positive std.")

    @classmethod
    def fit(cls, values, train_end):
        _check_values(values)
        if not isinstance(train_end, int) or not 0 < train_end <= len(values):
            raise ValueError("Invalid exclusive training boundary.")
        train = values[:train_end]
        return cls(float(np.mean(train, dtype=np.float64)), float(np.std(train, dtype=np.float64)))

    def transform(self, values):
        # Compute with float64 statistics before casting each small window.
        transformed = (values.astype(np.float64) - self.mean) / self.std
        result = transformed.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Normalized values overflowed float32.")
        return result

    def inverse_transform(self, values):
        return values * self.std + self.mean


class TrafficWindows(Dataset):
    """Lazy windows; x normalized, y original units, zero targets masked.

    Calendar columns are integer slot-of-day and weekday (Monday=0), reconstructed
    from assumed_start. They are not observed timestamps or a source validity mask.
    """

    def __init__(self, values, target_interval, scaler, history=12, horizon=12,
                 stride=1, assumed_start="2019-01-01T00:00:00", interval_minutes=15):
        _check_values(values)
        for size in (history, horizon, stride, interval_minutes):
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError("Window sizes, stride and cadence must be positive integers.")
        if 1440 % interval_minutes:
            raise ValueError("Cadence must divide a day exactly.")
        start, end = target_interval
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(values)):
            raise ValueError("Invalid half-open target interval.")
        self.assumed_start = datetime.fromisoformat(assumed_start)
        if (self.assumed_start.tzinfo is not None or self.assumed_start.second
                or self.assumed_start.microsecond
                or (self.assumed_start.hour * 60 + self.assumed_start.minute) % interval_minutes):
            raise ValueError("Assumed start must be timezone-naive and aligned to cadence.")
        self.values, self.scaler = values, scaler
        self.history, self.horizon = history, horizon
        # Index denotes the first target. Every target belongs to this split.
        self.target_starts = range(max(start, history), end - horizon + 1, stride)
        self.calendar = np.empty((len(values), 2), dtype=np.int64)
        for i in range(len(values)):
            stamp = self.assumed_start + timedelta(minutes=i * interval_minutes)
            self.calendar[i] = ((stamp.hour * 60 + stamp.minute) // interval_minutes, stamp.weekday())
        self.calendar.setflags(write=False)

    def __len__(self):
        return len(self.target_starts)

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        target = self.target_starts[index]
        end = target + self.horizon
        # Copies ensure sample mutation does not change originals or other windows.
        x = self.scaler.transform(self.values[target - self.history:target])
        y = self.values[target:end].astype(np.float32, copy=True)
        if not np.isfinite(y).all():
            raise ValueError("Targets overflowed float32.")
        return {"x": x, "y": y, "target_mask": y != 0,
                "x_calendar": self.calendar[target - self.history:target].copy(),
                "y_calendar": self.calendar[target:end].copy(),
                "target_start": target}
