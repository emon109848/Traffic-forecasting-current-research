"""Validate local SD loading and report split/window/scaler settings; no training."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows, load_processed_sd, make_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/largest_sd"))
    parser.add_argument("--output", type=Path, help="Optional JSON report; refuses overwrite.")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new report path.")
    values, metadata = load_processed_sd(args.data_dir)
    splits = make_splits(len(values))
    scaler = GlobalStandardScaler.fit(values, splits["train"][1])
    report = {
        "shape": list(values.shape), "source_dtype": str(values.dtype),
        "scaler": asdict(scaler), "scaler_zero_policy": "include_training_zeros",
        "input_zero_policy": "preserve_no_imputation", "target_mask": "target != 0",
        "history": 12, "horizon": 12, "stride": 1,
        "calendar": {"status": "assumed_not_observed", "start": "2019-01-01T00:00:00",
                     "cadence_minutes": 15, "timezone": None, "weekday_encoding": "Monday=0"},
        "sensor_alignment": "assumed_metadata_row_order", "files": {}, "splits": {},
        "numpy": np.__version__,
    }
    for name in ("flowsd.npz", "sd_meta.csv"):
        digest = hashlib.sha256()
        with (args.data_dir / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        report["files"][name] = {"sha256": digest.hexdigest()}
    for name, interval in splits.items():
        dataset = TrafficWindows(values, interval, scaler)
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        start, end = interval
        report["splits"][name] = {
            "target_interval": list(interval), "windows": len(dataset),
            "first_target_start": dataset.target_starts[0],
            "last_target_start": dataset.target_starts[-1],
            "zero_fraction_in_split_frames": float(np.mean(values[start:end] == 0)),
            "batch_x_shape": list(batch["x"].shape), "batch_y_shape": list(batch["y"].shape),
        }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents an overwrite after the initial check.
        with args.output.open("x") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
