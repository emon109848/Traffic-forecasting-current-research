"""Evaluate training-only seasonal averages on SD validation/test; no GPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import torch

from traffic_benchmark.baselines import SeasonalAverage
from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows, load_processed_sd, make_splits
from traffic_benchmark.evaluation import MaskedForecastMetrics


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/largest_sd"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error("Output must be new and batch size positive.")
    values, _ = load_processed_sd(args.data_dir)
    splits = make_splits(len(values))
    scaler = GlobalStandardScaler.fit(values, splits["train"][1])
    report = {
        "baseline": "seasonal_average", "implementation": "independent educational baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"history": 12, "horizon": 12, "stride": 1, "batch_size": args.batch_size,
                   "splits": splits, "target_mask": "finite nonnegative targets != 0",
                   "mape": "100 * mean(abs(error) / target) on nonzero targets; no epsilon",
                   "aggregation": "pooled valid forecast-origin/lead/sensor/channel observations",
                   "input_zero_policy": "preserve", "prediction_units": "original source flow units",
                   "training": "nonzero training-only group averages; no gradient optimization", "device": "cpu",
                   "seasonal_fit": "weekday/slot/sensor mean excluding zero training readings",
                   "fallback": "sensor nonzero training mean, then global nonzero training mean",
                   "seed": None, "seed_reason": "No stochastic operations; closed-form training statistics only.",
                   "calendar_assumption": "2019-01-01 00:00, 15 min, no timezone; Monday=0; used for fitting and forecasting",
                   "sensor_alignment": "assumed metadata row order"},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "git": {"revision": git_output("rev-parse", "HEAD"), "status": git_output("status", "--porcelain")},
        "source_hashes": {name: file_hash(args.data_dir / name) for name in ("flowsd.npz", "sd_meta.csv")},
        "implementation_hashes": {name: file_hash(Path(name)) for name in (
            "scripts/evaluate_seasonal_average.py", "src/traffic_benchmark/baselines/seasonal_average.py",
            "src/traffic_benchmark/evaluation/metrics.py", "src/traffic_benchmark/data/pipeline.py")},
        "results": {},
    }
    reference_windows = TrafficWindows(values, splits["train"], scaler)
    begin = perf_counter()
    model = SeasonalAverage().fit(values, reference_windows.calendar, splits["train"][1])
    report["fit"] = {"wall_seconds": perf_counter() - begin,
                     "table_shape": list(model.table.shape),
                     "fallback_cells": int((model.counts == 0).sum()),
                     "total_cells": int(model.counts.size)}
    report["fit"]["sensors_without_nonzero_training"] = int((~(values[:splits["train"][1]] != 0).any(axis=(0,2))).sum())
    for split in ("val", "test"):
        dataset = TrafficWindows(values, splits[split], scaler)
        metrics = MaskedForecastMetrics(12)
        starts = np.asarray(dataset.target_starts)
        begin = perf_counter()
        for offset in range(0, len(starts), args.batch_size):
            batch = starts[offset:offset + args.batch_size]
            target_indices = batch[:, None] + np.arange(12)[None, :]
            forecast_calendar = dataset.calendar[target_indices]
            targets = values[target_indices]
            metrics.update(model.predict(forecast_calendar), targets)
        report["results"][split] = {"windows": len(dataset), **metrics.compute(),
            "evaluation_wall_seconds": perf_counter() - begin}
    report["timing_scope"] = "CPU batch indexing, lookup forecasting, metric accumulation; excludes loading/setup/fitting; not an inference benchmark"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for split, result in report["results"].items():
        print(split, json.dumps(result["overall_pooled"]))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
