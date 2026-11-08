"""Evaluate deterministic persistence on SD validation/test; no fitting or GPU."""

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

from traffic_benchmark.baselines import persistence
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
        "baseline": "persistence", "implementation": "independent educational baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"history": 12, "horizon": 12, "stride": 1, "batch_size": args.batch_size,
                   "splits": splits, "target_mask": "finite nonnegative targets != 0",
                   "mape": "100 * mean(abs(error) / target) on nonzero targets; no epsilon",
                   "aggregation": "pooled valid forecast-origin/lead/sensor/channel observations",
                   "input_zero_policy": "preserve", "prediction_units": "original source flow units",
                   "parameters": 0, "training": False, "device": "cpu",
                   "seed": None, "seed_reason": "No stochastic operations or fitted parameters.",
                   "calendar_assumption": "2019-01-01 00:00, 15 min, no timezone; unused by persistence",
                   "sensor_alignment": "assumed metadata row order"},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "git": {"revision": git_output("rev-parse", "HEAD"), "status": git_output("status", "--porcelain")},
        "source_hashes": {name: file_hash(args.data_dir / name) for name in ("flowsd.npz", "sd_meta.csv")},
        "implementation_hashes": {name: file_hash(Path(name)) for name in (
            "scripts/evaluate_persistence.py", "src/traffic_benchmark/baselines/persistence.py",
            "src/traffic_benchmark/evaluation/metrics.py", "src/traffic_benchmark/data/pipeline.py")},
        "results": {},
    }
    for split in ("val", "test"):
        dataset = TrafficWindows(values, splits[split], scaler)
        metrics = MaskedForecastMetrics(12)
        starts = np.asarray(dataset.target_starts)
        begin = perf_counter()
        for offset in range(0, len(starts), args.batch_size):
            batch = starts[offset:offset + args.batch_size]
            # Same window indices as the pipeline; raw observations avoid z-score
            # round-trip error and unnecessary history copies for this baseline.
            latest = values[batch - 1][:, None]
            targets = values[batch[:, None] + np.arange(12)[None, :]]
            metrics.update(persistence(latest, 12), targets)
        report["results"][split] = {"windows": len(dataset), **metrics.compute(),
            "evaluation_wall_seconds": perf_counter() - begin}
    report["timing_scope"] = "CPU batch indexing, forecasting, metric accumulation; excludes loading/setup; not an inference benchmark"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for split, result in report["results"].items():
        print(split, json.dumps(result["overall_pooled"]))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
