"""Behavioral tests using synthetic observations, without private dataset files."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from torch.utils.data import DataLoader

from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows, make_splits, load_processed_sd


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.values = np.arange(80, dtype=np.float64).reshape(40, 2, 1)
        self.splits = make_splits(40)
        self.scaler = GlobalStandardScaler.fit(self.values, self.splits["train"][1])

    def windows(self, split, **kwargs):
        return TrafficWindows(self.values, self.splits[split], self.scaler,
                              history=3, horizon=2, **kwargs)

    def test_boundaries_and_expected_counts(self):
        self.assertEqual(make_splits(35040), {"train": (0, 21024), "val": (21024, 28032), "test": (28032, 35040)})
        self.assertEqual([len(self.windows(s)) for s in self.splits], [20, 7, 7])
        target_sets = []
        for split, (low, high) in self.splits.items():
            targets = set()
            for sample in self.windows(split):
                start = sample["target_start"]
                self.assertGreaterEqual(start, low)
                self.assertLessEqual(start + 2, high)
                targets.update(range(start, start + 2))
            target_sets.append(targets)
        self.assertFalse(target_sets[0] & target_sets[1])
        self.assertFalse(target_sets[1] & target_sets[2])

    def test_known_window_and_previous_split_context(self):
        sample = self.windows("val")[0]
        self.assertEqual(sample["target_start"], 24)
        np.testing.assert_allclose(sample["x"], self.scaler.transform(self.values[21:24]))
        np.testing.assert_array_equal(sample["y"], self.values[24:26])
        np.testing.assert_allclose(self.scaler.inverse_transform(sample["x"]), self.values[21:24], rtol=1e-6)

    def test_scaler_uses_training_only_and_includes_zero(self):
        changed = self.values.copy()
        changed[24:] = 1000000
        fitted = GlobalStandardScaler.fit(changed, 24)
        self.assertEqual(fitted, self.scaler)
        self.assertEqual(fitted.mean, 23.5)

    def test_zero_target_mask_and_no_input_imputation(self):
        values = self.values.copy()
        values[3:5] = 0
        dataset = TrafficWindows(values, (0, 24), self.scaler, history=3, horizon=2)
        sample = dataset[0]
        self.assertFalse(sample["target_mask"].any())
        np.testing.assert_array_equal(sample["y"], 0)
        # Zero at input frame zero is transformed, not filled or dropped.
        self.assertAlmostEqual(float(sample["x"][0, 0, 0]), -self.scaler.mean / self.scaler.std, places=6)

    def test_calendar_weekday_and_rollover(self):
        dataset = self.windows("train", assumed_start="2019-01-06T23:00:00")
        first = dataset[0]
        np.testing.assert_array_equal(first["x_calendar"], [[92, 6], [93, 6], [94, 6]])
        np.testing.assert_array_equal(first["y_calendar"], [[95, 6], [0, 0]])

    def test_samples_do_not_modify_source(self):
        sample = self.windows("train")[0]
        sample["x"][:] = -99
        sample["y"][:] = -99
        sample["x_calendar"][:] = -99
        self.assertEqual(self.values[3, 0, 0], 6)
        self.assertTrue((self.windows("train")[0]["x_calendar"] >= 0).all())

    def test_short_split_stride_and_bounds(self):
        empty = TrafficWindows(self.values, (0, 2), self.scaler, history=3, horizon=2)
        self.assertEqual(len(empty), 0)
        with self.assertRaises(IndexError):
            empty[0]
        self.assertEqual(list(self.windows("val", stride=2).target_starts), [24, 26, 28, 30])
        with self.assertRaises(IndexError):
            self.windows("val")[-1]

    def test_bad_values_scaler_and_configuration(self):
        for bad in (np.nan, np.inf, -1):
            values = self.values.copy()
            values[0, 0, 0] = bad
            with self.assertRaises(ValueError):
                GlobalStandardScaler.fit(values, 24)
        with self.assertRaises(ValueError):
            GlobalStandardScaler.fit(np.ones((10, 2, 1)), 6)
        with self.assertRaises(ValueError):
            self.windows("val", stride=0)
        with self.assertRaises(ValueError):
            make_splits(40, train_ratio=0.9, test_ratio=0.2)

    def test_loader_rejects_unexpected_archive_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "flowsd.npz", data=np.ones((3, 2, 1)))
            with self.assertRaisesRegex(ValueError, "Expected SD shape"):
                load_processed_sd(directory)

    def test_loader_preserves_metadata_order_and_rejects_duplicate_ids(self):
        # A broadcast view avoids allocating a synthetic full-size SD array.
        values = np.broadcast_to(np.array([1.0]), (35040, 716, 1))
        metadata = "ID,ID2,Lat,Lng,District\n" + "".join(
            f"{9000-i},{6931+i},32.5,-117.0,11\n" for i in range(716)
        )
        class Archive:
            files = ["data"]
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __getitem__(self, key): return values
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sd_meta.csv"
            path.write_text(metadata)
            with patch("traffic_benchmark.data.pipeline.np.load", return_value=Archive()):
                loaded, rows = load_processed_sd(directory)
                self.assertFalse(loaded.flags.writeable)
                self.assertEqual([rows[0]["ID"], rows[-1]["ID"]], ["9000", "8285"])
                path.write_text(metadata.replace("8999,6932", "9000,6932"))
                with self.assertRaisesRegex(ValueError, "ID must be present and unique"):
                    load_processed_sd(directory)

    def test_batch_shapes_and_dtypes(self):
        batch = next(iter(DataLoader(self.windows("train"), batch_size=2)))
        self.assertEqual(tuple(batch["x"].shape), (2, 3, 2, 1))
        self.assertEqual(tuple(batch["target_mask"].shape), (2, 2, 2, 1))
        self.assertEqual(str(batch["x"].dtype), "torch.float32")
        self.assertEqual(str(batch["target_mask"].dtype), "torch.bool")


if __name__ == "__main__":
    unittest.main()
