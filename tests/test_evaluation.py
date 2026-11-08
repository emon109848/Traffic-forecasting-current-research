"""Known-value and partition-invariance checks for baseline evaluation."""

import unittest
import numpy as np

from traffic_benchmark.baselines import persistence
from traffic_benchmark.evaluation import MaskedForecastMetrics


class EvaluationTests(unittest.TestCase):
    def test_known_metrics_zero_exclusion_and_percentage(self):
        target = np.array([2., 0., 4.]).reshape(1, 1, 3, 1)
        prediction = np.array([3., 999., 2.]).reshape(target.shape)
        metrics = MaskedForecastMetrics(1)
        metrics.update(prediction, target)
        scores = metrics.compute()["overall_pooled"]
        self.assertEqual(scores["mae"], 1.5)
        self.assertAlmostEqual(scores["rmse"], np.sqrt(2.5))
        self.assertEqual(scores["mape_percent"], 50.)
        self.assertEqual(scores["valid_targets"], 2)

    def test_batch_partition_invariance_and_pooled_rmse(self):
        target = np.array([1., 0., 2., 2., 0., 4.]).reshape(3, 2, 1, 1)
        prediction = np.ones_like(target) * 4
        full, chunks = MaskedForecastMetrics(2), MaskedForecastMetrics(2)
        full.update(prediction, target)
        chunks.update(prediction[:1], target[:1])
        chunks.update(prediction[1:], target[1:])
        self.assertEqual(full.compute(), chunks.compute())
        report = full.compute()
        self.assertAlmostEqual(report["overall_pooled"]["rmse"], np.sqrt(17/4))
        self.assertEqual([r["valid_targets"] for r in report["per_horizon"]], [2, 2])

    def test_empty_masks_and_no_updates(self):
        metrics = MaskedForecastMetrics(2)
        self.assertIsNone(metrics.compute()["overall_pooled"]["mae"])
        metrics.update(np.ones((1, 2, 1, 1)), np.zeros((1, 2, 1, 1)))
        report = metrics.compute()["overall_pooled"]
        self.assertIsNone(report["mae"])
        self.assertIsNone(report["rmse"])
        self.assertIsNone(report["mape_percent"])
        self.assertEqual(report["valid_fraction"], 0)

    def test_nonfinite_predictions_at_zero_targets_rejected(self):
        metrics = MaskedForecastMetrics(1)
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                metrics.update(np.full((1, 1, 1, 1), value), np.zeros((1, 1, 1, 1)))
        self.assertEqual(metrics.compute()["overall_pooled"]["total_targets"], 0)

    def test_bad_shapes_negative_targets_and_small_positive_mape(self):
        metrics = MaskedForecastMetrics(1)
        with self.assertRaises(ValueError): metrics.update(np.ones((1, 1)), np.ones((1, 1)))
        with self.assertRaises(ValueError): metrics.update(np.ones((1, 1, 1, 1)), -np.ones((1, 1, 1, 1)))
        metrics.update(np.array([0.02]).reshape(1,1,1,1), np.array([0.01]).reshape(1,1,1,1))
        self.assertAlmostEqual(metrics.compute()["overall_pooled"]["mape_percent"], 100)

    def test_persistence_uses_only_last_value_including_zero(self):
        history = np.array([10., 2., 8., 0.]).reshape(1,2,2,1)
        result = persistence(history, 3)
        np.testing.assert_array_equal(result, np.array([8., 0.] * 3).reshape(1,3,2,1))
        result[:] = -1
        self.assertEqual(history[0,-1,0,0], 8)
        with self.assertRaises(ValueError): persistence(history, 0)


if __name__ == "__main__":
    unittest.main()
