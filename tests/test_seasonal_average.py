"""Verify seasonal fitting, strict training boundaries, and fallback behavior."""

import unittest
import numpy as np
from traffic_benchmark.baselines import SeasonalAverage


class SeasonalTests(unittest.TestCase):
    def setUp(self):
        self.values = np.array([[2, 0], [4, 0], [6, 8], [999, 999]], dtype=float)[..., None]
        self.calendar = np.array([[0, 0], [0, 0], [1, 1], [0, 0]])

    def test_known_seasonal_mean_and_zero_exclusion(self):
        model = SeasonalAverage(2).fit(self.values, self.calendar, 3)
        prediction = model.predict(np.array([[[0, 0], [1, 1]]]))
        np.testing.assert_array_equal(prediction[0, :, :, 0], [[3, 8], [6, 8]])
        self.assertEqual(model.counts[0, 0, 0], 2)

    def test_heldout_values_and_calendar_cannot_change_fit(self):
        a = SeasonalAverage(2).fit(self.values, self.calendar, 3)
        changed = self.values.copy(); changed[3:] = np.nan
        calendar = self.calendar.copy(); calendar[3:] = -999
        b = SeasonalAverage(2).fit(changed, calendar, 3)
        np.testing.assert_array_equal(a.table, b.table)

    def test_unseen_slots_sensor_and_global_fallback(self):
        values = np.concatenate([self.values, np.zeros((4, 1, 1))], axis=1)
        model = SeasonalAverage(2).fit(values, self.calendar, 3)
        prediction = model.predict(np.array([[[1, 6]]]))[0, 0, :, 0]
        np.testing.assert_array_equal(prediction, [4, 8, 5])
        self.assertEqual(model.global_mean, 5)

    def test_all_zero_training_fails(self):
        with self.assertRaisesRegex(ValueError, "No nonzero"):
            SeasonalAverage(2).fit(np.zeros_like(self.values), self.calendar, 3)

    def test_bad_calendar_and_unfitted_model(self):
        model = SeasonalAverage(2)
        with self.assertRaises(ValueError): model.predict(np.array([[[0, 0]]]))
        model.fit(self.values, self.calendar, 3)
        for bad in (np.array([[[2, 0]]]), np.array([[[0, 7]]]), np.array([[[0., 0.]]])):
            with self.assertRaises(ValueError): model.predict(bad)
        result = model.predict(np.array([[[0, 0]]]))
        result[:] = -99
        self.assertGreaterEqual(model.table.min(), 0)

    def test_prediction_uses_each_forecast_time(self):
        model = SeasonalAverage(2).fit(self.values, self.calendar, 3)
        result = model.predict(np.array([[[0,0],[1,1]],[[1,1],[0,0]]]))
        self.assertEqual(result.shape, (2,2,2,1))
        self.assertEqual(result[0,0,0,0], 3)
        self.assertEqual(result[0,1,0,0], 6)
        self.assertEqual(result[1,1,0,0], 3)


if __name__ == "__main__":
    unittest.main()
