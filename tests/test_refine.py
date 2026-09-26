import unittest

import numpy as np
import pandas as pd

from lst_pilot.refine import (partitions, regular_blocks, row_weights, error_stats, _tail_no_worse,
                             CfbResidualModel, research_observed_time, _date_score)


class RefinementTests(unittest.TestCase):
    def test_whole_date_and_spatial_separation(self):
        frame = pd.DataFrame({
            "region_id": ["greater_london"] * 6,
            "datetime_utc": ["2022-07-18", "2023-05-26", "2023-09-07", "2024-06-29", "2022-07-18", "2023-05-26"],
            "pixel_x": [532800] * 4 + [552800] * 2,
            "pixel_y": [168400] * 4 + [188400] * 2,
        })
        masks = partitions(frame, {"blocks": regular_blocks()})
        self.assertEqual(np.flatnonzero(masks["train"]).tolist(), [0])
        self.assertEqual(np.flatnonzero(masks["development"]).tolist(), [1])
        self.assertEqual(np.flatnonzero(masks["calibration"]).tolist(), [2])
        self.assertEqual(np.flatnonzero(masks["test_2024"]).tolist(), [3])
        self.assertEqual(np.flatnonzero(masks["spatial_test"]).tolist(), [4, 5])
        self.assertTrue(np.all(np.sum(np.array(list(masks.values())), axis=0) == 1))

    def test_dense_pixels_do_not_change_date_total_weight(self):
        frame = pd.DataFrame({"region_id": ["greater_london"] * 6 + ["cabauw"] * 2,
                              "datetime_utc": ["2022-01-01"] * 2 + ["2022-02-01"] * 4 + ["2022-01-01"] * 2})
        w = row_weights(frame)
        self.assertAlmostEqual(w[:2].sum(), w[2:6].sum())
        self.assertAlmostEqual(w[:2].sum(), 2 * w[6:].sum())

    def test_tail_direction_and_empty_tail(self):
        s = error_stats([5, 40], [7, 35])
        self.assertEqual(s["bias_c"], -1.5)
        self.assertEqual(s["mae_c"], 3.5)
        self.assertIsNone(error_stats([], [])["mae_c"])
        b = {"cold_tail": {"n": 0, "mae_c": None}, "hot_tail": {"n": 2, "mae_c": 4.}}
        c = {"cold_tail": {"n": 0, "mae_c": None}, "hot_tail": {"n": 2, "mae_c": 3.}}
        self.assertTrue(_tail_no_worse(b, c))

    def test_correction_capped_and_inactive_outside_climate_or_at_night(self):
        class Constant:
            def __init__(self, value):
                self.value = value
            def predict(self, frame):
                return np.full(len(frame), self.value)
        frame = pd.DataFrame({"climate_class": ["Cfb", "Cfb", "BWh"], "solar_elevation_deg": [30, -5, 30],
                              "ndvi": [.3]*3, "ndbi": [.1]*3, "albedo_proxy": [.2]*3, "shortwave_down_w_m2": [500]*3})
        model = CfbResidualModel(Constant(10), Constant(50), np.zeros(7), np.ones(7))
        np.testing.assert_array_equal(model.predict(frame), [12, 10, 10])
        model.ridge.value = -50
        np.testing.assert_array_equal(model.predict(frame), [8, 10, 10])

    def test_research_time_does_not_relabel_overpass_or_allow_scenario(self):
        scene = {"properties": {"datetime": "2025-06-01T10:00:00Z"}}
        self.assertEqual(research_observed_time(scene, "2025-06-01T10:00:00Z", "observed")[2], 0)
        for date, mode in (("2025-06-01T11:00:00Z", "observed"), ("2025-06-01T10:00:00Z", "scenario"),
                           ("2024-06-01T10:00:00Z", "observed")):
            with self.assertRaises(ValueError):
                research_observed_time(scene, date, mode)

    def test_event_balanced_score_does_not_weight_large_raster_more(self):
        frame = pd.DataFrame({"region_id": ["greater_london"]*4, "lst_c": [20.]*4,
                              "datetime_utc": ["2023-05-01"]*3 + ["2023-09-01"]})
        result = _date_score(frame, [21., 21., 21., 25.])
        self.assertEqual(result["date_balanced_mae_c"], 3.)
        self.assertEqual(result["mae_c"], 2.)


if __name__ == "__main__":
    unittest.main()
