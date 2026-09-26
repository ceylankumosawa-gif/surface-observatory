import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from lst_pilot.radiation import _grid_indices, add_radiation


class RadiationTests(unittest.TestCase):
    def test_longitude_wrap_and_nearest_latitude(self):
        rows, cols = _grid_indices([51.49, 0, -90, 90], [-0.1, -179.9, 359.9, 180])
        np.testing.assert_equal(rows, [154, 360, 720, 0])
        np.testing.assert_equal(cols, [0, 720, 0, 720])

    def test_budget_checked_before_network_access(self):
        frame = pd.DataFrame({"latitude": [51.5, 51.5], "longitude": [-0.1, -0.1],
                              "datetime_utc": ["2023-01-01T10:43:00Z", "2023-01-01T11:43:00Z"]})
        result = add_radiation(frame, "/unused-cache", max_hours=1)
        self.assertTrue(result.era5_longwave_down_w_m2.isna().all())
        self.assertEqual(result.era5_snow_water_equivalent_m_status.tolist(), ["hour_budget_exceeded"] * 2)
        self.assertEqual(result.radiation_era5_time_utc.dt.hour.tolist(), [10, 11])
        self.assertEqual(result.radiation_era5_time_utc.dt.minute.tolist(), [0, 0])
        self.assertEqual(len(result.attrs["radiation_context"]["objects"]), 0)
        self.assertNotIn("era5_longwave_down_w_m2", frame)

    def test_metadata_failure_keeps_missing_explicit(self):
        frame = pd.DataFrame({"latitude": [51.5], "longitude": [-0.1], "datetime_utc": ["2023-01-01T10:43:00Z"]})
        with patch("fsspec.filesystem", side_effect=OSError("deliberate offline test")):
            result = add_radiation(frame, "/unused-cache")
        self.assertTrue(result.era5_longwave_down_w_m2.isna().all())
        self.assertEqual(result.era5_longwave_down_w_m2_status.iloc[0], "metadata_unavailable")
        self.assertIn("offline", result.attrs["radiation_context"]["errors"][0])


if __name__ == "__main__":
    unittest.main()
