from pathlib import Path
import tempfile
import unittest

import numpy as np

from lst_pilot.reference import COLUMNS, SIGMA, daily_url, parse_day, radiometric_temperature_c


class ReferenceTests(unittest.TestCase):
    def test_emissivity_and_reflected_longwave(self):
        expected_c, emissivity, down = 20.0, 0.97, 300.0
        up = emissivity * SIGMA * (expected_c + 273.15) ** 4 + (1 - emissivity) * down
        self.assertAlmostEqual(float(radiometric_temperature_c(up, down, emissivity)), expected_c, places=8)
        self.assertTrue(np.isnan(radiometric_temperature_c(-9999.9, down)))

    def test_noaa_qc_and_end_timestamp(self):
        values = {column: 0 for column in COLUMNS}
        values.update(year=2023, month=1, day=1, day_of_year=1, hour=0, minute=0,
                      solar_zenith_deg=100.55, longwave_down_w_m2=272.1,
                      longwave_up_w_m2=292.2, air_temperature_10m_c=-2.6,
                      direct_normal_w_m2=0.8, diffuse_w_m2=-0.2)
        row1 = [values[column] for column in COLUMNS]
        values["minute"] = 1
        values["longwave_up_w_m2_qc"] = 1
        row2 = [values[column] for column in COLUMNS]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sxf23001.dat"
            path.write_text(" Sioux Falls\n 43.73 -96.62 473 m version 1\n" + "\n".join(" ".join(map(str, row)) for row in [row1, row2]) + "\n")
            frame = parse_day(path)
        self.assertEqual(frame.timestamp_utc.iloc[0].isoformat(), "2023-01-01T00:00:00+00:00")
        self.assertEqual(frame.air_temperature_10m_c.iloc[0], -2.6)
        self.assertFalse(frame.is_daylight.iloc[0])
        self.assertEqual(frame.shortwave_components_w_m2.iloc[0], 0.0)
        self.assertTrue(frame.radiometric_temperature_c.isna().iloc[1])
        self.assertEqual(frame.longwave_up_w_m2_raw.iloc[1], 292.2)
        self.assertEqual(frame.attrs["air_temperature_height_m"], 10)

    def test_leap_day_url(self):
        self.assertTrue(daily_url("tbl", "2020-02-29").endswith("/2020/tbl20060.dat"))


if __name__ == "__main__":
    unittest.main()
