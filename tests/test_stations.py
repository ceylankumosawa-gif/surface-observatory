import csv
from pathlib import Path
import tempfile
import unittest

from lst_pilot.stations import parse_station_file, quality_usable, station_year_url


class StationParserTests(unittest.TestCase):
    def test_qc_codes_are_source_specific(self):
        self.assertTrue(quality_usable("temperature", "1", "223"))
        self.assertFalse(quality_usable("temperature", "0", "223"))
        self.assertTrue(quality_usable("temperature", "0", "313"))
        self.assertFalse(quality_usable("temperature", "s", "223"))
        self.assertFalse(quality_usable("temperature", "n", "999"))
        self.assertTrue(quality_usable("temperature", "", "999"))

    def test_physical_units_utc_qc_and_running_precip_are_preserved(self):
        columns = ["STATION", "DATE", "temperature", "temperature_Quality_Code", "temperature_Source_Code",
                   "wind_speed", "wind_speed_Quality_Code", "wind_speed_Source_Code", "precipitation",
                   "precipitation_Measurement_Code", "precipitation_Quality_Code", "precipitation_Source_Code"]
        rows = [
            ["UKI0000EGLL", "2023-01-01T13:20:00", "12.0", "1", "223", "7.7", "1", "223", "16.5", "", "1", "223"],
            ["UKI0000EGLL", "2023-01-01T13:50:00", "99.0", "s", "223", "-9999", "1", "223", "47.2", "", "1", "223"],
            ["UKI0000EGLL", "2023-01-01T14:50:00", "11.5", "1", "223", "2.2", "1", "223", "0.0", "T", "1", "223"],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.psv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="|")
                writer.writerow(columns)
                writer.writerows(rows)
            frame = parse_station_file(path)
        self.assertEqual(frame.air_temperature_c.iloc[0], 12.0)
        self.assertEqual(frame.wind_speed_m_s.iloc[0], 7.7)
        self.assertTrue(frame.air_temperature_c.isna().iloc[1])
        self.assertTrue(frame.wind_speed_m_s.isna().iloc[1])
        self.assertEqual(str(frame.timestamp_utc.dt.tz), "UTC")
        self.assertEqual(frame.timestamp_utc.dt.minute.tolist(), [20, 50, 50])
        self.assertEqual(frame.precipitation_mm.tolist(), [16.5, 47.2, 0.0])
        self.assertTrue(frame.precipitation_trace.iloc[2])

    def test_station_ids_cannot_escape_cache(self):
        with self.assertRaises(ValueError):
            station_year_url("../../evil", 2023)


if __name__ == "__main__":
    unittest.main()
