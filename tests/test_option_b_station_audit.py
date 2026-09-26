from pathlib import Path
import runpy

import pandas as pd


verify_rows = runpy.run_path(str(Path(__file__).resolve().parents[1] / "reports/night_replacement/audit_option_b_stations.py"))["verify_rows"]


def test_source_trace_requires_selected_time_and_exact_temperature():
    parsed = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2023-01-04T11:43Z", "2023-01-04T11:44Z"], utc=True),
                           "air_temperature_c": [-3.9, -3.8], "air_temperature_c_quality_code": ["1", "1"],
                           "air_temperature_c_source_code": ["313", "313"], "air_temperature_c_report_type": ["FM-15", "FM-15"]})
    frame = pd.DataFrame({"datetime_utc": pd.Timestamp("2023-01-04T11:52:55.953Z"),
                          "station_age_minutes": [9.93255, 9.93255, -1.],
                          "observed_station_air_temperature_c": [-3.9, -3.8, -3.9]}, index=[10, 20, 30])
    result = verify_rows(frame, parsed)
    assert result.verified_station_report_status.tolist() == ["exact_cached_report_verified", "report_time_or_value_mismatch", "report_time_or_value_mismatch"]
    assert result.verified_station_observation_datetime_utc.iloc[0] == pd.Timestamp("2023-01-04T11:43Z")
    assert result.verified_station_temperature_quality_code.tolist() == ["1", "", ""]


def test_source_trace_uses_same_duplicate_report_rule_as_assembler():
    parsed = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2022-01-01T01:00Z"] * 3, utc=True),
                           "air_temperature_c": [1., 2., float("nan")]})
    frame = pd.DataFrame({"datetime_utc": [pd.Timestamp("2022-01-01T01:30Z")], "station_age_minutes": [30.],
                          "observed_station_air_temperature_c": [2.]})
    assert verify_rows(frame, parsed).verified_station_report_status.iloc[0] == "exact_cached_report_verified"
