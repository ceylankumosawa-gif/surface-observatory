"""Documented physical units and narrow fallback scope; no network or fitting."""
import json

import numpy as np
import pandas as pd
import pytest

import lst_pilot.legacy_isd as isd


def raw_csv(tmp_path, temperatures):
    path = tmp_path / "94120099999_2023.csv"
    frame = pd.DataFrame({"STATION": isd.STATION_ID,
                          "DATE": pd.date_range("2023-01-01", periods=len(temperatures), freq="h").astype(str),
                          "LATITUDE": -12.414722, "LONGITUDE": 130.876667, "ELEVATION": 30,
                          "TMP": temperatures, "SOURCE": "4", "REPORT_TYPE": "FM-15"})
    frame.to_csv(path, index=False)
    return path


def test_physical_scaling_missing_sentinel_and_qc1_only(tmp_path):
    path = raw_csv(tmp_path, ["+0260,1", "-0050,1", "+9999,1", "+0310,2", "+0275,0", "+0275,5"])
    data, audit = isd.parse_isd_csv(path)
    np.testing.assert_array_equal(data.isd_temperature_c, [26., -5.])
    assert data.isd_quality_code.eq("1").all()
    assert audit["usable_qc1_rows"] == 2
    assert data.isd_source_code.eq("4").all()


def test_adapter_rejects_other_stations_and_tampered_provenance(tmp_path):
    path = raw_csv(tmp_path, ["+0260,1"])
    path.with_suffix(".provenance.json").write_text(json.dumps({"url": "https://wrong.example", "sha256": "not-the-file"}))
    with pytest.raises(ValueError, match="URL/hash provenance"):
        isd.parse_isd_csv(path)
    path.with_suffix(".provenance.json").unlink()
    frame = pd.read_csv(path, dtype=str)
    frame["STATION"] = "another_station"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="only nonempty Darwin"):
        isd.parse_isd_csv(path)


def fake_observations():
    return pd.DataFrame({"isd_observation_datetime_utc": pd.to_datetime(["2023-01-01T10:00Z", "2023-01-01T10:30Z"]),
                         "isd_temperature_c": [30., 40.], "isd_quality_code": ["1", "1"],
                         "isd_station_latitude": [-12.414722] * 2, "isd_station_longitude": [130.876667] * 2,
                         "isd_station_elevation_m": [30.] * 2, "isd_source_code": ["4"] * 2,
                         "isd_report_type": ["FM-15"] * 2, "isd_raw_url": ["https://www.ncei.noaa.gov/data/global-hourly/access/2023/94120099999.csv"] * 2,
                         "isd_raw_sha256": ["fixture-hash"] * 2})


def test_only_darwin_fallback_or_farther_station_is_replaced_without_future_matches(monkeypatch):
    monkeypatch.setattr(isd, "load_isd_archive", lambda *args: (fake_observations(), {"files": []}))
    queried_times = []
    def weather(query, cache, max_requests):
        queried_times.extend(query.datetime_utc.tolist())
        background = query.copy()
        background["background_air_temperature_c"] = 27.
        return background, []
    monkeypatch.setattr(isd, "enrich_weather", weather)
    observed_source = "observed_station_residual_plus_ERA5_spatial_background"
    frame = pd.DataFrame({
        "region_id": ["greater_london", isd.REGION_ID, isd.REGION_ID, isd.REGION_ID, isd.REGION_ID, isd.REGION_ID],
        "datetime_utc": pd.to_datetime(["2023-01-01T10:15Z"] * 4 + ["2023-01-01T12:31Z", "2023-01-01T10:15Z"]),
        "latitude": [51.5, -12.414722, -12.404722, -12.404722, -12.414722, 0.],
        "longitude": [0., 130.876667, 130.876667, 130.876667, 130.876667, 130.876667],
        "background_air_temperature_c": [25.] * 6, "air_temperature_c": [25., 25., 28.5, 31., 25., 25.],
        "air_temperature_source": [isd.ERA5_ONLY_SOURCE, isd.ERA5_ONLY_SOURCE, observed_source, observed_source, isd.ERA5_ONLY_SOURCE, isd.ERA5_ONLY_SOURCE],
        "station_id": ["", "", "CLOSE", "DISTANT", "", ""],
        "station_distance_km": [np.nan, np.nan, 0.1, 80., np.nan, np.nan],
        "station_age_minutes": [np.nan, np.nan, 10., 10., np.nan, np.nan],
        "lst_c": [40.] * 6,
    }, index=[10, 20, 30, 40, 50, 60])
    output, audit = isd.apply_darwin_fallback(frame, "unused", "unused")
    np.testing.assert_array_equal(output.air_temperature_c, [25., 28., 28.5, 28., 25., 25.])
    assert output.index.equals(frame.index)
    assert output.loc[20, "station_age_minutes"] == 15
    assert output.loc[20, "station_observation_source_code"] == "4"
    assert output.loc[20, "station_temperature_quality_code"] == "1"
    assert output.loc[30, "station_id"] == "CLOSE"
    assert output.loc[40, "air_temperature_source"] == isd.AIR_SOURCE
    assert queried_times == [pd.Timestamp("2023-01-01T10:00Z")]
    assert audit["changed_rows"] == 2
    assert audit["changed_from_era5_only"] == 1
    assert audit["changed_from_farther_observed_station"] == 1
    pd.testing.assert_series_equal(output.loc[10, frame.columns], frame.loc[10], check_names=False)
    pd.testing.assert_series_equal(output.lst_c, frame.lst_c)


def test_no_darwin_rows_never_loads_station_or_weather(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Unrelated regions should not trigger airport processing")
    monkeypatch.setattr(isd, "load_isd_archive", fail)
    frame = pd.DataFrame({"region_id": ["greater_london"], "datetime_utc": pd.to_datetime(["2023-01-01T10:00Z"]),
                          "latitude": [51.5], "longitude": [0.], "background_air_temperature_c": [10.],
                          "air_temperature_c": [10.], "air_temperature_source": [isd.ERA5_ONLY_SOURCE],
                          "station_id": [""], "station_distance_km": [np.nan], "station_age_minutes": [np.nan]})
    output, audit = isd.apply_darwin_fallback(frame, "unused", "unused")
    pd.testing.assert_frame_equal(output, frame)
    assert audit["changed_rows"] == 0
