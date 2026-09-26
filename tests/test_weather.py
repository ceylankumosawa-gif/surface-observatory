"""Response-contract tests use tiny fixtures and no network or model fitting."""

import numpy as np
import pandas as pd
import pytest

import lst_pilot.weather as weather_module
from lst_pilot.weather import EXPECTED_UNITS, VARIABLES, grid_distance_km, validate_weather_response


def response():
    return {"latitude": 51.5, "longitude": 0.0, "utc_offset_seconds": 0,
            "hourly_units": {"time": "iso8601", **EXPECTED_UNITS},
            "hourly": {"time": ["2023-01-01T00:00", "2023-01-01T01:00"], **{name: [1.0, 2.0] for name in VARIABLES}}}


def test_expected_units_pass_and_optional_null_soil_is_reported():
    data = response()
    data["hourly"]["soil_moisture_0_to_7cm"] = [None, None]
    result = validate_weather_response(data)
    assert result["all_missing_variables"] == ["soil_moisture_0_to_7cm"]
    assert "not weather-station" in result["lag_source"]


@pytest.mark.parametrize("variable,unit", [("temperature_2m", "K"), ("wind_speed_10m", "km/h"), ("surface_pressure", "Pa"), ("shortwave_radiation", "J/m²"), ("soil_moisture_0_to_7cm", "%")])
def test_unit_changes_are_rejected(variable, unit):
    data = response()
    data["hourly_units"][variable] = unit
    with pytest.raises(ValueError, match="unit mismatch"):
        validate_weather_response(data)


def test_missing_variable_or_missing_temperature_is_fatal():
    data = response()
    del data["hourly"]["temperature_2m"]
    with pytest.raises(ValueError, match="missing requested variables"):
        validate_weather_response(data)
    data = response()
    data["hourly"]["temperature_2m"][1] = None
    with pytest.raises(ValueError, match="missing air temperature"):
        validate_weather_response(data)


def test_shifted_timezone_is_rejected():
    data = response()
    data["utc_offset_seconds"] = 3600
    with pytest.raises(ValueError, match="not UTC"):
        validate_weather_response(data)


def test_unit_validation_requires_metadata_and_matching_lengths():
    data = response()
    del data["hourly_units"]["rain"]
    with pytest.raises(ValueError, match="unit mismatch"):
        validate_weather_response(data)
    data = response()
    data["hourly"]["rain"] = [1.0]
    with pytest.raises(ValueError, match="timestamp count"):
        validate_weather_response(data)


def test_grid_distance_handles_dateline_and_exact_collocation():
    distances = grid_distance_km([0, 51], [179.9, 0], [0, 51], [-179.9, 0])
    np.testing.assert_allclose(distances, [22.239016, 0], atol=1e-5)


def test_weather_requests_only_sample_span_plus_past_lag_context(tmp_path, monkeypatch):
    requests = []
    def fetch(latitude, longitude, start, end, cache):
        requests.append((start, end))
        frame = pd.DataFrame({"weather_datetime_utc": pd.to_datetime(["2023-02-20T12:00Z", "2023-02-22T06:00Z"]), "weather_grid_latitude": [latitude] * 2, "weather_grid_longitude": [longitude] * 2, "background_air_temperature_c": [10., 11.]})
        return frame, {"response": {"latitude": latitude, "longitude": longitude, "hourly_units": EXPECTED_UNITS}, "validation": {}}
    monkeypatch.setattr(weather_module, "fetch_archive", fetch)
    samples = pd.DataFrame({"datetime_utc": pd.to_datetime(["2023-02-20T12:30Z", "2023-02-22T06:30Z"]), "latitude": [51.5, 51.5], "longitude": [0., 0.]})
    output, _ = weather_module.enrich_weather(samples, tmp_path)
    assert requests == [(pd.Timestamp("2023-02-16", tz="UTC"), pd.Timestamp("2023-02-22", tz="UTC"))]
    assert output.weather_age_minutes.eq(30).all()


def test_validated_cache_is_written_atomically_and_reused(tmp_path, monkeypatch):
    import os
    import stat
    calls = []
    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return response()
    def get(*args, **kwargs):
        calls.append(kwargs["params"])
        return Response()
    monkeypatch.setattr(weather_module.requests, "get", get)
    monkeypatch.setattr(weather_module.time, "sleep", lambda seconds: None)
    previous_umask = os.umask(0o077)
    try:
        first, _ = weather_module.fetch_archive(51.5, 0., "2023-01-01", "2023-01-01", tmp_path)
    finally:
        os.umask(previous_umask)
    second, _ = weather_module.fetch_archive(51.5, 0., "2023-01-01", "2023-01-01", tmp_path)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert len(list((tmp_path / "weather").glob("*.json"))) == 1
    cached = next((tmp_path / "weather").glob("*.json"))
    assert stat.S_IMODE(cached.stat().st_mode) == 0o664
    assert cached.stat().st_gid == (tmp_path / "weather").stat().st_gid
    assert not list((tmp_path / "weather").glob("*.partial"))


def test_invalid_units_never_enter_the_weather_cache(tmp_path, monkeypatch):
    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            data = response()
            data["hourly_units"]["temperature_2m"] = "K"
            return data
    monkeypatch.setattr(weather_module.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="unit mismatch"):
        weather_module.fetch_archive(51.5, 0., "2023-01-01", "2023-01-01", tmp_path)
    assert not list((tmp_path / "weather").glob("*.json"))
