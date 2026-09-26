"""Integration contracts: leakage barriers, station timing and checkpoint identity."""

import numpy as np
import pandas as pd
import pytest

import lst_pilot.assemble as assembly
from lst_pilot.model import select_features
from lst_pilot.weather import WeatherResponseError


def test_checkpoint_identity_changes_with_options_source_version_and_areas(tmp_path, monkeypatch):
    observations, areas = tmp_path / "samples.parquet", tmp_path / "areas.json"
    observations.write_bytes(b"same observation contents")
    areas.write_text('{"areas": []}')
    initial = assembly.assembly_signature(observations, areas, {"terrain": False})
    repeated = assembly.assembly_signature(observations, areas, {"terrain": False})
    terrain = assembly.assembly_signature(observations, areas, {"terrain": True})
    assert initial["sha256"] == repeated["sha256"]
    assert initial["sha256"] != terrain["sha256"]
    output = tmp_path / "run"
    assembly.guard_checkpoints(output, initial)
    assembly.guard_checkpoints(output, repeated)
    with pytest.raises(ValueError, match="processing options or source version changed"):
        assembly.guard_checkpoints(output, terrain)
    monkeypatch.setattr(assembly, "PROCESSING_VERSION", "changed-parser-version")
    assert assembly.assembly_signature(observations, areas, {"terrain": False})["sha256"] != initial["sha256"]
    areas.write_text('{"areas": [{"id": "new_region"}]}')
    assert assembly.assembly_signature(observations, areas, {"terrain": False})["sha256"] != repeated["sha256"]


def test_legacy_checkpoint_cannot_bypass_processing_identity(tmp_path):
    (tmp_path / "context.parquet").write_bytes(b"old")
    with pytest.raises(ValueError, match="Legacy/unversioned"):
        assembly.guard_checkpoints(tmp_path, {"sha256": "new", "specification": {}})


def test_terrain_values_reach_model_with_declared_units():
    frame = pd.DataFrame({"elevation_m": [120.5, np.nan], "slope_deg": [10.0, 0.0], "terrain_relief_300m": [5.0, 0.0], "air_temperature_c": [15.0, 16.0], "climate_class": ["Cfb", "Cfb"], "snow_fraction": [0.0, 1.0]})
    result = assembly.normalise_terrain_columns(frame)
    np.testing.assert_allclose(result.elevation, [120.5, np.nan], equal_nan=True)
    np.testing.assert_array_equal(result.slope, [10.0, 0.0])
    features = select_features(result.columns)
    assert {"elevation", "slope", "terrain_relief_300m"}.issubset(features)
    assert "snow_fraction" not in features
    frame["elevation"] = [999.0, np.nan]
    with pytest.raises(ValueError, match="Conflicting terrain"):
        assembly.normalise_terrain_columns(frame)


def test_land_only_selection_removes_water_mixed_and_unknown_qa():
    source = pd.DataFrame({"water_fraction": [0.0, 1.0, 0.05, np.nan], "pixel_id": ["land", "water", "shore", "unknown"]})
    result, audit = assembly.select_land_samples(source)
    assert result.pixel_id.tolist() == ["land"]
    assert audit["removed_rows"] == 3
    assert audit["missing_water_fraction_rows"] == 1
    with pytest.raises(ValueError, match="water_fraction is required"):
        assembly.select_land_samples(source.drop(columns="water_fraction"))


def mock_station_environment(monkeypatch, observations):
    inventory = pd.DataFrame([
        {"station_id": "USW00094044", "name": "SURFRAD reference A", "latitude": 51., "longitude": 0., "elevation_m": 10., "icao": "AAAA", "distance_to_center_km": 0.},
        {"station_id": "USW00054918", "name": "SURFRAD reference B", "latitude": 51., "longitude": 0., "elevation_m": 10., "icao": "BBBB", "distance_to_center_km": 0.},
        {"station_id": "GB000000001", "name": "Independent station", "latitude": 51., "longitude": 0., "elevation_m": 10., "icao": "EGXX", "distance_to_center_km": 1.},
    ])
    monkeypatch.setattr(assembly, "station_inventory", lambda cache: inventory.copy())
    monkeypatch.setattr(assembly, "nearby_stations", lambda inventory, bbox, **kwargs: inventory.copy())
    monkeypatch.setattr(assembly, "region_bbox", lambda area: [-0.1, 50.9, 0.1, 51.1])
    monkeypatch.setattr(assembly, "station_year_available", lambda station, year: {"available": True})
    used = {"stations": [], "years": [], "weather_times": []}

    def series(station, years, cache):
        used["stations"].append(station)
        used["years"].extend(years)
        assert station not in assembly.EXCLUDED_AIR_STATION_IDS
        return observations.copy()

    def weather(query, cache, max_requests):
        used["weather_times"].extend(query.datetime_utc.tolist())
        result = query.copy()
        result["background_air_temperature_c"] = 17.0
        return result, []

    monkeypatch.setattr(assembly, "station_series", series)
    monkeypatch.setattr(assembly, "enrich_weather", weather)
    return used


def test_station_correction_is_backward_bounded_and_reference_independent(tmp_path, monkeypatch):
    observations = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2023-06-01T10:00Z", "2023-06-01T10:30Z"]), "air_temperature_c": [20., 99.]})
    used = mock_station_environment(monkeypatch, observations)
    data = pd.DataFrame({"region_id": ["test"] * 3, "datetime_utc": pd.to_datetime(["2023-06-01T10:15Z", "2023-06-01T12:31Z", "2023-06-01T10:15Z"]), "latitude": [51., 51., 54.], "longitude": [0., 0., 0.], "background_air_temperature_c": [15., 15., 15.]})
    result, audit = assembly.attach_stations(data, [{"id": "test"}], tmp_path, stations_per_region=2, max_distance_km=100)
    # First pixel: ERA5 pixel 15 + observed station 20 - station ERA5 17 = 18.
    # Second pixel is stale; third is too distant. Both retain explicit background.
    np.testing.assert_array_equal(result.air_temperature_c, [18., 15., 15.])
    assert result.loc[0, "station_age_minutes"] == 15
    assert result.loc[0, "observed_station_air_temperature_c"] == 20
    assert result.loc[1:, "air_temperature_source"].eq("ERA5_only_no_timely_station").all()
    assert used["stations"] == ["GB000000001"]
    excluded = {entry["station_id"] for entry in audit if entry["status"] == "excluded_reference_station"}
    assert excluded == assembly.EXCLUDED_AIR_STATION_IDS
    assert all(time <= pd.Timestamp("2023-06-01T10:15Z") for time in used["weather_times"])


def test_backward_station_lookup_includes_previous_year(tmp_path, monkeypatch):
    observations = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2023-12-31T23:45Z", "2024-01-01T00:30Z"]), "air_temperature_c": [20., 99.]})
    used = mock_station_environment(monkeypatch, observations)
    data = pd.DataFrame({"region_id": ["test"], "datetime_utc": pd.to_datetime(["2024-01-01T00:15Z"]), "latitude": [51.], "longitude": [0.], "background_air_temperature_c": [15.]})
    result, _ = assembly.attach_stations(data, [{"id": "test"}], tmp_path)
    assert set(used["years"]) == {2023, 2024}
    assert result.air_temperature_c.item() == 18
    assert result.station_age_minutes.item() == 30


def test_bad_station_background_weather_units_are_not_hidden_by_fallback(tmp_path, monkeypatch):
    observations = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2023-06-01T10:00Z"]), "air_temperature_c": [20.]})
    mock_station_environment(monkeypatch, observations)
    def bad_weather(*args, **kwargs):
        raise WeatherResponseError("Weather unit mismatch")
    monkeypatch.setattr(assembly, "enrich_weather", bad_weather)
    data = pd.DataFrame({"region_id": ["test"], "datetime_utc": pd.to_datetime(["2023-06-01T10:15Z"]), "latitude": [51.], "longitude": [0.], "background_air_temperature_c": [15.]})
    with pytest.raises(WeatherResponseError, match="unit mismatch"):
        assembly.attach_stations(data, [{"id": "test"}], tmp_path)


def test_parsed_station_cache_tracks_raw_contents(tmp_path, monkeypatch):
    raw = tmp_path / "raw.psv"
    raw.write_text("first raw version")
    calls = []
    monkeypatch.setattr(assembly, "download_station_year", lambda *args: raw)
    def parse(path):
        calls.append(path.read_text())
        return pd.DataFrame({"timestamp_utc": pd.to_datetime(["2023-01-01T00:00Z"]), "air_temperature_c": [float(len(calls))]})
    monkeypatch.setattr(assembly, "parse_station_file", parse)
    first = assembly.station_series("GB000000001", [2023], tmp_path)
    reused = assembly.station_series("GB000000001", [2023], tmp_path)
    assert len(calls) == 1
    assert first.air_temperature_c.item() == reused.air_temperature_c.item()
    raw.write_text("updated raw version")
    refreshed = assembly.station_series("GB000000001", [2023], tmp_path)
    assert len(calls) == 2
    assert refreshed.air_temperature_c.item() == 2
