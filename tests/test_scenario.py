"""Scenario provenance and separation of the unvalidated night baseline."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from lst_pilot import scenario
from lst_pilot.raster import validate_time
from lst_pilot.weather import WeatherResponseError


def test_surface_snapshot_is_seasonal_explicit_and_not_target_temperature_selected():
    scenes = [{"scene_id": "winter", "datetime_utc": "2021-12-20T10:00:00Z"},
              {"scene_id": "summer", "datetime_utc": "2024-07-01T10:00:00Z"}]
    assert scenario.select_surface_scene(scenes, "2030-12-25T23:00:00Z")["scene_id"] == "winter"
    assert scenario.select_surface_scene(scenes, "1901-07-02T23:00:00Z")["scene_id"] == "summer"
    assert scenario.select_surface_scene(scenes, "2030-12-25T23:00:00Z", "summer")["scene_id"] == "summer"
    with pytest.raises(ValueError, match="snapshot listed"):
        scenario.select_surface_scene(scenes, "2030-12-25T23:00:00Z", "external-id")


def test_february_reference_and_scenario_time_do_not_imply_contemporaneous_surface():
    assert scenario.reference_timestamp("2024-02-29T21:17:00Z").isoformat() == "2023-02-28T21:17:00+00:00"
    scene = {"properties": {"datetime": "2024-07-01T10:00:00Z"}}
    target, source, age = validate_time(scene, "1950-01-01T23:00:00Z", "scenario")
    assert age < 0 and target.year == 1950 and source.year == 2024
    with pytest.raises(ValueError):
        validate_time(scene, "1950-01-01T23:00:00Z", "experimental")


def test_night_and_mixed_requests_never_run_any_predictor():
    for elevations in [[-12.], [20.,-2.]]:
        frame = pd.DataFrame({"solar_elevation_deg":elevations})
        def forbidden(*args):
            raise AssertionError("No day or coarse-night prediction should be run")
        with pytest.raises(scenario.UnsupportedNighttime, match="withdrawn"):
            scenario.predict_with_night_baseline(frame, {}, forbidden)


def test_night_renderer_stops_before_model_or_remote_data(tmp_path):
    from lst_pilot.raster import render_raster
    from pyproj import Transformer
    from shapely.geometry import mapping
    from shapely.ops import transform
    polygon = mapping(transform(Transformer.from_crs(27700,4326,always_xy=True).transform, box(530800,178400,532800,180400)))
    region = {"id":"greater_london","epsg":27700,"extent_m":[492800,138400,572800,218400]}
    scene = {"properties":{"datetime":"2023-09-07T10:52:00Z"}}
    with pytest.raises(scenario.UnsupportedNighttime):
        render_raster(region, polygon, scene, "2025-09-07T00:00:00Z", tmp_path/"output", tmp_path/"cache", tmp_path/"absent-model", mode="scenario", air_override=20)
    assert not (tmp_path/"output").exists() and not (tmp_path/"cache").exists()


def test_daylight_guard_uses_utc_and_conservative_whole_area_support():
    area = box(-97.1,43.4,-96.4,43.9)
    with pytest.raises(scenario.UnsupportedNighttime):
        scenario.require_daylight_area("2026-09-08T12:00:00Z", area)
    scenario.require_daylight_area("2026-09-08T18:00:00Z", area)


@pytest.fixture
def scenario_frame():
    frame = pd.DataFrame({"datetime_utc": pd.to_datetime(["2035-07-03T12:00:00Z"]*2),
                         "latitude": [0., .01], "longitude": [0., .01],
                         "solar_elevation_deg": [30., 30.], "_raster_position": [4, 9]})
    grid = SimpleNamespace(epsg=3857, polygon=box(0, 0, 1000, 1000))
    return frame, grid


def stub_weather(frame, target, cache, need_skin):
    result = frame.copy()
    result["datetime_utc"] = target
    result["background_air_temperature_c"] = [20., 21.]
    result["era5_skin_temperature_c"] = [17., 18.]
    return result, [{"time": str(target)}], {}, {"dataset": "test skin"}


def test_reference_weather_is_explicit_and_reported_air_anchor_is_preserved(monkeypatch, scenario_frame):
    frame, grid = scenario_frame
    monkeypatch.setattr(scenario, "_weather_at", stub_weather)
    monkeypatch.setattr(scenario, "enrich_weather", lambda *a, **k: (pd.DataFrame({"background_air_temperature_c": [20.]}), []))
    result, context, *_ = scenario.prepare_scenario(frame, grid, "2035-07-03T12:00:00Z", "/unused", 30., now="2026-09-01T00:00:00Z")
    assert context["weather_basis"] == "seasonal_reference"
    assert context["weather_reference_datetime_utc"] == "2023-07-03T12:00:00+00:00"
    assert context["reference_is_climatology"] is False
    assert result.datetime_utc.dt.year.eq(2035).all()
    assert result.weather_reference_datetime_utc.dt.year.eq(2023).all()
    np.testing.assert_array_equal(result.air_temperature_c, [30., 31.])
    np.testing.assert_array_equal(result._raster_position, [4, 9])
    assert result.solar_elevation_deg.eq(30).all()  # Requested geometry stays intact.


def test_existing_hour_uses_actual_weather_and_bad_units_are_not_silently_replaced(monkeypatch, scenario_frame):
    frame, grid = scenario_frame
    monkeypatch.setattr(scenario, "_weather_at", stub_weather)
    monkeypatch.setattr(scenario, "enrich_weather", lambda *a, **k: (pd.DataFrame({"background_air_temperature_c": [20.]}), []))
    result, context, *_ = scenario.prepare_scenario(frame, grid, "2025-07-03T12:17:00Z", "/unused", 30., now="2026-09-01T00:00:00Z")
    assert context["weather_basis"] == "actual_reanalysis"
    assert context["weather_reference_datetime_utc"] == "2025-07-03T12:00:00+00:00"
    assert result.datetime_utc.dt.minute.eq(17).all()
    def wrong_units(*a, **k):
        raise WeatherResponseError("Unexpected Kelvin weather response")
    monkeypatch.setattr(scenario, "_weather_at", wrong_units)
    with pytest.raises(WeatherResponseError):
        scenario.prepare_scenario(frame, grid, "2025-07-03T12:00:00Z", "/unused", 30., now="2026-09-01T00:00:00Z")
