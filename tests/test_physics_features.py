import hashlib
import json
import socket

import numpy as np
import pandas as pd
import pytest

from lst_pilot import physics_features as physics


def frame(**changes):
    row = {
        "sample_id": "sample-one", "datetime_utc": pd.Timestamp("2022-01-02T00:30:00Z"),
        "air_temperature_c": 26.85, "albedo_proxy": .2,
        "shortwave_down_w_m2": 120., "era5_longwave_down_w_m2": 400.,
        "weather_datetime_utc": pd.Timestamp("2022-01-02T00:00:00Z"),
        "radiation_era5_time_utc": pd.Timestamp("2022-01-02T00:00:00Z"),
        "optical_latest_source_utc": pd.Timestamp("2022-01-01T12:00:00Z"),
        "memory_history_end_utc": pd.Timestamp("2022-01-02T00:00:00Z"),
        "memory_shortwave_energy_6h_j_m2": 100. * 6 * 3600,
        "memory_shortwave_6h_status": "complete", "memory_shortwave_6h_valid_hours": 6,
        "memory_shortwave_6h_coverage_fraction": 1.,
        "weather_grid_latitude": 51.5, "weather_grid_longitude": 0.,
    }
    row.update(changes)
    return pd.DataFrame([row], index=[37])


def cache(tmp_path, *, values=None, times=None, modify=None):
    root = tmp_path / "weather"
    root.mkdir(exist_ok=True)
    request = {"latitude": 51.5, "longitude": 0., "models": "era5", "timezone": "UTC",
               "elevation": "nan", "cell_selection": "nearest"}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
    record = {
        "source": physics.WEATHER_SOURCE, "model": "ERA5", "request": request,
        "response": {"latitude": 51.5, "longitude": 0., "utc_offset_seconds": 0,
                     "hourly_units": {"shortwave_radiation": "W/m²"},
                     "hourly": {"time": times or ["2022-01-01T19:00", "2022-01-01T20:00",
                         "2022-01-01T21:00", "2022-01-01T22:00", "2022-01-01T23:00",
                         "2022-01-02T00:00", "2022-01-02T01:00"],
                         "shortwave_radiation": values if values is not None else [20., 40., 60., 80., 100., 120., 99999.]}}
    }
    if modify:
        modify(record)
    path = root / f"era5_{key}.json"
    path.write_text(json.dumps(record))
    return key, path


def test_three_formulas_and_unchanged_input_identity():
    original = pd.concat([frame(), frame(sample_id="sample-two")])
    before = original.copy(deep=True)
    result, audit = physics.add_physics_features(original)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(result[original.columns], before)
    assert result.index.tolist() == [37, 37]
    assert result[physics.FEATURES[0]].tolist() == [96., 96.]
    assert result[physics.FEATURES[1]].iloc[0] == pytest.approx(400. - physics.SIGMA_W_M2_K4 * 300.**4)
    assert result[physics.FEATURES[2]].tolist() == [80., 80.]
    assert result[physics.RAW_HISTORY_FEATURE].tolist() == [100., 100.]
    assert result.physics_complete.all()
    assert audit["downloaded_bytes"] == 0
    assert audit["history_source_counts"] == {"existing_six_hour_energy": 2}


def test_reflective_endpoint_preserves_raw_control_without_division():
    result, _ = physics.add_physics_features(frame(albedo_proxy=1.))
    assert result[physics.FEATURES[0]].iloc[0] == 0
    assert result[physics.FEATURES[2]].iloc[0] == 0
    assert result[physics.RAW_HISTORY_FEATURE].iloc[0] == 100
    assert result.physics_complete.iloc[0]


@pytest.mark.parametrize("field,value", [
    ("albedo_proxy", -0.001), ("albedo_proxy", 1.001), ("albedo_proxy", np.inf),
    ("air_temperature_c", -273.15), ("air_temperature_c", np.nan),
    ("shortwave_down_w_m2", -1.), ("shortwave_down_w_m2", np.inf),
    ("era5_longwave_down_w_m2", 0.), ("era5_longwave_down_w_m2", -1.),
    ("memory_shortwave_energy_6h_j_m2", -1.),
])
def test_invalid_physical_input_is_unavailable_without_clipping(field, value):
    original = frame(**{field: value})
    result, audit = physics.add_physics_features(original)
    assert not result.physics_complete.iloc[0]
    assert result.physics_unavailable_reason.iloc[0]
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert audit["output_rows"] == 1


@pytest.mark.parametrize("field,value", [
    ("memory_shortwave_6h_status", "missing_history"),
    ("memory_shortwave_6h_valid_hours", 5),
    ("memory_shortwave_6h_coverage_fraction", 5 / 6),
    ("memory_history_end_utc", pd.Timestamp("2022-01-02T01:00Z")),
    ("memory_history_end_utc", pd.Timestamp("2022-01-01T23:00Z")),
])
def test_all_six_hour_proof_required_no_partial_or_shifted_substitution(field, value):
    result, _ = physics.add_physics_features(frame(**{field: value}, shortwave_down_mean3_w_m2=77.))
    assert pd.isna(result[physics.FEATURES[2]].iloc[0])
    assert pd.isna(result[physics.RAW_HISTORY_FEATURE].iloc[0])


def test_source_times_are_individually_causal_and_exact_hour():
    future = pd.Timestamp("2022-01-02T01:00Z")
    for field in ("optical_latest_source_utc", "optical_source_datetime_utc",
                  "weather_datetime_utc", "radiation_era5_time_utc"):
        result, _ = physics.add_physics_features(frame(**{field: future}))
        assert not result.physics_complete.iloc[0]
    result, _ = physics.add_physics_features(frame(optical_latest_source_utc=pd.NaT,
        optical_source_datetime_utc=pd.Timestamp("2021-12-25T12:00Z")))
    assert result.physics_complete.iloc[0]
    missing, _ = physics.add_physics_features(frame(optical_latest_source_utc=pd.NaT))
    assert not missing.physics_complete.iloc[0]


def test_target_exact_hour_uses_interval_that_has_just_finished():
    result, _ = physics.add_physics_features(frame(datetime_utc=pd.Timestamp("2022-01-02T00:00Z")))
    assert result.physics_complete.iloc[0]
    assert result.physics_history_start_utc.iloc[0] == pd.Timestamp("2022-01-01T18:00Z")
    assert result.physics_history_end_utc.iloc[0] == pd.Timestamp("2022-01-02T00:00Z")


def test_missing_target_time_cannot_produce_physics_complete():
    result, _ = physics.add_physics_features(frame(datetime_utc=pd.NaT))
    assert not result.physics_complete.iloc[0]
    assert "missing_target_time" in result.physics_unavailable_reason.iloc[0]


def test_unit_timezone_and_output_overwrite_contracts():
    wrong = dict(physics.INPUT_UNITS, memory_shortwave_energy_6h_j_m2="W/m²")
    with pytest.raises(ValueError, match="unit mismatch"):
        physics.add_physics_features(frame(), units=wrong)
    with pytest.raises(ValueError, match="timezone-aware"):
        physics.add_physics_features(frame(datetime_utc="2022-01-02 00:30"))
    with pytest.raises(ValueError, match="overwrite"):
        physics.add_physics_features(frame(physics_complete=True))


def test_cache_rebuild_is_exact_six_hours_excludes_future_and_never_networks(tmp_path, monkeypatch):
    key, path = cache(tmp_path)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network forbidden")))
    original = frame(memory_shortwave_energy_6h_j_m2=np.nan,
                     memory_shortwave_6h_status="missing_history", weather_cache_key=key)
    result, audit = physics.add_physics_features(original, cache_root=tmp_path)
    assert result.physics_complete.iloc[0]
    assert result[physics.RAW_HISTORY_FEATURE].iloc[0] == 70.
    assert result[physics.FEATURES[2]].iloc[0] == 56.
    assert result.physics_history_source.iloc[0] == "existing_weather_cache"
    assert result.physics_history_cache_sha256.iloc[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit["cache_read_bytes"] == path.stat().st_size
    assert audit["downloaded_bytes"] == 0


@pytest.mark.parametrize("values", [[20., 40., None, 80., 100., 120., 99999.],
                                   [20., 40., -1., 80., 100., 120., 99999.]])
def test_cache_gap_or_negative_hour_does_not_get_imputed(tmp_path, values):
    key, _ = cache(tmp_path, values=values)
    result, _ = physics.add_physics_features(frame(memory_shortwave_energy_6h_j_m2=np.nan,
        weather_cache_key=key), cache_root=tmp_path)
    assert not result.physics_complete.iloc[0]
    assert result.physics_history_valid_hours.iloc[0] == 5
    assert result.physics_history_unavailable_reason.iloc[0] == "cache_incomplete_six_hours"


@pytest.mark.parametrize("changes,expected", [
    ({"weather_grid_latitude": 52.}, "cache_grid_mismatch"),
    ({"shortwave_down_w_m2": 119.}, "cache_current_shortwave_mismatch"),
])
def test_cache_cannot_borrow_another_grid_or_current_weather(tmp_path, changes, expected):
    key, _ = cache(tmp_path)
    result, _ = physics.add_physics_features(frame(memory_shortwave_energy_6h_j_m2=np.nan,
        weather_cache_key=key, **changes), cache_root=tmp_path)
    assert not result.physics_complete.iloc[0]
    assert result.physics_history_unavailable_reason.iloc[0] == expected


@pytest.mark.parametrize("mutation", [
    lambda r: r["request"].update(latitude=52.),
    lambda r: r["response"]["hourly_units"].update(shortwave_radiation="J/m²"),
    lambda r: r["response"].update(utc_offset_seconds=3600),
    lambda r: r["response"]["hourly"]["time"].__setitem__(0, "2022-01-01T20:00"),
    lambda r: r.update(model="ERA5T"),
])
def test_cache_identity_units_duplicate_times_and_source_are_not_silently_repaired(tmp_path, mutation):
    key, _ = cache(tmp_path, modify=mutation)
    result, audit = physics.add_physics_features(frame(memory_shortwave_energy_6h_j_m2=np.nan,
        weather_cache_key=key), cache_root=tmp_path)
    assert not result.physics_complete.iloc[0]
    assert result.physics_history_unavailable_reason.iloc[0] == "cache_invalid"
    assert audit["cache_records"][0]["status"] == "cache_invalid"


def test_cache_missing_path_traversal_and_budget_are_explicit(tmp_path, monkeypatch):
    for key, expected in [("a" * 24, "cache_missing"), ("../../private", "invalid_cache_key")]:
        result, _ = physics.add_physics_features(frame(memory_shortwave_energy_6h_j_m2=np.nan,
            weather_cache_key=key), cache_root=tmp_path)
        assert result.physics_history_unavailable_reason.iloc[0] == expected
    key, _ = cache(tmp_path)
    monkeypatch.setattr(physics, "MAX_CACHE_FILE_BYTES", 10)
    result, audit = physics.add_physics_features(frame(memory_shortwave_energy_6h_j_m2=np.nan,
        weather_cache_key=key), cache_root=tmp_path)
    assert result.physics_history_unavailable_reason.iloc[0] == "cache_byte_budget_exceeded"
    assert audit["cache_read_bytes"] == 0


def test_shared_missing_history_reads_one_cache_file(tmp_path):
    key, path = cache(tmp_path)
    source = pd.concat([frame(memory_shortwave_energy_6h_j_m2=np.nan, weather_cache_key=key)] * 3)
    result, audit = physics.add_physics_features(source, cache_root=tmp_path)
    assert result.physics_complete.all()
    assert audit["cache_read_bytes"] == path.stat().st_size
    assert len(audit["cache_records"]) == 1
