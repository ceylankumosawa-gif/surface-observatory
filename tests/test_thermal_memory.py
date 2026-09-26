"""Causality, unit and missing-history tests; fixtures make no network calls."""
import numpy as np
import pandas as pd
import pytest
import requests

from lst_pilot import thermal_memory as memory
from lst_pilot.weather import WeatherResponseError


TARGET = pd.Timestamp("2023-09-07T12:30:00Z")
CUTOFF = TARGET.floor("h")


def history():
    times = pd.date_range(CUTOFF - pd.Timedelta(hours=23), CUTOFF, freq="h")
    return pd.DataFrame({"hour_ending_utc": times, "shortwave_down_w_m2": 100.,
                         "era5_longwave_down_w_m2": 300., "background_air_temperature_c": np.arange(24, dtype=float)})


def aggregate(table=None, target=TARGET, units=None):
    return memory.aggregate_history(history() if table is None else table, target,
                                    units=memory.INPUT_UNITS if units is None else units)


def test_completed_hour_energy_means_ranges_and_signed_changes():
    result = aggregate()
    assert result["memory_history_end_utc"] == CUTOFF
    assert result["memory_interval_start_utc"] == CUTOFF - pd.Timedelta(hours=24)
    assert result["memory_excluded_partial_hour_minutes"] == 30
    assert result["memory_shortwave_energy_6h_j_m2"] == 100 * 6 * 3600
    assert result["memory_shortwave_energy_12h_j_m2"] == 100 * 12 * 3600
    assert result["memory_shortwave_energy_24h_j_m2"] == 100 * 24 * 3600
    assert result["memory_longwave_mean_24h_w_m2"] == 300  # A flux, never divided by 3600 again.
    assert result["memory_air_mean_6h_c"] == 20.5
    assert result["memory_air_range_6h_c"] == 5
    assert result["memory_air_mean_24h_c"] == 11.5
    assert result["memory_air_range_24h_c"] == 23
    assert result["memory_air_change_3h_c"] == 3
    assert result["memory_air_change_6h_c"] == 6
    assert result["memory_status"] == "complete"


def test_future_and_partial_hour_never_contribute_and_midnight_uses_previous_day():
    future = history().iloc[[-1]].copy()
    future.hour_ending_utc = CUTOFF + pd.Timedelta(hours=1)
    future.loc[:, list(memory.INPUT_UNITS)] = 1e9
    result = aggregate(pd.concat([history(), future], ignore_index=True))
    expected = aggregate(target=CUTOFF)
    for name in memory.FEATURE_UNITS:
        assert result[name] == expected[name]
    assert result["memory_future_rows_excluded"] == 1
    moved = history()
    moved.hour_ending_utc -= pd.Timedelta(hours=12)
    midnight = aggregate(moved, TARGET - pd.Timedelta(hours=12))
    assert midnight["memory_history_end_utc"] == pd.Timestamp("2023-09-07T00:00Z")
    assert midnight["memory_shortwave_energy_24h_j_m2"] == 8_640_000


def test_missing_values_do_not_become_partial_energy_or_rescaled_means():
    data = history()
    data.loc[22, "shortwave_down_w_m2"] = np.nan
    data.loc[0, "era5_longwave_down_w_m2"] = np.nan
    result = aggregate(data)
    assert np.isnan(result["memory_shortwave_energy_6h_j_m2"])
    assert result["memory_shortwave_6h_valid_hours"] == 5
    assert result["memory_shortwave_6h_coverage_fraction"] == 5 / 6
    assert result["memory_shortwave_6h_status"] == "missing_history"
    assert result["memory_longwave_mean_6h_w_m2"] == 300
    assert np.isnan(result["memory_longwave_mean_24h_w_m2"])
    assert result["memory_status"] == "missing_history"


def test_empty_history_has_zero_coverage_and_no_invented_features():
    result = aggregate(history().iloc[:0])
    assert all(np.isnan(result[name]) for name in memory.FEATURE_UNITS)
    assert result["memory_shortwave_24h_coverage_fraction"] == 0
    assert result["memory_air_change_6h_valid_endpoints"] == 0
    assert result["memory_status"] == "missing_history"


def test_missing_timestamp_is_not_shifted_and_change_needs_both_endpoints():
    data = history().drop(index=20)
    result = aggregate(data)
    assert np.isnan(result["memory_air_change_3h_c"])
    assert result["memory_air_change_3h_valid_endpoints"] == 1
    assert result["memory_air_change_6h_c"] == 6
    assert np.isnan(result["memory_air_mean_6h_c"])
    # An intermediate gap does not fabricate/change a two-endpoint difference.
    result = aggregate(history().drop(index=22))
    assert result["memory_air_change_3h_c"] == 3
    assert result["memory_air_change_3h_coverage_fraction"] == 1


@pytest.mark.parametrize("column,unit", [("shortwave_down_w_m2", "J/m²"), ("era5_longwave_down_w_m2", "J/m²"), ("background_air_temperature_c", "K")])
def test_accumulated_energy_and_kelvin_cannot_masquerade_as_expected_inputs(column, unit):
    with pytest.raises(WeatherResponseError, match="unit mismatch"):
        aggregate(units={**memory.INPUT_UNITS, column: unit})


def test_zero_night_shortwave_is_valid_but_negative_flux_is_missing():
    data = history()
    data.shortwave_down_w_m2 = 0
    assert aggregate(data)["memory_shortwave_energy_24h_j_m2"] == 0
    data.loc[23, "shortwave_down_w_m2"] = -1
    assert np.isnan(aggregate(data)["memory_shortwave_energy_6h_j_m2"])
    data.loc[23, "era5_longwave_down_w_m2"] = 0
    assert np.isnan(aggregate(data)["memory_longwave_mean_6h_w_m2"])


def test_ambiguous_naive_duplicate_and_subhourly_timestamps_fail():
    with pytest.raises(ValueError, match="timezone-aware"):
        aggregate(target="2023-09-07T12:30")
    with pytest.raises(ValueError, match="unique whole"):
        aggregate(pd.concat([history(), history().iloc[[-1]]]))
    data = history()
    data.loc[0, "hour_ending_utc"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="unique whole"):
        aggregate(data)


def fake_adapters(monkeypatch, *, missing_weather=False, missing_radiation=False):
    calls = {"weather": [], "radiation": []}
    def fetch(latitude, longitude, start, end, cache):
        calls["weather"].append((latitude, longitude, start, end))
        if missing_weather:
            raise requests.ConnectionError("deliberate source outage")
        data = history().rename(columns={"hour_ending_utc": "weather_datetime_utc"})
        future = data.iloc[[-1]].copy()
        future.weather_datetime_utc += pd.Timedelta(hours=1)
        future.background_air_temperature_c = 999
        future.shortwave_down_w_m2 = 99999
        data = pd.concat([data, future], ignore_index=True)
        return data, {"response": {"latitude": latitude, "longitude": longitude, "hourly_units": memory.INPUT_UNITS},
                      "source": "https://archive-api.open-meteo.com/v1/archive", "validation": {"checked": True}}
    def radiation(frame, cache, max_hours, allow_provisional):
        calls["radiation"].append(frame.copy())
        out = frame.copy()
        out["era5_longwave_down_w_m2"] = np.nan if missing_radiation else 300.
        out["era5_longwave_down_w_m2_status"] = "source_unavailable" if missing_radiation else "coarse_reanalysis"
        out.attrs["radiation_context"] = {"errors": ["deliberate outage"] if missing_radiation else [], "downloaded_bytes": 0,
                                           "objects": [{"key": "fake-public-hour", "sha256": "testhash", "cached": True}]}
        return out
    monkeypatch.setattr(memory.weather, "fetch_archive", fetch)
    monkeypatch.setattr(memory.radiation, "add_radiation", radiation)
    return calls


def samples():
    return pd.DataFrame({"datetime_utc": [TARGET, TARGET], "latitude": [51.5, 51.5], "longitude": [-.1, -.1],
                         "air_temperature_c": [10., 40.], "station_id": ["one", "two"]}, index=[7, 7])


def test_adapter_history_is_independent_of_current_station_override_and_preserves_rows(tmp_path, monkeypatch):
    calls = fake_adapters(monkeypatch)
    source = samples()
    result, report = memory.add_thermal_memory(source, tmp_path)
    pd.testing.assert_frame_equal(result[source.columns], source)
    assert result.index.tolist() == [7, 7]
    for name in memory.FEATURE_UNITS:
        assert result[name].iloc[0] == result[name].iloc[1]
    assert len(calls["weather"]) == len(calls["radiation"]) == 1
    assert len(calls["radiation"][0]) == 24
    assert calls["radiation"][0].datetime_utc.max() == CUTOFF
    assert report["current_station_or_manual_override_applied_to_history"] is False
    assert report["radiation"]["objects"][0]["sha256"] == "testhash"
    assert report["complete_rows"] == 2


@pytest.mark.parametrize("options", [{"max_hours": 23}, {"cache_budget_bytes": 1}, {"max_samples": 1}, {"max_weather_requests": 0}])
def test_budgets_are_checked_before_any_source_reads(tmp_path, monkeypatch, options):
    calls = fake_adapters(monkeypatch)
    with pytest.raises(ValueError):
        memory.add_thermal_memory(samples(), tmp_path, **options)
    assert not calls["weather"] and not calls["radiation"]


@pytest.mark.parametrize("missing_weather,missing_radiation", [(True, False), (False, True)])
def test_source_outage_keeps_missing_features_and_provenance(tmp_path, monkeypatch, missing_weather, missing_radiation):
    fake_adapters(monkeypatch, missing_weather=missing_weather, missing_radiation=missing_radiation)
    result, report = memory.add_thermal_memory(samples(), tmp_path)
    assert result.memory_status.eq("missing_history").all()
    assert report["errors"]
    if missing_weather:
        assert result.memory_air_mean_24h_c.isna().all()
        assert result.memory_longwave_mean_24h_w_m2.eq(300).all()
    else:
        assert result.memory_longwave_mean_24h_w_m2.isna().all()
        assert result.memory_shortwave_energy_24h_j_m2.eq(8_640_000).all()


def test_weather_unit_contract_error_is_fatal_not_replaced(tmp_path, monkeypatch):
    calls = fake_adapters(monkeypatch)
    def bad(*args):
        raise WeatherResponseError("wrong source units")
    monkeypatch.setattr(memory.weather, "fetch_archive", bad)
    with pytest.raises(WeatherResponseError, match="wrong source units"):
        memory.add_thermal_memory(samples(), tmp_path)
    assert not calls["radiation"]
