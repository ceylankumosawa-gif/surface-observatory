"""Independent, network-free checks for the historical patch/source contract."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lst_global import patch, receipts
from lst_global.model import FEATURES, NUMERIC_FEATURES
from lst_pilot import stations, weather


def weather_fixture(cache, stamp, latitude=51.75, longitude=-1.25):
    params = dict(latitude=latitude, longitude=longitude,
                  start_date=str(stamp.floor("D") - pd.Timedelta(days=4))[:10],
                  end_date=str(stamp.floor("D"))[:10], hourly=",".join(weather.VARIABLES),
                  models="era5", timezone="UTC", wind_speed_unit="ms", temperature_unit="celsius",
                  precipitation_unit="mm", elevation="nan", cell_selection="nearest")
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:24]
    times = pd.date_range(pd.Timestamp(params["start_date"], tz="UTC"), stamp.floor("D") + pd.Timedelta(hours=23), freq="h")
    constants = dict(temperature_2m=10., relative_humidity_2m=60., dew_point_2m=3.,
                     wind_speed_10m=2., wind_direction_10m=90., surface_pressure=1000., cloud_cover=25.,
                     shortwave_radiation=200., direct_radiation=140., diffuse_radiation=60.,
                     precipitation=0., rain=0., soil_moisture_0_to_7cm=.2)
    record = {"source": weather.URL, "request": params,
              "response": {"latitude": latitude, "longitude": longitude, "utc_offset_seconds": 0,
                           "hourly_units": weather.EXPECTED_UNITS,
                           "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                                      **{name: [value] * len(times) for name, value in constants.items()}}}}
    path = cache / "weather" / f"era5_{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return key, path


def source_fixture(tmp_path, station=True):
    stamp = pd.Timestamp("2023-06-21T12:00Z")
    key, path = weather_fixture(tmp_path, stamp)
    frame = pd.DataFrame({name: [0.] for name in NUMERIC_FEATURES})
    frame["climate_class"] = "Cfb"
    values = dict(sample_id="one", datetime_utc=stamp, latitude=51.75, longitude=-1.25,
                  weather_cache_key=key, weather_datetime_utc=stamp, weather_grid_latitude=51.75,
                  weather_grid_longitude=-1.25, background_air_temperature_c=10.,
                  relative_humidity_pct=60., dewpoint_c=3., wind_speed_m_s=2.,
                  wind_direction_sin=1., wind_direction_cos=np.cos(np.deg2rad(90.)),
                  surface_pressure_hpa=1000., cloud_cover_fraction=.25, shortwave_down_w_m2=200.,
                  direct_shortwave_w_m2=140., diffuse_shortwave_w_m2=60., soil_moisture_m3_m3=.2,
                  rain_mm_h=0., air_temperature_lag1_c=10., air_temperature_lag3_c=10.,
                  air_temperature_lag24_c=10., shortwave_down_lag1_w_m2=200., shortwave_down_mean3_w_m2=200.,
                  air_temperature_c=20. if station else 10., station_id="TEST0000001" if station else "",
                  station_age_minutes=10. if station else np.nan, station_distance_km=5. if station else np.nan,
                  station_air_correction_c=10. if station else np.nan,
                  observed_station_air_temperature_c=20. if station else np.nan,
                  air_temperature_source="observed_station_residual_plus_ERA5_spatial_background" if station else "ERA5_only_no_timely_station")
    for name, value in values.items():
        frame[name] = value
    if station:
        root = tmp_path / "stations"
        root.mkdir()
        pd.DataFrame({"GHCN_ID": ["TEST0000001"], "LATITUDE": [51.75], "LONGITUDE": [-1.25]}).to_csv(root / "ghcnh-station-list.csv", index=False)
        raw = root / "2023" / "GHCNh_TEST0000001_2023.psv"
        raw.parent.mkdir()
        raw.write_text("STATION|DATE|temperature\nTEST0000001|2023-06-21T11:50:00|20\n")
        identity = {"schema_version": 1, "raw_sha256": receipts._sha(raw),
                    "parser_source_sha256": receipts._sha(stations.__file__), "keep_flags": True}
        parser_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        parsed = raw.parent / f"TEST0000001_parsed_{parser_key}.parquet"
        pd.DataFrame({"station_id": ["TEST0000001"], "timestamp_utc": [stamp-pd.Timedelta(minutes=10)],
                      "air_temperature_c": [20.]}).to_parquet(parsed, index=False)
        parsed.with_suffix(".provenance.json").write_text(json.dumps(identity))
    return frame, path


def test_receipt_verifies_exact_report_background_and_preserves_all_predictors(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Receipt tried to acquire data")
    monkeypatch.setattr(weather, "fetch_archive", forbidden)
    monkeypatch.setattr(stations, "download_station_year", forbidden)
    monkeypatch.setattr(stations, "station_inventory", forbidden)
    before, _ = source_fixture(tmp_path)
    after, audit = receipts.bind_cached_sources(before, tmp_path)
    pd.testing.assert_frame_equal(before, after[before.columns], check_exact=True)
    assert after.verified_station_report_status.iloc[0] == receipts.STATION_OK
    assert after.verified_station_observation_datetime_utc.iloc[0] == pd.Timestamp("2023-06-21T11:50Z")
    assert audit["network_requests"] == 0
    assert len(audit["files"]) == 5  # One weather file shared by pixel and station.
    for binding in audit["files"]:
        assert receipts._sha(binding["path"]) == binding["sha256"]


@pytest.mark.parametrize("change", ["station_value", "correction", "model_air", "weather_lag", "weather_request", "weather_units", "other_station"])
def test_receipt_rejects_corrupted_alignment_or_delivery(tmp_path, change):
    frame, path = source_fixture(tmp_path)
    if change == "station_value":
        frame.loc[0, "observed_station_air_temperature_c"] = 21.
    elif change == "correction":
        frame.loc[0, "station_air_correction_c"] = 9.
    elif change == "model_air":
        frame.loc[0, "air_temperature_c"] = 19.
    elif change == "weather_lag":
        frame.loc[0, "air_temperature_lag24_c"] = 9.
    elif change in ("weather_request", "weather_units"):
        payload = json.loads(path.read_text())
        if change == "weather_request":
            payload["request"]["models"] = "era5_land"
        else:
            payload["response"]["hourly_units"]["temperature_2m"] = "K"
        path.write_text(json.dumps(payload))
    else:
        parsed = next((tmp_path / "stations" / "2023").glob("*.parquet"))
        data = pd.read_parquet(parsed)
        data["station_id"] = "TEST0000002"
        data.to_parquet(parsed, index=False)
    with pytest.raises(ValueError):
        receipts.bind_cached_sources(frame, tmp_path)


def test_missing_station_does_not_become_verified_and_parser_gap_does_not_fetch(tmp_path):
    frame, _ = source_fixture(tmp_path, station=False)
    result, receipt = receipts.bind_cached_sources(frame, tmp_path)
    assert result.verified_station_report_status.iloc[0] == "no_station_pair"
    assert receipt["network_requests"] == 0


def test_missing_matching_parser_remains_explicit(tmp_path):
    frame, _ = source_fixture(tmp_path)
    next((tmp_path / "stations" / "2023").glob("*.parquet")).unlink()
    result, _ = receipts.bind_cached_sources(frame, tmp_path)
    assert result.verified_station_report_status.iloc[0] == "matching_parser_cache_missing"


@pytest.mark.parametrize("field,value", [("region_id", "different"), ("epsg", 32631)])
def test_patch_restoration_rejects_region_and_projection_mutations(field, value):
    _, before = patch.patch_area(-1.25, 51.75, 2)
    before["datetime_utc"] = pd.Timestamp("2023-06-21T12:00Z")
    changed = before.copy()
    changed.loc[0, field] = value
    with pytest.raises((ValueError, AssertionError)):
        patch.restore_rows(before, changed)


def test_supported_patch_rejects_unverified_station_and_twilight(tmp_path):
    frame, _ = source_fixture(tmp_path)
    frame = pd.concat([frame] * 4, ignore_index=True)
    frame["zone_owned"] = True
    frame["worldcover_classified_fraction"] = 1.
    frame["worldcover_land_fraction"] = 1.
    frame["solar_elevation_deg"] = [10., -6., 0., 20.]
    frame["verified_station_report_status"] = [receipts.STATION_OK] * 3 + ["no_station_pair"]
    np.testing.assert_array_equal(patch.support_reasons(frame), [0, 0, 7, 8])


@pytest.mark.parametrize('operational',[False,True])
def test_patch_orchestration_preserves_source_rows_and_masks_before_model(tmp_path, monkeypatch, operational):
    from lst_global import weather_access,weather_stations
    cache = tmp_path / "cache"
    template, _ = source_fixture(cache)
    calls = []

    def check(frame, stage):
        assert not {"lst_c", "label_product", "acquisition_id", "target_offset"}.intersection(frame)
        calls.append(stage)

    def surfaces(frame, area, cache, **kwargs):
        check(frame, "surface")
        assert kwargs["max_scenes"] == 16
        data = frame.copy()
        for name, value in template.iloc[0].items():
            if name not in data:
                data[name] = value
        data["solar_elevation_deg"] = 20.
        data["worldcover_classified_fraction"] = 1.
        data["worldcover_land_fraction"] = 1.
        data.loc[data.grid_col.eq(1), "water_fraction"] = .1
        data["optical_observation_count"] = 2
        data["optical_latest_source_utc"] = pd.Timestamp("2023-06-18T11:00Z")
        return data.iloc[::-1].copy(), {"synthetic": True}

    def context(frame):
        check(frame, "context")
        return frame.assign(climate_source="synthetic classified source").iloc[::-1].copy()

    def atmos(frame, cache, **kwargs):
        check(frame, "weather")
        assert kwargs["max_requests"] == (64 if operational else 16)
        return frame.iloc[::-1].copy(), [{"synthetic": True}]

    def station(frame, areas, cache, **kwargs):
        check(frame, "station")
        if operational:
            frame=frame.assign(air_temperature_source='observed_METAR_residual_plus_GFS_spatial_background',
                               verified_station_report_status=weather_stations.STATION_OK)
        else:
            assert kwargs == dict(stations_per_region=2, max_distance_km=100, max_age_minutes=90)
        return frame.iloc[::-1].copy(), [{"synthetic": True}]

    def radiation(frame, cache, **kwargs):
        check(frame, "radiation")
        assert kwargs["max_hours"] == 1
        result = frame.iloc[::-1].copy()
        result.attrs["radiation_context"] = {"synthetic": True}
        return result

    class Model:
        def predict(self, frame, *, source_provenance):
            calls.append("predict")
            assert tuple(frame.columns) == FEATURES
            assert len(frame) == 2 and frame.water_fraction.eq(0).all()
            assert source_provenance["feature_table_sha256"]
            return SimpleNamespace(values=pd.DataFrame({"predicted_lst_c": frame.air_temperature_c + 2.5}, index=frame.index),
                                   provenance={"synthetic": True})

    monkeypatch.setattr(patch.FrozenF, "load", lambda: Model())
    monkeypatch.setattr(patch.surface, "append_surfaces", surfaces)
    monkeypatch.setattr(patch.raster, "add_raster_context", context)
    monkeypatch.setattr(patch.weather, "enrich_weather", atmos)
    monkeypatch.setattr(patch.assemble, "attach_stations", station)
    monkeypatch.setattr(patch.radiation, "add_radiation", radiation)
    plan=None
    if operational:
        plan={'source':'experimental_gfs','enabled':True,'valid_time_utc':'2023-06-21T12:00Z',
              'is_reanalysis':False,'source_limitations':['Synthetic source change fixture']}
        monkeypatch.setattr(weather_access,'prepare_background',lambda frame,cache,plan,**kw:atmos(frame,cache,**kw))
        monkeypatch.setattr(weather_stations,'attach_recent_stations',lambda frame,areas,cache,plan:station(frame,areas,cache))
        monkeypatch.setattr(weather_access,'add_radiation',lambda frame,cache,plan:(radiation(frame,cache,max_hours=1),{'synthetic':True}))
        def no_era5_receipt(*args,**kwargs):
            raise AssertionError('GFS must never be validated as ERA5')
        monkeypatch.setattr(receipts,'bind_cached_sources',no_era5_receipt)
    output = tmp_path / "result"
    result = patch._run_bounded(output, -1.25, 51.75, "2023-06-21T12:00Z", cells=2, cache=cache,weather_plan=plan)
    assert calls == ["surface", "context", "weather", "station", "radiation", "predict"]
    assert result["predicted_pixels"] == 2 and result["thermal_labels_read"] is False
    saved = pd.read_parquet(output / "inputs.parquet")
    assert saved.grid_row.to_list() == [0, 0, 1, 1]
    assert saved.optical_observation_count.eq(2).all()
    assert saved.verified_station_report_status.eq(weather_stations.STATION_OK if operational else receipts.STATION_OK).all()
    if operational:
        assert result['actual_weather_only'] is False and result['weather_is_reanalysis'] is False
    assert not {"lst_c", "target_offset", "label_product"}.intersection(saved)
    with patch.rasterio.open(output / "lst.tif") as source:
        np.testing.assert_array_equal(source.read(1), [[22.5, -9999.], [22.5, -9999.]])
    with patch.rasterio.open(output / "support.tif") as source:
        np.testing.assert_array_equal(source.read(1), [[0, 4], [0, 4]])
    for name, binding in result["artifacts"].items():
        assert receipts._sha(output / name) == binding["sha256"]
