import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from lst_pilot import multisensor_pair as pair


def sample(region="boulder", year=2021):
    area = {"epsg": 32630, "extent_m": [500000, 5700000, 520000, 5720000], "grid_shape": [200, 200]}
    lon, lat = Transformer.from_crs(32630, 4326, always_xy=True).transform(500550, 5719450)
    stamp = pd.Timestamp(f"{year}-06-01T10:00Z")
    data = pd.DataFrame({"sample_id": ["new"], "acquisition_id": ["new_acq"], "region_id": [region],
                         "datetime_utc": [stamp], "latitude": [lat], "longitude": [lon],
                         "grid_row": [5], "grid_col": [5], "epsg": [32630], "lst_c": [24.],
                         "cohort_origin": ["expanded"], "label_product": ["ecostress_v2"],
                         "source_screen_pass": [True], "native_fit_support_pass": [True],
                         "label_source_sha256": ["a"*64], "day_night": ["day"],
                         "worldcover_land_fraction": [1.], "worldcover_water_fraction": [0.],
                         "station_pair_available": [True], "station_age_minutes": [30.],
                         "station_distance_km": [10.],
                         "verified_station_report_status": ["exact_cached_report_verified"],
                         "optical_earliest_source_utc": [stamp-pd.Timedelta(days=5)],
                         "optical_latest_source_utc": [stamp-pd.Timedelta(days=2)]})
    for name in pair.features.BASE_FEATURES:
        data[name] = "Dfb" if name == "climate_class" else 20.
    old = data.copy()
    old["sample_id"] = "old"
    old["acquisition_id"] = "old_acq"
    return data, {region: area}, old


def test_reserved_year_is_rejected_before_thermal_column_read(monkeypatch):
    calls = []
    def read(path, columns=None):
        calls.append(columns)
        assert columns is not None and "lst_c" not in columns
        return pd.DataFrame({"region_id": ["boulder"], "datetime_utc": ["2024-01-01T12:00Z"],
                             "label_product": ["ecostress_v2"]})
    monkeypatch.setattr(pair.pd, "read_parquet", read)
    with pytest.raises(ValueError, match="2021"):
        pair.inspect_time_boundary("unused")
    assert len(calls) == 1


def test_coarse_product_cannot_become_a_fine_label():
    data, areas, old = sample()
    data["label_product"] = "VNP21"
    with pytest.raises(ValueError, match="fine sources"):
        pair.validate_identity(data, areas, old, {"dates": []}, "b"*64)


def test_same_date_new_sensor_is_allowed_but_physical_duplicate_is_not():
    data, areas, old = sample()
    pair.validate_identity(data, areas, old, {"dates": []}, "b"*64)
    data["acquisition_id"] = "old_acq"
    with pytest.raises(ValueError, match="physical observation"):
        pair.validate_identity(data, areas, old, {"dates": []}, "b"*64)


def test_freshness_is_derived_from_prior_pilot_dates_across_sensors():
    data, areas, old = sample(year=2023)
    fresh = pair.validate_identity(data, areas, old, {"dates": []}, "b"*64)
    assert fresh.fresh_2023.iloc[0]
    repeated = pair.validate_identity(data, areas, old,
        {"dates": [{"region_id": "boulder", "utc_date": "2023-06-01"}]}, "c"*64)
    assert not repeated.fresh_2023.iloc[0]
    assert repeated.freshness_audit_sha256.iloc[0] == "c"*64


def test_ecostress_day_and_night_follow_actual_solar_geometry():
    data, areas, _ = sample()
    fit, _, _, _ = pair.source_admission(data, areas)
    assert len(fit) == 1 and fit.phase.iloc[0] == "day"
    data["solar_elevation_deg"] = -20.
    fit, _, excluded, _ = pair.source_admission(data, areas)
    assert fit.empty and "declared_phase" in excluded.research_admissibility_reason.iloc[0]
    data["day_night"] = "night"
    fit, _, _, _ = pair.source_admission(data, areas)
    assert len(fit) == 1 and fit.phase.iloc[0] == "night"
    data["solar_elevation_deg"] = -2.
    fit, _, excluded, _ = pair.source_admission(data, areas)
    assert fit.empty and excluded.research_admissibility_reason.iloc[0] == "unsupported_twilight"


@pytest.mark.parametrize("column,value,reason", [
    ("source_screen_pass", pd.NA, "fine_source_screen_incomplete"),
    ("station_age_minutes", -1., "station_time_mismatch"),
    ("station_age_minutes", 91., "station_time_mismatch"),
    ("station_distance_km", 101., "station_distance_mismatch"),
    ("verified_station_report_status", "not_verified", "unverified_actual_station_report"),
    ("worldcover_water_fraction", .1, "independent_water_fraction_exceeds_five_percent"),
    ("ndvi", np.nan, "incomplete_base40"),
])
def test_missing_or_invalid_proof_never_silently_passes(column, value, reason):
    data, areas, _ = sample()
    data[column] = value
    fit, _, excluded, _ = pair.source_admission(data, areas)
    assert fit.empty and excluded.research_admissibility_reason.iloc[0] == reason


def test_native_support_exclusion_prevents_fit_but_keeps_2023_evaluation():
    data, areas, _ = sample()
    data["native_fit_support_pass"] = False
    fit, _, excluded, _ = pair.source_admission(data, areas)
    assert fit.empty and excluded.new_fit_exclusion_reason.iloc[0].startswith("native_support")
    data["datetime_utc"] += pd.DateOffset(years=2)
    data["optical_earliest_source_utc"] += pd.DateOffset(years=2)
    data["optical_latest_source_utc"] += pd.DateOffset(years=2)
    fit, evaluation, excluded, _ = pair.source_admission(data, areas)
    assert fit.empty and len(evaluation) == 1 and excluded.empty


def test_cabauw_is_never_fitted_and_future_optics_are_rejected():
    data, areas, _ = sample(region="cabauw")
    fit, _, excluded, _ = pair.source_admission(data, areas)
    assert fit.empty and excluded.new_fit_exclusion_reason.iloc[0] == "withheld_region"
    data["optical_latest_source_utc"] = data.datetime_utc+pd.Timedelta(seconds=1)
    _, _, excluded, _ = pair.source_admission(data, areas)
    assert excluded.research_admissibility_reason.iloc[0] == "missing_future_or_stale_optical"


def test_buffer_is_excluded_from_new_evaluation_without_discarding_reserved_cells():
    data, areas, _ = sample(region="greater_london", year=2023)
    area = areas["greater_london"]
    # 10 km block r0c0 is reserved; cell centre x=510550 is in its 1 km buffer.
    for col, expected in [(105, "buffer"), (5, "reserved")]:
        data["grid_col"] = col
        data["grid_row"] = 150
        x = area["extent_m"][0]+(col+.5)*100
        y = area["extent_m"][3]-(150+.5)*100
        lon, lat = Transformer.from_crs(area["epsg"], 4326, always_xy=True).transform(x, y)
        data["longitude"], data["latitude"] = lon, lat
        _, evaluation, excluded, _ = pair.source_admission(data, areas)
        if expected == "buffer":
            assert evaluation.empty and excluded.new_evaluation_exclusion_reason.iloc[0] == "holdout_buffer"
        else:
            assert len(evaluation) == 1 and evaluation.spatial_holdout.iloc[0]


def station_fixture(tmp_path):
    data, _, _ = sample()
    root = tmp_path/"root"
    script = root/"reports/night_replacement/audit_option_b_stations.py"
    script.parent.mkdir(parents=True)
    script.write_text("# synthetic audit identity\n")
    output = tmp_path/"output"
    directory = output/"station_audit"
    directory.mkdir(parents=True)
    input_path = output/"features.parquet"
    destination = directory/"features_station_audited.parquet"
    data.to_parquet(input_path, index=False)
    data.to_parquet(destination, index=False)
    report = {"reused_darwin_adapter": False,
              "input_sha256": pair.features._sha(input_path),
              "output_sha256": pair.features._sha(destination),
              "audit_code_sha256": pair.features._sha(script),
              "orchestration_sha256": pair.features._sha(pair.more_days_pair.__file__)}
    report_path = directory/"station_source_audit.json"
    pair.features._write_json(report_path, report)
    return data, root, output, input_path, destination, report_path, report


def test_incomplete_station_checkpoint_is_rerun_not_trusted(tmp_path):
    data, root, output, source, destination, report_path, _ = station_fixture(tmp_path)
    assert pair.station_checkpoint(output, data, source, root) == destination
    report_path.unlink()
    assert pair.station_checkpoint(output, data, source, root) is None


@pytest.mark.parametrize("tamper", ["output", "input", "source_proof", "raw_predictor"])
def test_station_checkpoint_binds_real_input_output_and_unchanged_proof(tmp_path, tamper):
    data, root, output, source, destination, report_path, report = station_fixture(tmp_path)
    if tamper == "input":
        changed = pd.read_parquet(source)
        changed["ndvi"] = .1
        changed.to_parquet(source, index=False)
    else:
        changed = pd.read_parquet(destination)
        column = "label_source_sha256" if tamper == "source_proof" else "ndvi"
        changed[column] = "f"*64 if column == "label_source_sha256" else .1
        changed.to_parquet(destination, index=False)
        if tamper != "output":
            # Even a self-consistent altered report cannot excuse changed source
            # proofs or predictor values: those must match the audit's real input.
            report["output_sha256"] = pair.features._sha(destination)
            pair.features._write_json(report_path, report)
    with pytest.raises((ValueError, AssertionError)):
        pair.station_checkpoint(output, data, source, root)
