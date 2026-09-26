import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from lst_pilot import more_days_pair as pair


def sample():
    area = {"epsg": 32630, "extent_m": [500000, 5700000, 501000, 5701000], "grid_shape": [10, 10]}
    lon, lat = Transformer.from_crs(32630, 4326, always_xy=True).transform(500550, 5700450)
    frame = pd.DataFrame({"sample_id": ["new"], "acquisition_id": ["new_acq"], "region_id": ["greater_london"],
                          "datetime_utc": [pd.Timestamp("2021-06-01T10:00Z")], "latitude": [lat], "longitude": [lon],
                          "grid_row": [5], "grid_col": [5], "epsg": [32630], "lst_c": [22.],
                          "cohort_origin": ["expanded"], "label_product": ["landsat_c2_l2"]})
    old = pd.DataFrame({"sample_id": ["old"], "acquisition_id": ["old_acq"], "region_id": ["greater_london"],
                        "datetime_utc": [pd.Timestamp("2022-06-01T10:00Z")]})
    return frame, {"greater_london": area}, old


def test_identity_rejects_old_night_date_and_reserved_year_before_pairing():
    frame, areas, old = sample()
    pair.verify_new_identity(frame, areas, old)
    old["datetime_utc"] = pd.Timestamp("2021-06-01T23:00Z")
    with pytest.raises(ValueError, match="pilot/date"):
        pair.verify_new_identity(frame, areas, old)
    frame["datetime_utc"] = pd.Timestamp("2023-06-01T10:00Z")
    with pytest.raises(ValueError, match="2021"):
        pair.verify_new_identity(frame, areas, old)


def test_identity_rejects_shifted_coordinate_within_old_two_metre_tolerance():
    frame, areas, old = sample()
    lon, lat = Transformer.from_crs(32630, 4326, always_xy=True).transform(500551.5, 5700450)
    frame["longitude"], frame["latitude"] = lon, lat
    with pytest.raises(ValueError, match="1 m"):
        pair.verify_new_identity(frame, areas, old)


def test_fit_admission_keeps_only_audited_complete_safe_new_rows():
    frame, _, _ = sample()
    data = pd.concat([frame]*5, ignore_index=True)
    data["sample_id"] = ["safe", "held", "unverified", "missing", "source_reject"]
    for name in pair.f.BASE_FEATURES:
        data[name] = "Cfb" if name == "climate_class" else 20.
    data["research_admissibility_reason"] = ["", "", "", "", "no_actual_station_pair"]
    data["spatial_holdout"] = [False, True, False, False, False]
    data["in_holdout_buffer"] = False
    data["verified_station_report_status"] = ["exact_cached_report_verified"]*5
    data.loc[2, "verified_station_report_status"] = "no_station_pair"
    data.loc[3, "ndvi"] = np.nan
    selected, rejected, audit = pair.fit_only_rows(data.iloc[:4], data.iloc[4:], list(data.sample_id))
    assert selected.sample_id.tolist() == ["safe"]
    assert selected.research_admissibility_reason.tolist() == [""]
    assert set(rejected.new_fit_exclusion_reason) == {"spatial_holdout", "unverified_actual_station_report", "incomplete_base40", "no_actual_station_pair"}
    assert audit["fit_rows"] == 1


def test_legacy_isd_audit_requires_exact_raw_value_hash_and_qc():
    frame, _, _ = sample()
    frame["station_id"] = pair.legacy_isd.STATION_ID
    frame["station_age_minutes"] = 30.
    frame["station_distance_km"] = 10.
    frame["observed_station_air_temperature_c"] = 25.
    frame["station_raw_sha256"] = "rawhash"
    frame["station_raw_url"] = "https://example.test/raw"
    frame["air_temperature_source"] = pair.legacy_isd.AIR_SOURCE
    frame["verified_station_report_status"] = "raw_cache_missing_or_non_GHCNh_source"
    frame["verified_station_observation_datetime_utc"] = pd.Series(pd.NaT, dtype="datetime64[ns, UTC]")
    observations = pd.DataFrame({"isd_observation_datetime_utc": [pd.Timestamp("2021-06-01T09:30Z")],
                                  "isd_temperature_c": [25.], "isd_quality_code": ["1"], "isd_raw_sha256": ["rawhash"],
                                  "isd_raw_url": ["https://example.test/raw"], "isd_source_code": ["4"], "isd_report_type": ["FM-15"]})
    verified = pair.verify_isd_rows(frame, observations)
    assert verified.verified_station_report_status.iloc[0] == "exact_cached_legacy_isd_qc1_report_verified"
    pd.testing.assert_frame_equal(frame[[c for c in frame if not c.startswith("verified_")]], verified[[c for c in frame if not c.startswith("verified_")]], check_exact=True)
    observations["isd_temperature_c"] = 25.1
    assert pair.verify_isd_rows(frame, observations).verified_station_report_status.iloc[0] not in pair.VERIFIED
