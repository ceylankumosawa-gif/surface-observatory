"""Small contract tests; run on the pilot server, with no synthetic model fitting."""

import numpy as np
import pandas as pd
import pytest

from lst_pilot.model import (
    ModelConfig, assign_splits, build_estimators, prepare_frame,
    residual_interval_radius, select_features, target_offset,
    temperature_from_offset, validate_splits,
)


def config(**changes):
    values = dict(heldout_regions=("london",), train_end="2023-01-02", calibration_end="2023-01-04", min_train_rows=2, min_calibration_rows=2, min_evaluation_rows=2)
    values.update(changes)
    return ModelConfig(**values)


def observations():
    records = []
    for region in ("london", "cabauw"):
        for date in pd.date_range("2023-01-01", periods=6, tz="UTC"):
            for hour in (0, 12):
                records.append(dict(region_id=region, datetime_utc=date + pd.Timedelta(hours=hour), latitude=51.5, longitude=0.0, lst_c=17.0, air_temperature_c=12.0, climate_class="Cfb", ndvi=0.4))
    return pd.DataFrame(records)


def test_whole_regions_and_days_are_disjoint():
    data, _ = prepare_frame(observations(), config())
    split = assign_splits(data, config())
    validate_splits(split, config())
    assert set(split.loc[split.region_id.eq("london"), "split"]) == {"test_region"}
    assert not split.loc[split.split.isin(["train", "calibration"]), "region_id"].eq("london").any()
    known = split.loc[split.region_id.eq("cabauw")]
    assert known.groupby("utc_day").split.nunique().eq(1).all()
    assert known.loc[known.split.eq("train"), "utc_day"].max() < known.loc[known.split.eq("calibration"), "utc_day"].min()
    assert known.loc[known.split.eq("calibration"), "utc_day"].max() < known.loc[known.split.eq("test_temporal"), "utc_day"].min()
    assert known.loc[known.datetime_utc.eq(pd.Timestamp("2023-01-02T12:00Z")), "split"].item() == "train"


def test_utc_dates_drive_boundaries_not_local_calendar_dates():
    data = pd.DataFrame({"region_id": ["cabauw", "cabauw", "london"], "datetime_utc": ["2023-01-03T00:30:00+02:00", "2023-01-03T00:30:00Z", "2023-01-01T00:00:00Z"]})
    split = assign_splits(data, config())
    assert split.split.tolist() == ["train", "calibration", "test_region"]


def test_offset_target_and_reconstruction_are_exact():
    data = pd.DataFrame({"lst_c": [35.0, -10.0, 15.0], "air_temperature_c": [25.0, -4.0, 15.0]})
    offsets = target_offset(data)
    np.testing.assert_array_equal(offsets, [10.0, -6.0, 0.0])
    np.testing.assert_array_equal(temperature_from_offset(data.air_temperature_c, offsets), data.lst_c)


@pytest.mark.parametrize("forbidden", ["lst_c", "latitude", "longitude", "region_id", "station_id", "brightness_temperature_c", "emissivity", "target_emissivity", "thermal_band"])
def test_feature_allowlist_rejects_label_and_identity_leakage(forbidden):
    columns = ["air_temperature_c", "climate_class", "ndvi", forbidden]
    with pytest.raises(ValueError, match="Unsafe or unsupported"):
        select_features(columns, ["ndvi", forbidden])
    assert forbidden not in select_features(columns)


def test_unknown_climate_needs_explicit_smoke_override():
    data = observations()
    data["climate_class"] = None
    with pytest.raises(ValueError, match="All climate classifications"):
        prepare_frame(data, config())
    prepared, _ = prepare_frame(data, config(allow_unknown_climate_smoke=True))
    assert prepared.climate_class.eq("__unknown__").all()


def test_missing_requested_region_fails_before_fitting():
    with pytest.raises(ValueError, match="no records"):
        assign_splits(observations(), config(heldout_regions=("not_present",)))


def test_empty_calibration_fails_before_fitting():
    data, _ = prepare_frame(observations(), config())
    split = assign_splits(data, config())
    split = split.loc[~split.split.eq("calibration")]
    with pytest.raises(ValueError, match="calibration.*0 records"):
        validate_splits(split, config())


def test_estimator_does_not_create_hidden_random_validation_split():
    estimator, _ = build_estimators(["air_temperature_c", "ndvi", "climate_class"], config())
    assert estimator.named_steps["regressor"].early_stopping is False
    assert estimator.named_steps["regressor"].max_iter == 150


def test_calibration_radius_uses_conservative_finite_sample_rank():
    assert residual_interval_radius(np.arange(1, 10), 0.8) == 8.0
    with pytest.raises(ValueError, match="Too few calibration"):
        residual_interval_radius([1.0], 0.9)
