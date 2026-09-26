"""Synthetic integrity tests only; no source acquisition or real model fitting."""
import numpy as np
import pandas as pd
import pytest

from lst_pilot import multisensor_train as multi
from lst_pilot import option_b_train as old
from lst_pilot import physics_correction as weak
from lst_pilot import physics_features as physical


def cohort():
    times = pd.DatetimeIndex([pd.Timestamp(f"2021-{month:02d}-{day:02d}T12:30Z")
                              for month in range(1, 7) for day in (5, 15)])
    n = len(times)
    solar = np.tile([25., -20.], 6)
    frame = pd.DataFrame({
        "sample_id": [f"row-{i}" for i in range(n)],
        "region_id": np.where(solar > 0, "greater_london", "sioux_falls"),
        "datetime_utc": times, "utc_day": times.floor("D"), "split": "fit",
        "spatial_holdout": False, "in_holdout_buffer": False,
        "phase": np.where(solar > 0, "day", "night"),
        "climate_class": np.where(solar > 0, "Cfb", "Dfa"),
        "solar_elevation_deg": solar, "air_temperature_c": np.linspace(-5., 25., n),
        "cloud_cover_fraction": np.linspace(.05, .9, n), "wind_speed_m_s": np.linspace(1., 6., n),
        "relative_humidity_pct": np.linspace(40., 90., n),
        "era5_snow_water_equivalent_m": np.linspace(0., .01, n),
        "albedo_proxy": np.linspace(.1, .3, n), "shortwave_down_w_m2": np.maximum(solar, 0) * 12,
        "era5_longwave_down_w_m2": np.linspace(200., 340., n),
        "acquisition_id": [f"a-{i}" for i in range(n)], "weight_surface_group": "land",
        "physics_complete": True,
    })
    frame[physical.RAW_HISTORY_FEATURE] = np.linspace(50., 250., n)
    frame[weak.PHYSICS[0]] = (1 - frame.albedo_proxy) * frame.shortwave_down_w_m2
    frame[weak.PHYSICS[1]] = frame.era5_longwave_down_w_m2 - physical.SIGMA_W_M2_K4 * (frame.air_temperature_c + 273.15)**4
    frame[weak.PHYSICS[2]] = (1 - frame.albedo_proxy) * frame[physical.RAW_HISTORY_FEATURE]
    offset = np.linspace(1., 3., n)
    frame["lst_c"] = frame.air_temperature_c + offset + np.linspace(-3., 5., n)
    return frame, offset


def fit(kind="WP"):
    frame, offset = cohort()
    folds = multi.month_folds(frame)
    weights = old.balanced_weights(frame)
    support = multi.group_support(frame, folds)
    result = weak.WeakCorrection.fit(frame, offset, folds, weights, kind, support)
    return result, frame, offset, folds, weights, support


@pytest.mark.parametrize("kind,count", [("W", 7), ("P", 3), ("WP", 10), ("WR", 11)])
def test_exact_feature_allowlists_exclude_temperature_labels(kind, count):
    frame, offset = cohort()
    values, climates, phase, names = weak.numerical_inputs(frame, offset, kind)
    assert values.shape == (len(frame), count)
    assert len(names) == len(set(names)) == count
    assert set(climates) == {"C", "D"}
    assert set(phase) == {"day", "night"}
    changed = frame.copy()
    changed["lst_c"] += 10000
    changed["label_product"] = "different_sensor"
    changed["acquisition_id"] = "different_scene"
    changed["phase"] = "wrong_metadata_phase"
    repeated = weak.numerical_inputs(changed, offset, kind)
    np.testing.assert_array_equal(values, repeated[0])
    np.testing.assert_array_equal(phase, repeated[2])
    if kind == "P":
        assert names == list(weak.PHYSICS)
        np.testing.assert_array_equal(values, weak.numerical_inputs(frame, offset + 100, kind)[0])
    else:
        assert names[:7] == list(multi.NUMERIC)
        np.testing.assert_array_equal(values[:, 0], offset)


def test_raw_control_does_not_derive_sunlight_by_dividing_absorbed_by_one_minus_albedo():
    frame, offset = cohort()
    frame["albedo_proxy"] = 1.
    frame[weak.PHYSICS[0]] = 0.
    frame[weak.PHYSICS[2]] = 0.
    values, _, _, names = weak.numerical_inputs(frame, offset, "WR")
    np.testing.assert_array_equal(values[:, names.index(physical.RAW_HISTORY_FEATURE)], frame[physical.RAW_HISTORY_FEATURE])
    assert np.isfinite(values).all()
    assert weak.valid_physics(frame).all()


def test_bad_arm_or_offset_alignment_raises():
    frame, offset = cohort()
    with pytest.raises(ValueError, match="Unknown"):
        weak.numerical_inputs(frame, offset, "invented")
    with pytest.raises(ValueError, match="misaligned"):
        weak.numerical_inputs(frame, offset[:-1], "P")


@pytest.mark.parametrize("field", [*weak.PHYSICS, physical.RAW_HISTORY_FEATURE])
def test_validity_requires_every_physics_and_raw_control_value(field):
    frame, _ = cohort()
    frame.loc[0, field] = np.nan
    frame.loc[1, field] = np.inf
    valid = weak.valid_physics(frame)
    np.testing.assert_array_equal(valid, [False, False] + [True] * 10)


def test_explicit_failed_physics_proof_cannot_be_recovered_from_finite_values():
    frame, _ = cohort()
    frame.loc[0, "physics_complete"] = False
    assert not weak.valid_physics(frame)[0]


@pytest.mark.parametrize("value", [None, "false", np.nan])
def test_missing_or_nonboolean_physics_proof_is_not_truthy(value):
    frame, _ = cohort()
    frame["physics_complete"] = frame.physics_complete.astype(object)
    frame.loc[0, "physics_complete"] = value
    try:
        valid = weak.valid_physics(frame)
    except ValueError:
        return
    assert not valid[0]


def test_weather_head_matches_original_generic_coefficients_before_weaker_strength():
    correction, frame, offset, folds, _, _ = fit("W")
    generic = multi.ResidualCorrection.fit(frame, offset, folds)
    np.testing.assert_allclose(correction.mean, generic.mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(correction.scale, generic.scale, rtol=0, atol=1e-12)
    np.testing.assert_allclose(correction.ridge.coef_, generic.ridge.coef_, rtol=0, atol=1e-12)
    assert correction.effective_dates == pytest.approx(generic.effective_utc_dates)


@pytest.mark.parametrize("kind", list(weak.ARMS))
def test_fitting_permutation_and_absolute_weight_scale_do_not_change_solution(kind):
    correction, frame, offset, folds, weights, support = fit(kind)
    order = np.arange(len(frame))[::-1]
    repeated = weak.WeakCorrection.fit(frame.iloc[order], offset[order], folds[order],
                                      weights[order] * 13, kind, support)
    np.testing.assert_allclose(correction.ridge.coef_, repeated.ridge.coef_, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(correction.predict(frame, offset)[0], repeated.predict(frame, offset)[0], rtol=0, atol=1e-11)


def test_replicated_pixels_do_not_weaken_regularization_or_add_independent_dates():
    correction, frame, offset, folds, weights, support = fit("WP")
    repeated_frame = pd.concat([frame, frame], ignore_index=True)
    repeated = weak.WeakCorrection.fit(repeated_frame, np.tile(offset, 2), np.tile(folds, 2),
                                      np.tile(weights / 2, 2), "WP", support)
    assert repeated.effective_dates == pytest.approx(correction.effective_dates)
    np.testing.assert_allclose(repeated.ridge.coef_, correction.ridge.coef_, rtol=0, atol=1e-11)


@pytest.mark.parametrize("kind", list(weak.ARMS))
def test_missing_proxy_and_unsupported_climate_have_identical_exact_fallback(kind):
    correction, frame, offset, *_ = fit(kind)
    frame.loc[0, physical.RAW_HISTORY_FEATURE] = np.nan
    frame.loc[1, "climate_class"] = "BWh"
    frame.loc[2, "solar_elevation_deg"] = 0.  # twilight lacks declared support
    original = frame.copy(deep=True)
    adjustment, supported = correction.predict(frame, offset)
    np.testing.assert_array_equal(supported[:3], [False, False, False])
    np.testing.assert_array_equal(adjustment[:3], np.zeros(3))
    base = np.linspace(-17.123456789, 44.987654321, len(frame))
    np.testing.assert_array_equal((base + adjustment)[~supported], base[~supported])
    pd.testing.assert_frame_equal(frame, original)


def test_prediction_does_not_consume_labels_and_all_arms_use_same_support():
    support_arrays = []
    for kind in weak.ARMS:
        correction, frame, offset, *_ = fit(kind)
        before = correction.predict(frame, offset)
        unlabeled = frame.drop(columns=["lst_c"])
        after = correction.predict(unlabeled, offset)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        support_arrays.append(after[1])
    for values in support_arrays:
        np.testing.assert_array_equal(values, support_arrays[0])


class FixedResidual:
    def __init__(self, values):
        self.values = values

    def predict(self, matrix):
        return np.resize(np.asarray(self.values, dtype=float), len(matrix))


def test_adjustments_have_frozen_quarter_strength_and_one_degree_cap():
    correction, frame, offset, *_ = fit("WP")
    correction.ridge = FixedResidual([2., -2., 8., -8., 0.])
    adjustment, support = correction.predict(frame, offset)
    assert support.all()
    np.testing.assert_array_equal(adjustment[:5], [.5, -.5, 1., -1., 0.])
    assert weak.SPEC["shrinkage"] == .25
    assert weak.SPEC["max_abs_adjustment_c"] == 1.
    assert np.max(np.abs(adjustment)) <= 1.


def test_no_supported_rows_never_call_regressor():
    correction, frame, offset, *_ = fit("W")
    correction.ridge = None
    correction.support = {}
    adjustment, support = correction.predict(frame, offset)
    np.testing.assert_array_equal(adjustment, np.zeros(len(frame)))
    assert not support.any()


def test_misaligned_folds_and_reserved_or_wrong_year_fit_are_rejected():
    _, frame, offset, folds, weights, support = fit("W")
    with pytest.raises(ValueError, match="alignment"):
        weak.WeakCorrection.fit(frame, offset, (folds + 1) % 3, weights, "W", support)
    for field, value in (("spatial_holdout", True), ("in_holdout_buffer", True),
                         ("region_id", "cabauw"), ("split", "development"),
                         ("datetime_utc", pd.Timestamp("2023-01-01T00:00Z"))):
        changed = frame.copy()
        changed.loc[0, field] = value
        with pytest.raises(ValueError):
            weak.WeakCorrection.fit(changed, offset, folds, weights, "P", support)


@pytest.mark.parametrize("kind", list(weak.ARMS))
def test_all_arms_reject_incomplete_common_fitting_cohort(kind):
    _, frame, offset, folds, weights, support = fit(kind)
    frame.loc[0, physical.RAW_HISTORY_FEATURE] = np.nan
    with pytest.raises(ValueError, match="same physical-complete"):
        weak.WeakCorrection.fit(frame, offset, folds, weights, kind, support)


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_invalid_fitting_weights_cannot_change_effective_date_mass(value):
    _, frame, offset, folds, weights, support = fit("P")
    weights[0] = value
    with pytest.raises(ValueError, match="weights"):
        weak.WeakCorrection.fit(frame, offset, folds, weights, "P", support)


@pytest.mark.parametrize("changed", ["protocol", "source", "freeze", "model", "fit_cache", "eval_cache"])
def test_freeze_rejects_changed_code_protocol_artifacts_or_exact_cache_sources(tmp_path, monkeypatch, changed):
    source = tmp_path / "physics_correction.py"
    source.write_text("frozen implementation")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("fixed experimental protocol")
    output = tmp_path / "run"
    output.mkdir()
    model = output / "W.joblib"
    model.write_bytes(b"synthetic frozen correction")
    freeze = output / "fit_freeze.json"
    freeze.write_text("synthetic freeze")
    fit_cache = tmp_path / "fit-weather.json"
    fit_cache.write_text("existing fitting cache")
    eval_cache = tmp_path / "evaluation-weather.json"
    eval_cache.write_text("existing evaluation cache")
    monkeypatch.setattr(weak, "__file__", str(source))
    records = {
        "protocol_sha256": old.sha(protocol),
        "source_hashes": {source.name: old.sha(source)},
        "feature_audit": {"cache_records": [{"path": str(fit_cache), "sha256": old.sha(fit_cache)}]},
    }
    frozen = {model.name: old.sha(model)}
    freeze_sha = old.sha(freeze)
    extra = ({"cache_records": [{"path": str(eval_cache), "sha256": old.sha(eval_cache)}]},)
    weak.verify_experiment_freeze(records, frozen, protocol, output, freeze_sha, extra)
    paths = {"protocol": protocol, "source": source, "freeze": freeze, "model": model,
             "fit_cache": fit_cache, "eval_cache": eval_cache}
    paths[changed].write_text("modified after fitting")
    with pytest.raises(ValueError, match="changed"):
        weak.verify_experiment_freeze(records, frozen, protocol, output, freeze_sha, extra)
