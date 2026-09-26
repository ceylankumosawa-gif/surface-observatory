"""Evaluation contract tests; no models, source downloads or live jobs."""
import json

import numpy as np
import pandas as pd
import pytest

from lst_pilot import option_b_train as old
from lst_pilot import physics_metrics as pm


def frame(days=6, pilots=("greater_london", "sioux_falls"), pixels=1):
    rows = []
    for region in pilots:
        for day in range(days):
            time = pd.Timestamp("2023-01-01T12:00:00Z") + pd.Timedelta(days=day)
            for pixel in range(pixels):
                rows.append({
                    "sample_id": f"{region}:{day}:{pixel}", "region_id": region,
                    "datetime_utc": time, "utc_day": time.floor("D"),
                    "phase": "day", "season": "DJF", "air_group": "cold",
                    "acquisition_id": f"{region}:{day}", "weight_surface_group": "built",
                    "label_product": "landsat", "lst_c": float(pixel),
                })
    return pd.DataFrame(rows)


def row(rows, model=None, kind="overall", segment="overall", comparison=None):
    return next(r for r in rows if r["segment_type"] == kind and r["segment"] == segment
                and (model is None or r.get("model") == model)
                and (comparison is None or r.get("comparison") == comparison))


def test_exact_old_metric_parity_and_no_input_mutation():
    data = frame(pixels=3)
    data.index = np.arange(len(data)) * 7 + 10  # array alignment is positional
    data.loc[data.index[::3], "weight_surface_group"] = "tree"
    original = data.copy(deep=True)
    y = data.lst_c.to_numpy()
    pred = {"F": y + np.resize([0, -6, 8], len(data)),
            "WP": y + np.resize([-1, -5, 7], len(data))}
    changes = {"WP": pred["WP"] - pred["F"]}
    metrics, pairs = pm.evaluate_cohort(data, pred, changes, {"WP": np.ones(len(data), bool)}, "fixed")
    actual = row(metrics, "WP")
    for name, value in old.metrics(data, pred["WP"]).items():
        assert actual[name] == value
    assert actual["unweighted_gt5_fraction"] == pytest.approx(1 / 3)
    assert actual["unweighted_gt7_fraction"] == 0
    assert actual["adjusted_fraction"] == pytest.approx(1)
    assert actual["mean_abs_adjustment_c"] == pytest.approx(1)
    paired = row(pairs, comparison="WP_vs_F")
    assert paired["delta_mae_c"] == actual["mae_c"] - row(metrics, "F")["mae_c"]
    assert paired["bootstrap_draws"] == 2000
    json.dumps({"metrics": metrics, "paired": pairs}, allow_nan=False)
    pd.testing.assert_frame_equal(data, original)


def test_global_date_clusters_jointly_resample_pilots():
    data = frame()
    base = np.full(len(data), 10.0)
    corrected = np.where(data.region_id.eq("greater_london"), 9.0, 11.0)
    _, pairs = pm.evaluate_cohort(data, {"F": base, "WP": corrected}, {}, {}, "fixed")
    result = row(pairs, comparison="WP_vs_F")
    assert result["date_count"] == 12  # pilot-dates
    assert result["utc_date_count"] == result["bootstrap_cluster_count"] == 6
    # Every global date contains both opposing errors. Resampling pilot-dates
    # or pixels independently would incorrectly create a nonzero interval.
    assert result["delta_mae_c"] == pytest.approx(0)
    assert result["delta_mae_c_ci95"] == pytest.approx([0, 0], abs=1e-14)
    assert result["bootstrap_cluster"] == "global_utc_date"


def test_date_numerators_and_denominators_resample_with_frozen_weights():
    data = frame(days=8)
    # The second region has just one date, so global dates have unequal mass.
    data = data.loc[data.region_id.eq("greater_london") | data.utc_day.eq(data.utc_day.min())].reset_index(drop=True)
    predicted = np.arange(1, len(data) + 1, dtype=float)
    _, pairs = pm.evaluate_cohort(data, {"F": np.zeros(len(data)), "WP": predicted}, {}, {}, "fixed")
    result = row(pairs, comparison="WP_vs_F")
    weights = old.balanced_weights(data)
    dates, code = np.unique(data.utc_day.to_numpy(), return_inverse=True)
    draws = np.random.default_rng(2708).multinomial(len(dates), np.full(len(dates), 1 / len(dates)), 2000)
    numerator = np.bincount(code, weights=weights * predicted)
    denominator = np.bincount(code, weights=weights)
    expected = np.quantile((draws @ numerator) / (draws @ denominator), [.025, .975])
    assert result["delta_mae_c_ci95"] == pytest.approx(expected)
    assert result["delta_mae_c"] == pytest.approx(np.sum(weights * predicted))
    # Duplicating all pixels must not inflate independent date support or alter
    # the weighted interval; repeated copies belong to the same clusters.
    duplicated = pd.concat([data, data.assign(sample_id=data.sample_id + ":copy")], ignore_index=True)
    _, repeated = pm.evaluate_cohort(duplicated, {"F": np.zeros(len(duplicated)), "WP": np.tile(predicted, 2)}, {}, {}, "fixed")
    repeat_result = row(repeated, comparison="WP_vs_F")
    assert repeat_result["bootstrap_cluster_count"] == 8
    assert repeat_result["delta_mae_c_ci95"] == pytest.approx(expected)


def test_sparse_global_dates_not_pilot_dates_and_fixed_segment_policy():
    data = frame(days=3)
    _, pairs = pm.evaluate_cohort(data, {"F": np.ones(len(data)), "WP": np.zeros(len(data))}, {}, {}, "fixed")
    result = row(pairs, comparison="WP_vs_F")
    assert result["date_count"] == 6 and result["utc_date_count"] == 3
    assert result["bootstrap_status"] == "sparse_global_utc_dates"
    assert result["delta_mae_c_ci95"] is None
    assert result["bootstrap_draws"] == 0
    assert "observed 3" in result["bootstrap_reason"]
    source = row(pairs, kind="source", segment="landsat", comparison="WP_vs_F")
    assert source["bootstrap_status"] == "not_requested_for_segment"
    assert source["delta_mae_c_ci95"] is None


def test_observed_surface_thresholds_air_independence_and_zero_negative_values():
    data = frame(days=1, pilots=("greater_london",), pixels=5)
    data["lst_c"] = [-10, 0, 0.1, 34.9, 35]
    data["air_group"] = "hot"
    values = data.lst_c.to_numpy()
    metrics, _ = pm.evaluate_cohort(data, {"F": values}, {}, {}, "fixed")
    assert row(metrics, "F")["mae_c"] == 0
    for surface, count in (("cold", 2), ("mild", 2), ("hot", 1)):
        assert row(metrics, "F", "region_phase_surface", f"greater_london|day|{surface}")["n"] == count
    assert row(metrics, "F", "region_phase_air", "greater_london|day|hot")["n"] == 5


def test_all_comparisons_and_adjustment_support_are_distinct():
    data = frame(days=1, pilots=("greater_london",), pixels=3)
    y = data.lst_c.to_numpy()
    predictions = {name: y + offset for name, offset in (("F", 5), ("W", 4), ("P", 3), ("WP", 2), ("WR", 1))}
    adjustments = {"WP": np.array([0, 0, -1.0])}
    supported = {"WP": np.array([False, True, True])}
    metrics, pairs = pm.evaluate_cohort(data, predictions, adjustments, supported, "fixed")
    result = row(metrics, "WP")
    assert result["adjusted_fraction"] == pytest.approx(1 / 3)
    assert result["supported_fraction"] == pytest.approx(2 / 3)
    assert result["mean_abs_adjustment_c"] == pytest.approx(1 / 3)
    assert {r["comparison"] for r in pairs} == {f"{a}_vs_{b}" for a, b in pm.COMPARISONS}
    for result in [r for r in pairs if r["segment_type"] == "overall"]:
        assert result["n"] == 3
        assert result["sample_id_sha256"] == old.row_hash(data)


@pytest.mark.parametrize("problem", ["prediction_nan", "prediction_shape", "target_nan", "missing_group", "wrong_date", "duplicate_id", "cross_date_acquisition", "string_support", "mismatched_keys"])
def test_invalid_rows_cannot_be_dropped_or_silently_realigned(problem):
    data = frame()
    prediction = np.zeros(len(data))
    changes, support = {}, {}
    if problem == "prediction_nan":
        prediction[0] = np.nan
    elif problem == "prediction_shape":
        prediction = prediction[:-1]
    elif problem == "target_nan":
        data.loc[0, "lst_c"] = np.nan
    elif problem == "missing_group":
        data.loc[0, "label_product"] = None
    elif problem == "wrong_date":
        data.loc[0, "utc_day"] += pd.Timedelta(days=1)
    elif problem == "duplicate_id":
        data.loc[1, "sample_id"] = data.loc[0, "sample_id"]
    elif problem == "cross_date_acquisition":
        data.loc[1, "acquisition_id"] = data.loc[0, "acquisition_id"]
    elif problem == "string_support":
        changes, support = {"F": prediction}, {"F": np.repeat("false", len(data))}
    elif problem == "mismatched_keys":
        changes = {"F": prediction}
    with pytest.raises(ValueError):
        pm.evaluate_cohort(data, {"F": prediction}, changes, support, "fixed")


def test_empty_cohort_is_explicit_and_has_no_intervals():
    metrics, pairs = pm.evaluate_cohort(frame().iloc[:0], {"F": np.array([]), "WP": np.array([])}, {}, {}, "empty")
    assert len(metrics) == 2 and len(pairs) == 1
    assert metrics[0]["status"] == "no_observations"
    assert pairs[0]["delta_mae_c"] is None
    assert pairs[0]["delta_mae_c_ci95"] is None
    json.dumps({"metrics": metrics, "paired": pairs}, allow_nan=False)


def test_intervals_are_reproducible_under_model_dictionary_order():
    data = frame()
    first = {"F": np.arange(len(data), dtype=float), "WP": np.ones(len(data))}
    _, a = pm.evaluate_cohort(data, first, {}, {}, "fixed")
    _, b = pm.evaluate_cohort(data, dict(reversed(list(first.items()))), {}, {}, "fixed")
    assert a == b
