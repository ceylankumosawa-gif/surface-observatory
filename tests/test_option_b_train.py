"""Research trainer contract tests; run on Hetzner, no external data access."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from lst_pilot import option_b_train as train


def paired():
    records = []
    dates = ["2021-01-01", "2021-01-02", "2023-01-01", "2023-02-01", "2023-08-01", "2023-09-01"]
    for region in ("greater_london", "sioux_falls", "cabauw"):
        for date in dates:
            for pixel in range(4):
                row = {f: 1.0 for f in train.BASE_FEATURES + train.MEMORY_FEATURES}
                row.update(sample_id=f"{region}/{date}/{pixel}", region_id=region,
                           datetime_utc=date + "T00:00:00Z", latitude=50 + pixel / 100,
                           longitude=0.0, lst_c=15.0, air_temperature_c=10.0,
                           climate_class="Cfb", solar_elevation_deg=-20.0,
                           era5_snow_water_equivalent_m=0.0, label_product="ECOSTRESS_V002",
                           acquisition_id=f"{region}/{date}", block_id="reserved" if pixel == 3 else "known",
                           spatial_holdout=pixel == 3, in_holdout_buffer=pixel == 2,
                           cohort_origin="legacy" if pixel == 0 else "expanded",
                           weight_surface_group="tree" if pixel % 2 else "built")
                records.append(row)
    return pd.DataFrame(records)


def test_date_boundaries_whole_region_and_spatial_exclusions():
    data = train.prepare_input(paired())
    usable = data.loc[data.split.isin(["fit", "development", "calibration"])]
    assert not usable.region_id.eq("cabauw").any()
    assert not usable.spatial_holdout.any()
    assert not usable.in_holdout_buffer.any()
    assert usable.loc[usable.split.eq("fit"), "datetime_utc"].max() < pd.Timestamp("2023-01-01", tz="UTC")
    assert usable.loc[usable.split.eq("development"), "datetime_utc"].max() < pd.Timestamp("2023-07-01", tz="UTC")
    assert usable.loc[usable.split.eq("calibration"), "datetime_utc"].min() >= pd.Timestamp("2023-07-01", tz="UTC")
    assert data.loc[data.region_id.eq("cabauw"), "split"].eq("heldout_region").all()


@pytest.mark.parametrize("stamp", ["2020-12-31T23:59:00Z", "2024-01-01T00:00:00Z", "2025-04-01T12:00:00Z"])
def test_later_test_labels_and_preperiod_rejected_before_fitting(stamp):
    data = paired()
    data.loc[0, "datetime_utc"] = stamp
    with pytest.raises(ValueError, match="2021--2023"):
        train.prepare_input(data)


def test_file_reader_rejects_blind_dates_without_reading_thermal_columns(monkeypatch):
    calls = []
    def read(path, columns=None):
        calls.append(columns)
        assert columns == ["datetime_utc"], "Thermal table must never be read for a rejected file"
        return pd.DataFrame({"datetime_utc": ["2025-01-01T00:00:00Z"]})
    monkeypatch.setattr(pd, "read_parquet", read)
    with pytest.raises(ValueError, match="thermal columns were not loaded"):
        train.load_paired_input("not-actually-opened.parquet")
    assert calls == [["datetime_utc"]]


def test_string_false_is_not_silently_true():
    data = paired().astype({"spatial_holdout": object})
    data.loc[0, "spatial_holdout"] = "False"
    with pytest.raises(ValueError, match="explicit nonmissing booleans"):
        train.prepare_input(data)


def test_duplicate_physical_label_not_an_independent_sample():
    data = paired()
    duplicate = data.iloc[[0]].copy()
    duplicate["sample_id"] = "different-version-same-pixel"
    duplicate["label_product"] = "ECOSTRESS_V003"
    with pytest.raises(ValueError, match="Duplicate physical sample"):
        train.prepare_input(pd.concat([data, duplicate], ignore_index=True))


def test_acquisition_cannot_cross_time_partitions():
    data = paired()
    data.loc[data.datetime_utc.eq("2023-01-01T00:00:00Z") & data.region_id.eq("greater_london"), "acquisition_id"] = "greater_london/2021-01-01"
    with pytest.raises(ValueError, match="crosses temporal partitions"):
        train.prepare_input(data)


def test_reserved_blocks_cannot_be_relabelled_on_other_dates():
    data = paired()
    mask = data.region_id.eq("greater_london") & data.block_id.eq("reserved")
    data.loc[data.index[mask][0], "spatial_holdout"] = False
    with pytest.raises(ValueError, match="changes its reserved status"):
        train.prepare_input(data)


def test_date_and_surface_weights_are_invariant_to_more_pixels():
    data = train.prepare_input(paired())
    data = data.loc[data.split.eq("fit")].copy()
    copies = pd.concat([data.iloc[[0]]] * 17, ignore_index=True)
    copies["sample_id"] = [f"extra-{i}" for i in range(len(copies))]
    expanded = pd.concat([data, copies], ignore_index=True)
    original = data.assign(weight=train.balanced_weights(data))
    extra = expanded.assign(weight=train.balanced_weights(expanded))
    keys = ["region_id", "phase", "utc_day", "acquisition_id", "weight_surface_group"]
    pd.testing.assert_series_equal(original.groupby(keys).weight.sum(), extra.groupby(keys).weight.sum())
    assert np.isclose(extra.weight.sum(), 1)


def test_matched_features_remove_same_rows_from_A_and_B():
    data = paired()
    missing_id = data.iloc[0].sample_id
    data.loc[0, train.MEMORY_FEATURES[0]] = np.nan
    cohorts, evaluation, audit = train.candidate_cohorts(train.prepare_input(data))
    assert cohorts["A"].sample_id.tolist() == cohorts["B"].sample_id.tolist()
    assert missing_id not in set(cohorts["A"].sample_id)
    assert missing_id in set(cohorts["S0"].sample_id)
    np.testing.assert_array_equal(train.balanced_weights(cohorts["A"]), train.balanced_weights(cohorts["B"]))
    assert audit["incomplete_or_twilight_rows"] == 1


def test_one_new_night_date_reported_without_blocking_other_groups():
    data = paired()
    remove = data.region_id.eq("sioux_falls") & data.datetime_utc.eq("2021-01-02T00:00:00Z") & data.cohort_origin.eq("expanded")
    data = data.loc[~remove]
    cohorts, _, audit = train.candidate_cohorts(train.prepare_input(data))
    assert len(cohorts["A"]) > 0
    unsupported = cohorts["A"].region_id.eq("sioux_falls") & cohorts["A"].cohort_origin.eq("expanded")
    assert not unsupported.any()
    assert audit["excluded_sparse_fit_groups"][0]["fit_dates"] == 1


def test_feature_sets_never_contain_labels_or_identity():
    forbidden = {"lst_c", "latitude", "longitude", "sample_id", "acquisition_id", "label_product", "region_id", "emissivity", "max_source_lst_error_k", "era5_skin_temperature_c"}
    for names in train.FEATURE_SETS.values():
        assert not set(names) & forbidden
    assert len(train.FEATURE_SETS["A"]) == 40
    assert len(train.FEATURE_SETS["B"]) == 51
    assert len(train.FEATURE_SETS["C"]) == 45
    assert len(train.FEATURE_SETS["D"]) == 56


def test_centered_contrast_separates_whole_acquisition_bias():
    data = train.prepare_input(paired()).iloc[:4]
    prediction = data.lst_c.to_numpy() + 9
    stats = train.metrics(data, prediction)
    assert stats["mae_c"] == pytest.approx(9)
    assert stats["centered_contrast_mae_c"] == pytest.approx(0)
    assert stats["fraction_abs_error_gt_7c"] == pytest.approx(1)


def scores(mae, dates=3):
    return {"overall": {"mae_c": mae, "date_count": dates}, "by_region_phase": {"greater_london|night": {"mae_c": mae, "date_count": dates}}}


def test_selection_prefers_fewer_features_near_best_and_requires_support():
    support = [{"region_id": "greater_london", "phase": "night", "expanded_dates": 2}]
    result = train.choose_candidate({"A": scores(2), "B": scores(1.7), "C": scores(1.73)}, support)
    assert result["selected_candidate"] == "C"
    result = train.choose_candidate({"A": scores(2, dates=1), "B": scores(0.1, dates=1)}, support)
    assert result["selected_candidate"] == "A"
    assert result["status"] == "unsupported_development"


def test_sparse_legacy_groups_do_not_prevent_supported_expanded_selection():
    support = [{"region_id": "greater_london", "phase": "night", "expanded_dates": 2},
               {"region_id": "gobabeb", "phase": "day", "expanded_dates": 0}]
    base, candidate = scores(2), scores(1.7)
    base["by_region_phase"]["gobabeb|day"] = {"mae_c": 2, "date_count": 1}
    candidate["by_region_phase"]["gobabeb|day"] = {"mae_c": 1.9, "date_count": 1}
    selected = train.choose_candidate({"A": base, "B": candidate}, support)
    assert selected["selected_candidate"] == "B"
    assert selected["sparse_observed_groups"] == ["gobabeb|day"]
    candidate["by_region_phase"]["gobabeb|day"]["mae_c"] = 2.5
    selected = train.choose_candidate({"A": base, "B": candidate}, support)
    assert selected["selected_candidate"] == "A"
    assert selected["checks"]["B"]["regressing_groups"] == ["gobabeb|day"]


def test_missing_expanded_phase_development_still_blocks_selection():
    support = [{"region_id": "greater_london", "phase": "night", "expanded_dates": 2},
               {"region_id": "sioux_falls", "phase": "night", "expanded_dates": 2}]
    selected = train.choose_candidate({"A": scores(2), "B": scores(1.5)}, support)
    assert selected["status"] == "unsupported_development"
    assert selected["missing_or_sparse_groups"] == ["sioux_falls|night"]


def test_cap_is_deterministic_under_row_reordering():
    data = train.prepare_input(paired())
    assert train.cap_rows(data, 10).sample_id.tolist() == train.cap_rows(data.sample(frac=1, random_state=1), 10).sample_id.tolist()


class ConstantRegressor:
    def fit(self, x, y, **kwargs):
        assert "regressor__sample_weight" in kwargs
        self.constant = np.average(y, weights=kwargs["regressor__sample_weight"])
        self.columns = list(x.columns)
        return self

    def predict(self, x):
        assert list(x.columns) == self.columns
        return np.full(len(x), self.constant)


def test_end_to_end_keeps_holdout_labels_out_of_fit_and_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "build_estimators", lambda *args: (ConstantRegressor(), ConstantRegressor()))
    data = paired()
    # Large geographic errors cannot change the fitted mean or dev selection.
    data.loc[data.region_id.eq("cabauw") | data.spatial_holdout | data.in_holdout_buffer, "lst_c"] = 999.0
    output = tmp_path / "run"
    result = train.run_experiment(data, output)
    assert result["selection"]["selected_candidate"] == "A"
    assert result["candidates"]["A"]["evaluation"]["development"]["model"]["overall"]["mae_c"] == pytest.approx(0)
    assert result["candidates"]["A"]["evaluation"]["heldout_region"]["model"]["overall"]["mae_c"] > 900
    fitting = pd.read_parquet(output / "fitting_rows.parquet")
    forbidden_ids = set(data.loc[data.region_id.eq("cabauw") | data.spatial_holdout | data.in_holdout_buffer, "sample_id"])
    assert not set(fitting.sample_id) & forbidden_ids
    assert (output / "selection.json").exists()
    assert (output / "frozen_manifest.json").exists()
    assert result["auto_promotion"] is False
    json.loads((output / "results.json").read_text())


def test_empty_development_still_saves_exploratory_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "build_estimators", lambda *args: (ConstantRegressor(), ConstantRegressor()))
    data = paired()
    data = data.loc[~data.datetime_utc.isin(["2023-01-01T00:00:00Z", "2023-02-01T00:00:00Z"])]
    result = train.run_experiment(data, tmp_path / "run")
    assert result["selection"]["status"] == "unsupported_development"
    assert result["selected_model_path"].endswith("A/model.joblib")


def frozen_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "build_estimators", lambda *args: (ConstantRegressor(), ConstantRegressor()))
    baseline = ConstantRegressor()
    baseline.constant = 4.0
    baseline.columns = list(train.BASE_FEATURES)
    baseline_path = tmp_path / "baseline.joblib"
    joblib.dump({"model": baseline, "features": list(train.BASE_FEATURES)}, baseline_path)
    run = tmp_path / "fitted"
    train.run_experiment(paired(), run, baseline_model_path=baseline_path)
    return run, baseline_path


def test_modified_selection_rejected_before_legacy_input_is_read(tmp_path, monkeypatch):
    run, baseline = frozen_fixture(tmp_path, monkeypatch)
    (run / "selection.json").write_text("{}")
    monkeypatch.setattr(train, "load_paired_input", lambda *args, **kwargs: pytest.fail("No data read before frozen hash verification"))
    with pytest.raises(ValueError, match="Frozen artifact hash mismatch"):
        train.evaluate_legacy_2024("not-opened.parquet", run, tmp_path / "legacy", baseline_model_path=baseline)


def test_separate_legacy_evaluator_never_fits_and_keeps_selected_model(tmp_path, monkeypatch):
    run, baseline = frozen_fixture(tmp_path, monkeypatch)
    data = paired()
    data = data.loc[data.datetime_utc.str.startswith("2021")].copy()
    data["datetime_utc"] = data.datetime_utc.str.replace("2021", "2024")
    path = tmp_path / "synthetic_legacy.parquet"
    data.to_parquet(path, index=False)
    original_model_hash = train.sha(run / "A/model.joblib")
    monkeypatch.setattr(train, "build_estimators", lambda *args: pytest.fail("Legacy evaluation must never construct a new estimator"))
    result = train.evaluate_legacy_2024(path, run, tmp_path / "legacy", baseline_model_path=baseline)
    assert result["selected_candidate"] == "A"
    assert result["metrics"]["candidate"]["overall"]["mae_c"] == pytest.approx(0)
    london = result["metrics"]["candidate"]["by_region"]["greater_london"]
    assert london["fraction_abs_error_gt_5c"] == 0
    assert london["fraction_abs_error_gt_7c"] == 0
    assert train.sha(run / "A/model.joblib") == original_model_hash
    assert result["status"] == "legacy_2024_evaluated_no_refit_no_promotion"


def test_legacy_reader_rejects_2025_before_thermal_columns(monkeypatch):
    calls = []
    def read(path, columns=None):
        calls.append(columns)
        assert columns == ["datetime_utc"]
        return pd.DataFrame({"datetime_utc": ["2025-01-01T00:00Z"]})
    monkeypatch.setattr(pd, "read_parquet", read)
    with pytest.raises(ValueError, match="thermal columns were not loaded"):
        train.load_paired_input("not-opened.parquet", evaluation_2024=True)
    assert len(calls) == 1
