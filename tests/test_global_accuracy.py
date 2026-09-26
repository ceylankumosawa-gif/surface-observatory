import numpy as np
import pandas as pd
import pytest
from lst_global.accuracy import Policy, evaluate


def fixtures():
    dates = [f"2023-{month:02d}-{day:02d}T12:00:00Z" for month in (1, 4, 7, 10) for day in (1, 2, 3)]
    reference = pd.DataFrame({"sample_id": [str(i) for i in range(120)], "region_id": "a", "phase": "day",
                              "resolution_m": 100, "validation_kind": "date_holdout",
                              "datetime_utc": np.repeat(dates, 10), "lst_c": 0.})
    predictions = reference[["sample_id"]].assign(predicted_lst_c=0., model_fit_id="fit")
    training = pd.DataFrame({"sample_id": ["train"], "region_id": ["a"], "datetime_utc": ["2022-01-01T00:00Z"], "model_fit_id": ["fit"]})
    requirements = reference[["region_id", "phase", "resolution_m", "validation_kind"]].drop_duplicates().to_dict("records")
    return reference, predictions, training, requirements


def run(args, **kwargs):
    return evaluate(*args, model_id="test", evidence_status="fresh_locked_confirmation", **kwargs)


def test_panel_pass_never_becomes_a_global_claim():
    out = run(fixtures())
    assert out["regional_panel_target_met"] and out["passed_groups"] == 1
    assert out["global_target_met"] is False


def test_missing_hard_predictions_cannot_pass_by_masking():
    args = fixtures(); args[1].loc[0, "predicted_lst_c"] = np.nan
    out = run(args)
    assert not out["regional_panel_target_met"]
    assert out["groups"][0]["reference_rows"] == 120
    assert "missing_predictions" in out["groups"][0]["reasons"]


def test_missing_night_and_resolution_are_counted_not_ignored():
    args = fixtures(); args[3].extend([{**args[3][0], "phase": "night"}, {**args[3][0], "resolution_m": 1000}])
    out = run(args)
    assert out["required_groups"] == 3 and out["missing_groups"] == 2 and out["passed_groups"] == 1
    assert not out["regional_panel_target_met"]


def test_omitted_observed_group_is_an_error():
    args = fixtures(); args[0].loc[0, "phase"] = "night"
    with pytest.raises(ValueError, match="omitted"):
        run(args)


@pytest.mark.parametrize("kind", ["date_holdout", "region_holdout"])
def test_actual_training_date_or_region_overlap_blocks_acceptance(kind):
    args = fixtures(); args[0]["validation_kind"] = kind; args[3][0]["validation_kind"] = kind
    if kind == "date_holdout": args[2]["datetime_utc"] = "2023-01-20T00:00Z"
    out = run(args)
    assert "training_overlap" in out["groups"][0]["reasons"]


def test_repeated_evidence_is_not_fresh_confirmation():
    out = evaluate(*fixtures(), model_id="test")
    assert "previously_inspected_evidence" in out["groups"][0]["reasons"]


def test_equal_date_metric_cannot_be_overwhelmed_by_dense_good_pixels():
    args = fixtures()
    args[1].loc[:9, "predicted_lst_c"] = 24.
    repeated = pd.concat([args[0], args[0].iloc[10:].copy()], ignore_index=True)
    repeated.loc[120:, "sample_id"] = ["extra" + str(i) for i in range(110)]
    pred = repeated[["sample_id"]].assign(predicted_lst_c=0., model_fit_id="fit")
    pred.loc[:9, "predicted_lst_c"] = 24.
    out = run((repeated, pred, args[2], args[3]))["groups"][0]
    assert out["date_balanced_mae_c"] == pytest.approx(2.)
    assert out["pixel_mae_c"] == pytest.approx(240/230)
    assert "quarter_1_mae_target_failed" in out["reasons"]


def test_zero_negative_errors_and_exact_three_are_preserved():
    args = fixtures(); args[1]["predicted_lst_c"] = -3.
    out = run(args)
    assert out["passed_groups"] == 1
    assert out["groups"][0]["pixel_bias_c"] == -3.


def test_small_date_count_is_not_optional_even_with_many_pixels():
    args = fixtures(); args[0]["datetime_utc"] = "2023-01-01T00:00Z"
    out = run(args)
    assert "insufficient_independent_dates" in out["groups"][0]["reasons"]
    assert "insufficient_seasonal_dates" in out["groups"][0]["reasons"]


def test_duplicate_predictions_and_unknown_fit_cannot_enter_metrics():
    args = fixtures()
    with pytest.raises(ValueError, match="duplicate"):
        run((args[0], pd.concat([args[1], args[1].iloc[:1]]), args[2], args[3]))
    args[1]["model_fit_id"] = "invented"
    with pytest.raises(ValueError, match="unknown fitting"):
        run(args)


def blend_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/"reports/global_release_20260918/accuracy/run_experiment.py"
    spec = importlib.util.spec_from_file_location("bounded_skin_blend", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_skin_blend_requires_independent_dates_and_preserves_fallback():
    module = blend_module()
    frame, _, _, _ = fixtures()
    frame["datetime_utc"] = pd.to_datetime(frame.datetime_utc,utc=True)-pd.DateOffset(years=1)
    frame = frame.assign(utc_day=frame.datetime_utc.dt.strftime('%Y-%m-%d'), acquisition_id=frame.datetime_utc.astype(str),
                         weight_surface_group="ground", aligned_swe_m=0., aligned_skin_c=10., lst_c=5.)
    f = np.zeros(len(frame))
    coefficients = module.fit_coefficients(frame, f)
    assert 0 < coefficients['day|no_snow']['lambda'] < .5
    candidate, _ = module.apply_coefficients(frame, f, coefficients)
    assert np.all((candidate > 0) & (candidate < 5))
    assert coefficients['night|snow']['supported'] is False
    frame['datetime_utc'] = pd.Timestamp('2022-01-01',tz='UTC'); frame['utc_day'] = '2022-01-01'
    sparse = module.fit_coefficients(frame,f)
    candidate, _ = module.apply_coefficients(frame, f, sparse)
    np.testing.assert_array_equal(candidate, f)
    assert not sparse['day|no_snow']['supported']


def test_blend_rejects_evaluation_year_during_coefficient_fit():
    module = blend_module()
    frame, _, _, _ = fixtures()
    with pytest.raises(ValueError, match="later labels"):
        module.fit_coefficients(frame, np.zeros(len(frame)))


def surfrad_module():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'reports/global_release_20260918/accuracy/collect_surfrad.py'
    spec=importlib.util.spec_from_file_location('bounded_ground_radiometry',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def ground_file(tmp_path, bad_qc=False):
    module=surfrad_module()
    row=[2021,15,1,15,0,0,0.,120.]
    pairs={name:0. for name in module.reference.PAIRS}
    pairs.update(longwave_down_w_m2=250.,longwave_up_w_m2=280.,air_temperature_10m_c=-5.,relative_humidity_percent=90.)
    for name in module.reference.PAIRS: row.extend([pairs[name],int(bad_qc and name=='longwave_up_w_m2')])
    path=tmp_path/'tbl21015.dat';path.write_text('Table Mountain\n40.12498 -105.23680 1689 4\n'+' '.join(map(str,row))+'\n')
    return module,path


def test_ground_radiometry_preserves_negative_temperature_and_qc(tmp_path):
    module,path=ground_file(tmp_path)
    frame=module.parse(path,'2021-01-15','tbl')
    assert frame.radiometric_e0p97_c.iloc[0]<0 and frame.air_temperature_10m_c.iloc[0]==-5
    assert not frame.training_eligible.any() and frame.phase.iloc[0]=='night'
    assert frame.emissivity_sensitivity_max_c.iloc[0]>frame.emissivity_sensitivity_min_c.iloc[0]


def test_ground_qc_flag_rejects_only_bad_value_without_dropping_reference_time(tmp_path):
    module,path=ground_file(tmp_path,bad_qc=True)
    frame=module.parse(path,'2021-01-15','tbl')
    assert len(frame)==1 and frame.radiometric_e0p97_c.isna().all()
    assert frame.longwave_up_w_m2_raw.iloc[0]==280.


def test_ground_expected_date_is_enforced(tmp_path):
    module,path=ground_file(tmp_path)
    with pytest.raises(ValueError,match='minute times'):
        module.parse(path,'2021-04-15','tbl')
