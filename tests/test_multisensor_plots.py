"""Figure-data integrity only; never render fabricated experiment figures."""
import json

import numpy as np
import pandas as pd
import pytest

from lst_pilot import multisensor_plots as plots
from lst_pilot import option_b_train as metrics


def completed_group():
    frame=pd.DataFrame({"sample_id":["a","b","c","d"],"region_id":["greater_london"]*4,
        "datetime_utc":pd.to_datetime(["2023-01-01T12:00Z"]*3+["2023-01-02T12:00Z"],utc=True),
        "phase":["day"]*4,"air_group":["cold"]*4,"label_product":["ecostress_v2"]*4,
        "lst_c":[0.,0.,0.,0.],"E_lst_c":[0.,6.,8.,2.],"F_lst_c":[1.,7.,9.,3.],
        "G_lst_c":[.5,6.5,8.5,2.5],"G_adjustment_c":[-.5]*4,"G_correction_supported":[True]*4})
    frame["utc_day"]=frame.datetime_utc.dt.floor("D")
    output={"metrics":{}}
    for model in plots.MODELS:
        errors=frame[f"{model}_lst_c"].to_numpy()
        row={"n":4,"sample_id_sha256":metrics.row_hash(frame),"date_count":2,"utc_date_count":2,
             "unweighted_pixel_mae_c":float(np.mean(np.abs(errors))),"mae_c":4.,"bias_c":4.,
             "fraction_abs_error_gt_5c":.2,"fraction_abs_error_gt_7c":.1,"centered_contrast_mae_c":2.}
        output["metrics"][model]={"overall":row,"by_region_phase":{"greater_london|day":row}}
    return frame,output


def test_raw_tails_are_computed_on_exact_saved_rows_and_weighted_values_preserved():
    frame,result=completed_group()
    group=plots.matched_group(result,frame,"London")
    assert group["metrics"]["E"]["fraction_abs_error_gt_5c"]==.2
    assert group["metrics"]["E"]["raw_fraction_abs_error_gt_5c"]==.5
    assert group["metrics"]["E"]["raw_fraction_abs_error_gt_7c"]==.25
    assert group["date_count_kind"]=="UTC dates"


def test_mismatched_model_identity_rejected():
    frame,result=completed_group()
    result["metrics"]["G"]["overall"]["sample_id_sha256"]="different"
    with pytest.raises(ValueError,match="exactly the same observations"):
        plots.matched_group(result,frame,"London")


def test_modified_prediction_rejected_against_frozen_raw_mae():
    frame,result=completed_group();frame.loc[0,"E_lst_c"]=123
    with pytest.raises(ValueError,match="ordinary pixel MAE"):
        plots.matched_group(result,frame,"London")


def test_unsupported_rows_cannot_show_an_applied_correction():
    frame,result=completed_group();frame["G_correction_supported"]=False
    with pytest.raises(ValueError,match="Unsupported correction"):
        plots.matched_group(result,frame,"London")


def test_subgroup_masks_do_not_pool_unmatched_phase_or_climate_regime():
    frame,_=completed_group()
    frame.loc[0,"phase"]="night";frame.loc[1,"air_group"]="hot"
    selected=plots.select_rows(frame,"by_region_phase_air","greater_london|day|cold")
    assert selected.sample_id.to_list()==["c","d"]


def test_unsupported_no_observations_are_not_fabricated():
    frame,result=completed_group()
    absent=plots.matched_group(result,frame,"London night","by_region_phase","greater_london|night")
    assert absent["status"]=="unsupported_no_observations" and absent["metrics"]=={}


def test_bad_prediction_hash_blocks_even_metadata_read(tmp_path,monkeypatch):
    path=tmp_path/"predictions.parquet";path.write_bytes(b"tampered")
    monkeypatch.setattr(pd,"read_parquet",lambda *a,**k:pytest.fail("Read a changed prediction table"))
    with pytest.raises(ValueError,match="differs from"):
        plots.load_predictions(path,"wrong",[2023])


def test_future_year_rejected_before_label_read(tmp_path,monkeypatch):
    path=tmp_path/"predictions.parquet";path.write_bytes(b"test metadata")
    calls=[]
    def read(path,columns=None):
        calls.append(columns)
        assert columns==["datetime_utc"]
        return pd.DataFrame({"datetime_utc":["2025-01-01T00:00Z"]})
    monkeypatch.setattr(pd,"read_parquet",read)
    with pytest.raises(ValueError,match="label columns were not read"):
        plots.load_predictions(path,plots.sha(path),[2023])
    assert calls==[["datetime_utc"]]


def test_coverage_verifies_old_e_and_counts_overlapping_sensor_dates_separately():
    frame=pd.DataFrame({"sample_id":["a","b","c"],"original_E":[True,True,False],
        "region_id":["greater_london"]*3,"datetime_utc":["2021-01-01T12:00Z"]*3,
        "phase":["day"]*3,"label_product":["landsat_c2_l2","landsat_c2_l2","ecostress_v2"],
        "acquisition_id":["landsat","landsat","ecostress"]})
    manifest={"fit_rows":3,"e_fit_rows":2,"fit_row_sha256":metrics.row_hash(frame),
              "e_fit_row_sha256":metrics.row_hash(frame.iloc[:2])}
    rows=plots.fitting_coverage(frame,manifest)["by_source_phase"]
    assert sum(r["F_G"]["pilot_dates"] for r in rows)==2
    assert {r["label_product"]:r["E"]["rows"] for r in rows}=={"ecostress_v2":0,"landsat_c2_l2":2}
    frame.loc[0,"original_E"]=False
    with pytest.raises(ValueError,match="preservation manifest"):
        plots.fitting_coverage(frame,manifest)


def test_existing_figure_folder_is_immutable(tmp_path,monkeypatch):
    destination=tmp_path/"figures";destination.mkdir()
    monkeypatch.setattr(plots.train,"verify_run",lambda *a:pytest.fail("Should stop before reading experiment"))
    with pytest.raises(FileExistsError,match="new figure folder"):
        plots.render("not-read",destination)


def test_H_context_metrics_keep_missing_rows_and_exact_G_fallback():
    frame,results=completed_group()
    frame['H_lst_c']=frame.G_lst_c;frame.loc[0,'H_lst_c']=0.25
    frame['H_adjustment_c']=frame.H_lst_c-frame.F_lst_c
    frame['H_correction_supported']=[True,False,False,False]
    frame['coarse_context_eligible']=[True,False,False,False]
    frame['coarse_product']=['MOD21',None,None,None]
    h=dict(results['metrics']['G']['overall']);h['unweighted_pixel_mae_c']=float(frame.H_lst_c.abs().mean())
    results['metrics']['H']={'overall':h}
    output=plots.matched_group(results,frame,'London')
    assert output['metrics']['H']['n']==4
    assert output['context_correction']['fallback_rows_equal_G']==3
    assert output['context_correction']['context_covered_pilot_dates']==1
    frame.loc[1,'H_lst_c']+=.1;frame['H_adjustment_c']=frame.H_lst_c-frame.F_lst_c
    results['metrics']['H']['overall']['unweighted_pixel_mae_c']=float(frame.H_lst_c.abs().mean())
    with pytest.raises(ValueError,match='exact G fallback'):plots.matched_group(results,frame,'London')
