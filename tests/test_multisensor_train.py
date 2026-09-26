"""Synthetic tests of month separation, date-scale regularization and frozen replay."""
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
import pytest

from lst_pilot import multisensor_train as multi
from lst_pilot import option_b_train as old
from lst_pilot import more_days_train as more
from lst_pilot.option_b_cohort import spatial_flags


class SmallRegressor:
    def fit(self, x, y, **kwargs):
        self.columns = list(x)
        self.value = float(np.average(y, weights=kwargs["regressor__sample_weight"]))
        return self

    def predict(self, x):
        assert list(x) == self.columns
        return np.full(len(x), self.value)


def areas():
    return {name: {"id": name, "epsg": 32630, "extent_m": [500000,5700000,580000,5780000],
                   "grid_shape": [800,800]} for name in ("greater_london", "sioux_falls", "cabauw")}


def rows(dates, *, regions=("greater_london", "sioux_falls", "cabauw"), tag="old",
         product="landsat_c2_l2", all_safe=False):
    result = []
    project = Transformer.from_crs(32630, 4326, always_xy=True)
    for region in regions:
        for date in dates:
            for pixel in range(4):
                rr, cc = ((350, 150+pixel) if all_safe or pixel < 2 else
                          ((389,150) if pixel == 2 else (450,150)))
                lon, lat = project.transform(500000+(cc+.5)*100,5780000-(rr+.5)*100)
                row = {f: 1.0 for f in old.BASE_FEATURES}
                row.update(sample_id=f"{tag}:{region}:{date}:{pixel}", region_id=region,
                    datetime_utc=date+"T12:00:00Z", latitude=lat, longitude=lon,
                    lst_c=15.0+int(date[5:7])/10+pixel/20, air_temperature_c=10.0,
                    climate_class="Cfb" if region != "sioux_falls" else "Dfa",
                    solar_elevation_deg=30.0, era5_snow_water_equivalent_m=0.0,
                    label_product=product, acquisition_id=f"{tag}:{region}:{date}",
                    cohort_origin="expanded", weight_surface_group="tree" if pixel%2 else "built",
                    grid_row=rr, grid_col=cc, research_admissibility_reason="",
                    source_screen_pass=True, label_source_sha256="a"*64, native_fit_support_pass=True)
                result.append(row)
    return spatial_flags(pd.DataFrame(result), areas())


def prepared(dates=None, **kwargs):
    return old.prepare_input(rows(dates or [f"2021-{m:02d}-10" for m in range(1,7)],
                                   regions=("greater_london",), all_safe=True, **kwargs))


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("multisensor")
    dates = [f"2021-{m:02d}-01" for m in range(1,7)]+["2023-01-01","2023-02-01","2023-08-01","2023-09-01"]
    frame = rows(dates)
    frame.to_parquet(root/"original.parquet",index=False)
    estimator = SmallRegressor().fit(frame[list(multi.BASE)],np.full(len(frame),2.0),
                                    regressor__sample_weight=np.ones(len(frame)))
    joblib.dump({"model":estimator,"features":list(multi.BASE)},root/"v1.joblib")
    (root/"protocol.md").write_text("Frozen synthetic multisensor specification.")
    (root/"areas.json").write_text(json.dumps({"areas":list(areas().values())}))
    (root/"freshness.json").write_text(json.dumps({"version":"multisensor-prior-date-registry-v1",
        "dates":[{"region_id":region,"utc_date":day} for region in areas() for day in dates]}))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(old,"build_estimators",lambda *args:(SmallRegressor(),SmallRegressor()))
        old.run_experiment(frame,root/"a_run",baseline_model_path=root/"v1.joblib",
                          protocol_path=root/"protocol.md",input_path=root/"original.parquet")
        legacy = rows(["2024-01-01","2024-02-01"])
        legacy.to_parquet(root/"2024.parquet",index=False)
        old.evaluate_legacy_2024(root/"2024.parquet",root/"a_run",root/"a_2024",baseline_model_path=root/"v1.joblib")
        rows(["2022-07-01","2022-08-01","2022-09-01"],regions=("greater_london","sioux_falls"),
             tag="e_add",all_safe=True).to_parquet(root/"e_new.parquet",index=False)
        more.run_experiment(root/"original.parquet",root/"e_new.parquet",root/"a_run",root/"areas.json",
                            root/"v1.joblib",root/"protocol.md",root/"e_run",original_2024_dir=root/"a_2024")
        more.evaluate_legacy_2024(root/"2024.parquet",root/"e_run",root/"a_run",root/"a_2024",root/"v1.joblib",root/"e_2024")
    rows(["2021-01-01","2022-04-10"],regions=("greater_london","sioux_falls"),tag="new_fine",
         product="ecostress_v2",all_safe=True).to_parquet(root/"new_fit.parquet",index=False)
    evaluation = rows(["2023-03-10","2023-10-10"],tag="new_eval",product="aster_ast08_v004")
    evaluation = evaluation.loc[~evaluation.in_holdout_buffer].copy()
    evaluation["fresh_2023"] = True
    evaluation["freshness_audit_sha256"] = old.sha(root/"freshness.json")
    evaluation.to_parquet(root/"new_eval.parquet",index=False)
    return root


def run(reference, output, monkeypatch):
    monkeypatch.setattr(old,"build_estimators",lambda *args:(SmallRegressor(),SmallRegressor()))
    return multi.run_experiment(original_input=reference/"original.parquet",e_additions=reference/"e_new.parquet",
        original_run=reference/"a_run",e_run=reference/"e_run",e_2024=reference/"e_2024",
        new_fit=reference/"new_fit.parquet",new_evaluation=reference/"new_eval.parquet",
        freshness_audit=reference/"freshness.json",areas_path=reference/"areas.json",
        baseline=reference/"v1.joblib",protocol=reference/"protocol.md",output=output)


@pytest.mark.parametrize("stamp,evaluation",[("2023-01-01Z",False),("2024-01-01Z",False),
    ("2025-01-01Z",False),("2022-01-01Z",True),("2024-01-01Z",True),("2025-01-01Z",True)])
def test_year_guard_precedes_thermal_decode(monkeypatch,stamp,evaluation):
    calls=[]
    def read(path,columns=None):
        calls.append(columns)
        assert columns==["datetime_utc"]
        return pd.DataFrame({"datetime_utc":[stamp.replace("Z","T00:00:00Z")]})
    monkeypatch.setattr(pd,"read_parquet",read)
    with pytest.raises(ValueError,match="thermal columns were not loaded"):
        multi.load_new("forbidden",evaluation=evaluation)
    assert calls==[["datetime_utc"]]


@pytest.mark.parametrize("field,value,match",[("source_screen_pass",False,"source screen"),
    ("source_screen_pass","true","explicit nonmissing booleans"),
    ("native_fit_support_pass",False,"native contributor"),
    ("label_source_sha256","bad","SHA-256"),
    ("label_product","modis_1km","fine labels"),
    ("research_admissibility_reason","failed_QA","passed research admission"),
    ("ndvi",np.nan,"complete base40"),
    ("longitude",0.0,"Coordinates disagree")])
def test_fine_admission_gate(field,value,match):
    new=rows(["2022-01-10"],regions=("greater_london",),tag="fine",product="ecostress_v2",all_safe=True)
    new[field]=value
    with pytest.raises(ValueError,match=match):
        multi.validate_new(new,prepared(),areas())


def test_nullable_missing_reason_is_not_admitted():
    new=rows(["2022-01-10"],regions=("greater_london",),product="ecostress_v2",all_safe=True)
    new["research_admissibility_reason"]=pd.Series(pd.NA,index=new.index,dtype="string")
    with pytest.raises(ValueError,match="passed research admission"):
        multi.validate_new(new,prepared(),areas())


def test_new_sensor_same_pilot_date_allowed_but_same_physical_cell_rejected():
    reference=prepared()
    new=rows(["2021-01-10"],regions=("greater_london",),tag="fine",product="ecostress_v2",all_safe=True)
    assert len(multi.validate_new(new,reference,areas()))==4
    new["acquisition_id"]=reference.acquisition_id.iloc[0]
    with pytest.raises(ValueError,match="physical acquisition/grid cell overlaps"):
        multi.validate_new(new,reference,areas())


def test_whole_cell_buffer_cannot_be_falsely_marked_safe():
    new=rows(["2022-01-10"],regions=("greater_london",),tag="fine",product="ecostress_v2")
    new=new.iloc[[2]].copy()
    new["in_holdout_buffer"]=False
    with pytest.raises(ValueError,match="whole-cell spatial classification"):
        multi.validate_new(new,prepared(),areas())


def test_join_preserves_all_e_values_and_does_not_cap_old_rows():
    original=prepared()
    extra=prepared(["2022-05-10"],tag="fine",product="ecostress_v2")
    result=multi.join_fitting(original,extra)
    assert len(result)==len(original)+len(extra)
    pd.testing.assert_frame_equal(result.set_index("sample_id").loc[original.sample_id],original.set_index("sample_id"))
    bad=extra.copy();bad["split"]="calibration"
    with pytest.raises(ValueError,match="Only fitting"):
        multi.join_fitting(original,bad)


def test_months_held_globally_across_sensor_pilot_and_phase():
    frame=old.prepare_input(rows(["2021-01-02","2021-01-24","2022-01-02"],
        regions=("greater_london","sioux_falls"),all_safe=True))
    frame.loc[frame.index%2==0,"label_product"]="ecostress_v2"
    frame.loc[frame.index%2==0,"solar_elevation_deg"]=-20
    fold=multi.month_folds(frame)
    assert set(fold)=={0}
    assert len(set(zip(frame.region_id,frame.label_product)))==4


@pytest.mark.parametrize("field,value",[("region_id","cabauw"),("spatial_holdout",True),
                                       ("in_holdout_buffer",True),("split","development")])
def test_oof_excludes_reserved_rows(field,value):
    frame=prepared();frame.loc[0,field]=value
    with pytest.raises(ValueError,match="reserved geographic/temporal"):
        multi.month_folds(frame)


def test_held_month_labels_cannot_change_own_oof_prediction(monkeypatch):
    frame=prepared()
    seen=[]
    def fit(training):
        seen.append(set(training.datetime_utc.dt.strftime("%Y-%m")))
        return SmallRegressor().fit(training[list(multi.BASE)],old.target_offset(training),
                                   regressor__sample_weight=old.balanced_weights(training))
    monkeypatch.setattr(multi,"fit_base",fit)
    initial,folds,audit=multi.out_of_fold(frame)
    changed=frame.copy();changed.loc[folds==0,"lst_c"]+=1000
    repeated,_,_=multi.out_of_fold(changed)
    np.testing.assert_array_equal(initial[folds==0],repeated[folds==0])
    assert not np.allclose(initial[folds==1],repeated[folds==1])
    for record in audit: assert not set(record["fit_months"])&set(record["held_months"])


def test_ridge_penalty_invariant_to_duplicated_pixel_observations():
    frame=prepared()
    offset=2+np.arange(len(frame))/50
    first=multi.ResidualCorrection.fit(frame,offset,multi.month_folds(frame))
    copies=[]
    for i in range(9):
        part=frame.copy();part["sample_id"]=part.sample_id+f":copy{i}";copies.append(part)
    repeated=pd.concat(copies,ignore_index=True)
    second=multi.ResidualCorrection.fit(repeated,np.tile(offset,9),multi.month_folds(repeated))
    assert first.effective_utc_dates==pytest.approx(6)
    assert second.effective_utc_dates==pytest.approx(first.effective_utc_dates)
    np.testing.assert_allclose(first.mean,second.mean,atol=1e-12)
    np.testing.assert_allclose(first.ridge.coef_,second.ridge.coef_,atol=1e-10)
    np.testing.assert_allclose(first.predict(frame,offset)[0],second.predict(frame,offset)[0],atol=1e-10)


def test_same_utc_date_at_more_pilots_does_not_inflate_independence():
    frame=old.prepare_input(rows(["2021-01-10"],regions=("greater_london","sioux_falls"),all_safe=True))
    assert multi.global_date_scale(frame,old.balanced_weights(frame))==pytest.approx(1)


def test_no_identity_predictors_and_correction_not_error_bound():
    frame=prepared();frame["lst_c"]=10000
    offset=np.full(len(frame),2.0)
    correction=multi.ResidualCorrection.fit(frame,offset,multi.month_folds(frame))
    expected,support=correction.predict(frame,offset)
    assert support.all() and np.max(np.abs(expected))<=3
    changed=frame.copy()
    for field in ("region_id","label_product","acquisition_id","sample_id"):changed[field]="not_a_predictor"
    changed["latitude"]=-80;changed["longitude"]=150
    np.testing.assert_array_equal(expected,correction.predict(changed,offset)[0])
    assert np.max(np.abs(frame.lst_c-(frame.air_temperature_c+offset+expected)))>9000


def test_unsupported_phase_climate_keeps_uncorrected_f():
    frame=prepared();offset=np.full(len(frame),2.0)
    correction=multi.ResidualCorrection.fit(frame,offset,multi.month_folds(frame))
    changed=frame.copy();changed["solar_elevation_deg"]=-20
    adjustment,supported=correction.predict(changed,offset)
    assert not supported.any() and not adjustment.any()
    sparse=prepared(["2021-01-10","2021-02-10","2021-03-10"])
    inactive=multi.ResidualCorrection.fit(sparse,np.ones(len(sparse)),multi.month_folds(sparse))
    assert inactive.ridge is None


def test_g_cannot_fit_residuals_from_2023():
    frame=prepared(["2023-01-10"])
    with pytest.raises(ValueError,match="only 2021--2022"):
        multi.ResidualCorrection.fit(frame,np.ones(len(frame)),np.zeros(len(frame),int))


def test_freshness_is_whole_pilot_date_and_hash_bound():
    reference=prepared()
    new=rows(["2023-04-10"],regions=("greater_london",),tag="eval",product="ecostress_v2",all_safe=True)
    new["fresh_2023"]=True;new["freshness_audit_sha256"]="b"*64
    registry={"dates":[]}
    assert multi.validate_new(new,reference,areas(),evaluation=True,freshness_sha="b"*64,registry=registry).fresh_2023.all()
    with pytest.raises(ValueError,match="Freshness audit hash"):
        multi.validate_new(new,reference,areas(),evaluation=True,freshness_sha="c"*64,registry=registry)
    new.loc[0,"fresh_2023"]=False
    with pytest.raises(ValueError,match="both fresh and repeated"):
        multi.validate_new(new,reference,areas(),evaluation=True,freshness_sha="b"*64,registry=registry)


def test_old_2023_date_cannot_be_called_fresh_even_from_different_sensor():
    reference=prepared(["2023-04-10"])
    new=rows(["2023-04-10"],regions=("greater_london",),tag="new_sensor",product="ecostress_v2",all_safe=True)
    new["fresh_2023"]=True;new["freshness_audit_sha256"]="b"*64
    with pytest.raises(ValueError,match="previously observed pilot-date"):
        multi.validate_new(new,reference,areas(),evaluation=True,freshness_sha="b"*64,registry={"dates":[]})
    new["fresh_2023"]=False
    assert not multi.validate_new(new,reference,areas(),evaluation=True,freshness_sha="b"*64,registry={"dates":[]}).fresh_2023.any()


def test_attempted_but_unadmitted_date_is_not_fresh_before_label_read(tmp_path,monkeypatch):
    registry=tmp_path/"prior.json"
    registry.write_text(json.dumps({"version":"multisensor-prior-date-registry-v1",
        "dates":[{"region_id":"greater_london","utc_date":"2023-04-10","prior_sources":["failed_attempt"]}]}))
    expected=old.sha(registry)
    calls=[]
    def read(path,columns=None):
        calls.append(columns)
        assert columns==["region_id","datetime_utc","fresh_2023","freshness_audit_sha256"],"Thermal labels were read"
        return pd.DataFrame({"region_id":["greater_london"],"datetime_utc":["2023-04-10T12:00:00Z"],
                             "fresh_2023":[True],"freshness_audit_sha256":[expected]})
    monkeypatch.setattr(pd,"read_parquet",read)
    with pytest.raises(ValueError,match="inspected/attempted-date registry"):
        multi.load_new_evaluation("not-read",registry,expected,prepared())
    assert len(calls)==1


def test_pending_new_eval_cannot_change_after_fitting(tmp_path):
    paths={name:tmp_path/name for name in ("new_evaluation","freshness_audit","areas_path")}
    for path in paths.values():path.write_text("initial")
    manifest={"paths":paths,"input_hashes":{name:old.sha(path) for name,path in paths.items()}}
    multi.verify_pending_evaluation_inputs(manifest)
    paths["new_evaluation"].write_text("changed after F/G fitting")
    with pytest.raises(ValueError,match="before labels were read"):
        multi.verify_pending_evaluation_inputs(manifest)


def test_correction_usage_and_export_are_explicit(tmp_path):
    frame=prepared();base=np.full(len(frame),2.)
    correction=multi.ResidualCorrection.fit(frame,base,multi.month_folds(frame))
    adjustment,supported=correction.predict(frame,base)
    values={"F":frame.air_temperature_c.to_numpy()+base,"G":frame.air_temperature_c.to_numpy()+base+adjustment}
    result=multi.score(frame,values,correction=correction)
    assert result["correction_usage"]["supported_rows"]==len(frame)
    file=tmp_path/"predictions.parquet"
    multi.save_predictions(frame,values,file,correction)
    stored=pd.read_parquet(file)
    np.testing.assert_array_equal(stored.G_correction_supported,supported)
    np.testing.assert_allclose(stored.G_adjustment_c,adjustment)


def test_full_run_freezes_before_new_labels_and_preserves_reference(reference,tmp_path,monkeypatch):
    output=tmp_path/"fg"
    read=pd.read_parquet
    def guarded(path,*args,**kwargs):
        if path==reference/"new_eval.parquet" and kwargs.get("columns") is None:
            freeze=json.loads((output/"fit_freeze.json").read_text())
            assert freeze["models_frozen_before_new_evaluation"]
            for name in ("F.joblib","G.joblib"):
                assert old.sha(output/name)==freeze["artifacts"][name]
        return read(path,*args,**kwargs)
    monkeypatch.setattr(pd,"read_parquet",guarded)
    result=run(reference,output,monkeypatch)
    assert result["auto_promotion"] is False
    e=json.loads((reference/"e_run/manifest.json").read_text())
    m=json.loads((output/"manifest.json").read_text())
    assert m["e_fit_rows"]==e["fit_rows"]
    assert m["e_fit_row_sha256"]==e["fit_row_sha256"]
    for split,group in result["old_evaluation"].items():
        expected=json.loads((reference/"e_run/results.json").read_text())["evaluation"][split]
        assert group["row_sha256"]==expected["row_sha256"]
        assert group["weight_sha256"]==expected["weight_sha256"]
        for name in ("E","A","v1"):
            assert group["metrics"][name]==expected["metrics"][name]
    before={n:old.sha(output/n) for n in ("F.joblib","G.joblib","fit_freeze.json")}
    evaluated=multi.evaluate_2024(reference/"2024.parquet",output,tmp_path/"legacy")
    assert evaluated["auto_promotion"] is False
    assert evaluated["predictions_sha256"]==old.sha(tmp_path/"legacy/predictions.parquet")
    assert all(old.sha(output/n)==h for n,h in before.items())
    original=json.loads((reference/"e_2024/results.json").read_text())
    assert evaluated["weight_sha256"]==original["weight_sha256"]
    for name in ("E","A","v1"): assert evaluated["metrics"][name]==original["metrics"][name]


@pytest.fixture(scope="module")
def finished(reference,tmp_path_factory):
    out=tmp_path_factory.mktemp("fg_completed")/"run"
    with pytest.MonkeyPatch.context() as patch:run(reference,out,patch)
    return out


def test_source_change_blocks_2024_before_label_read(finished,reference,tmp_path,monkeypatch):
    actual=multi.source_hashes()
    monkeypatch.setattr(multi,"source_hashes",lambda:{**actual,"source_sha256":"changed"})
    monkeypatch.setattr(pd,"read_parquet",lambda *a,**k:pytest.fail("Label read before source check"))
    with pytest.raises(ValueError,match="trainer/dependency source changed"):
        multi.evaluate_2024(reference/"2024.parquet",finished,tmp_path/"never")


def test_model_change_blocks_2024_before_label_read(finished,reference,tmp_path,monkeypatch):
    copy=tmp_path/"tampered";shutil.copytree(finished,copy)
    (copy/"G.joblib").write_bytes(b"tampered")
    monkeypatch.setattr(pd,"read_parquet",lambda *a,**k:pytest.fail("Label read before model check"))
    with pytest.raises(ValueError,match="artifact changed: G.joblib"):
        multi.evaluate_2024(reference/"2024.parquet",copy,tmp_path/"never")


def test_2024_changed_input_rejected_before_thermal_columns(finished,reference,tmp_path,monkeypatch):
    file=tmp_path/"changed.parquet";file.write_bytes(b"not-the-frozen-reference")
    monkeypatch.setattr(pd,"read_parquet",lambda *a,**k:pytest.fail("Read changed reference labels"))
    with pytest.raises(ValueError,match="pre-fit frozen hash"):
        multi.evaluate_2024(file,finished,tmp_path/"never")


def contextual(frame):
    frame=frame.copy();frame['coarse_context_eligible']=True
    frame['coarse_lst_c']=frame.air_temperature_c+np.linspace(-10,10,len(frame))
    frame['coarse_age_hours']=np.linspace(1,23,len(frame))
    return frame


def test_H_missing_context_is_exact_G_fallback_and_keeps_every_row():
    frame=contextual(prepared());oof=np.zeros(len(frame));folds=multi.month_folds(frame)
    core=multi.ResidualCorrection.fit(frame,oof,folds)
    h=multi.ContextResidualCorrection.fit(frame,oof,folds,core)
    base=SmallRegressor().fit(frame[list(multi.BASE)],oof,regressor__sample_weight=np.ones(len(frame)))
    estimator=multi.ContextCalibratedEstimator(base,core,h)
    missing=multi.context.missing_context(frame)
    assert np.array_equal(estimator.predict(missing),multi.CalibratedEstimator(base,core).predict(missing))
    assert len(estimator.predict(missing))==len(frame)


def test_H_requires_context_covered_dates_not_the_whole_core_support():
    frame=contextual(prepared());frame.loc[4:,'coarse_context_eligible']=False
    frame.loc[4:,['coarse_lst_c','coarse_age_hours']]=np.nan
    oof=np.zeros(len(frame));folds=multi.month_folds(frame);core=multi.ResidualCorrection.fit(frame,oof,folds)
    h=multi.ContextResidualCorrection.fit(frame,oof,folds,core)
    assert core.support['C|day']['supported']
    assert not h.support['C|day']['supported'] and h.ridge is None
    assert not h.predict(frame,oof)[1].any()


def test_H_regularization_is_invariant_to_duplicate_pixel_density():
    frame=contextual(prepared());oof=np.zeros(len(frame));folds=multi.month_folds(frame)
    core=multi.ResidualCorrection.fit(frame,oof,folds);h=multi.ContextResidualCorrection.fit(frame,oof,folds,core)
    repeated=pd.concat([frame]*5,ignore_index=True);rep_oof=np.zeros(len(repeated));rep_folds=multi.month_folds(repeated)
    rep_core=multi.ResidualCorrection.fit(repeated,rep_oof,rep_folds)
    rep_h=multi.ContextResidualCorrection.fit(repeated,rep_oof,rep_folds,rep_core)
    assert np.isclose(h.effective_utc_dates,rep_h.effective_utc_dates)
    assert np.allclose(h.ridge.coef_,rep_h.ridge.coef_,atol=1e-10)


def test_H_total_correction_is_bounded_and_replaces_not_stacks_G():
    frame=contextual(prepared());frame.lst_c=1000
    oof=np.zeros(len(frame));folds=multi.month_folds(frame);core=multi.ResidualCorrection.fit(frame,oof,folds)
    h=multi.ContextResidualCorrection.fit(frame,oof,folds,core)
    base=SmallRegressor().fit(frame[list(multi.BASE)],oof,regressor__sample_weight=np.ones(len(frame)))
    values=multi.ContextCalibratedEstimator(base,core,h).predict(frame)
    assert np.max(np.abs(values))<=3 and np.any(np.isclose(values,3))


def test_H_fit_rejects_mismatched_month_folds():
    frame=contextual(prepared());oof=np.zeros(len(frame));folds=multi.month_folds(frame)
    core=multi.ResidualCorrection.fit(frame,oof,folds)
    with pytest.raises(ValueError,match='H OOF'):multi.ContextResidualCorrection.fit(frame,oof,(folds+1)%3,core)


def test_H_end_to_end_freezes_before_context_evaluation_and_2024_is_G(reference,tmp_path,monkeypatch):
    output=tmp_path/'fgh';spec=tmp_path/'context_spec.json';spec.write_text('{}')
    calls=[];attach=multi.context.attach
    def checked_attach(frame,specification,area,*,for_fitting):
        calls.append(for_fitting)
        if not for_fitting:
            freeze=json.loads((output/'fit_freeze.json').read_text())
            assert 'H.joblib' in freeze['artifacts'] and 'h_correction.json' in freeze['artifacts']
        return attach(frame,specification,area,for_fitting=for_fitting)
    monkeypatch.setattr(multi.context,'attach',checked_attach)
    monkeypatch.setattr(old,'build_estimators',lambda *args:(SmallRegressor(),SmallRegressor()))
    multi.run_experiment(original_input=reference/'original.parquet',e_additions=reference/'e_new.parquet',
        original_run=reference/'a_run',e_run=reference/'e_run',e_2024=reference/'e_2024',
        new_fit=reference/'new_fit.parquet',new_evaluation=reference/'new_eval.parquet',
        freshness_audit=reference/'freshness.json',areas_path=reference/'areas.json',baseline=reference/'v1.joblib',
        protocol=reference/'protocol.md',output=output,context_spec=spec,h_protocol=reference/'protocol.md')
    assert calls==[True,False,False]
    frozen=json.loads((output/'manifest.json').read_text())
    assert frozen['h_enabled'] and frozen['context_fit_audit']['rows']==frozen['fit_rows']
    assert frozen['context_fit_audit']['eligible_rows']==0
    multi.verify_run(output)
    result=multi.evaluate_2024(reference/'2024.parquet',output,tmp_path/'legacy')
    assert calls==[True,False,False]  # 2024 cannot join/open native context.
    predicted=pd.read_parquet(tmp_path/'legacy/predictions.parquet')
    assert np.array_equal(predicted.H_lst_c,predicted.G_lst_c)
    assert result['H_equals_G_by_design'] and not result['H_context_evaluation']
    assert result['context_correction_usage']['fallback_rows_equal_G']==len(predicted)


def test_H_reports_same_supported_subset_and_independent_native_counts():
    frame=contextual(prepared());frame['coarse_product']='MOD21';frame['coarse_native_id']='shared_native_cell'
    frame['coarse_acquisition_id']='shared_native_acquisition'
    oof=np.zeros(len(frame));folds=multi.month_folds(frame);core=multi.ResidualCorrection.fit(frame,oof,folds)
    h=multi.ContextResidualCorrection.fit(frame,oof,folds,core)
    adjustment,supported=h.predict(frame,oof);g,_=core.predict(frame,oof)
    air=frame.air_temperature_c.to_numpy();values={'F':air,'G':air+g,'H':air+adjustment}
    result=multi.score(frame,values,correction=core,h_correction=h)
    usage=result['context_correction_usage']
    assert usage['unique_native_cells']==1 and usage['unique_native_acquisitions']==1
    assert usage['supported_rows']==len(frame)
    assert all(v['overall']['sample_id_sha256']==old.row_hash(frame) for v in usage['supported_subset_metrics'].values())


def test_module_cli_uses_importable_estimator_classes_in_saved_bundles(tmp_path,monkeypatch):
    import runpy
    import subprocess
    import sys
    path=tmp_path/'portable.joblib';called=[]
    def entry():
        called.append(True)
        joblib.dump(multi.ContextCalibratedEstimator(None,None,None),path)
    monkeypatch.setattr(multi,'main',entry)
    runpy.run_module('lst_pilot.multisensor_train',run_name='__main__')
    assert called==[True]
    result=subprocess.run([sys.executable,'-c',
        'import joblib,sys; value=joblib.load(sys.argv[1]); print(type(value).__module__)',str(path)],
        check=True,capture_output=True,text=True)
    assert result.stdout.strip()=='lst_pilot.multisensor_train'
