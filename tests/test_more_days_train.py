"""Fit-only ablation integrity; synthetic fixtures, no acquisitions or live data."""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
import pytest

from lst_pilot import more_days_train as more
from lst_pilot import option_b_train as old
from lst_pilot.option_b_cohort import spatial_flags


class SmallRegressor:
    def fit(self, x, y, **kwargs):
        self.columns = list(x.columns)
        self.constant = float(np.average(y, weights=kwargs["regressor__sample_weight"]))
        self.fitting_rows = len(x)
        return self

    def predict(self, x):
        assert list(x.columns) == self.columns
        return np.full(len(x), self.constant)


def areas():
    return {name:{"id":name,"epsg":32630,"extent_m":[500000,5700000,580000,5780000],"grid_shape":[800,800]}
            for name in ("greater_london","sioux_falls","cabauw","gobabeb")}


def rows(dates, regions=("greater_london","sioux_falls","cabauw"), new=False):
    records=[]
    to_lonlat=Transformer.from_crs(32630,4326,always_xy=True)
    for region in regions:
        for date in dates:
            for pixel in range(4):
                rr,cc=(350,150+pixel) if new or pixel<2 else ((389,150) if pixel==2 else (450,150))
                lon,lat=to_lonlat.transform(500000+(cc+.5)*100,5780000-(rr+.5)*100)
                record={f:1.0 for f in old.BASE_FEATURES+old.MEMORY_FEATURES+old.SURFACE_FEATURES}
                record.update(sample_id=f'{region}:{date}:{pixel}',region_id=region,datetime_utc=date+'T12:00:00Z',
                    latitude=lat,longitude=lon,lst_c=20.0 if new else 12.0,air_temperature_c=10.0,climate_class='Cfb',
                    solar_elevation_deg=20.0 if new or region!='greater_london' else -20.0,
                    era5_snow_water_equivalent_m=0.0,label_product='landsat_c2_l2',
                    acquisition_id=f'{region}:{date}',cohort_origin='expanded' if new or pixel else 'legacy',
                    weight_surface_group='tree' if pixel%2 else 'built',grid_row=rr,grid_col=cc,
                    research_admissibility_reason='')
                records.append(record)
    return spatial_flags(pd.DataFrame(records),areas())


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    root=tmp_path_factory.mktemp('more-days-reference')
    frame=rows(['2021-01-01','2021-02-01','2023-01-01','2023-02-01','2023-08-01','2023-09-01'])
    frame.to_parquet(root/'original.parquet',index=False)
    a=SmallRegressor().fit(frame[list(old.BASE_FEATURES)],np.full(len(frame),2.0),regressor__sample_weight=np.ones(len(frame)))
    joblib.dump({'model':a,'features':list(old.BASE_FEATURES)},root/'v1.joblib')
    (root/'protocol.md').write_text('Frozen synthetic one-candidate protocol.')
    (root/'areas.json').write_text(json.dumps({'areas':list(areas().values())}))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(old,'build_estimators',lambda *args:(SmallRegressor(),SmallRegressor()))
        old.run_experiment(frame,root/'original_run',baseline_model_path=root/'v1.joblib',
                          protocol_path=root/'protocol.md',input_path=root/'original.parquet')
    legacy=rows(['2024-01-01','2024-02-01'])
    legacy.to_parquet(root/'legacy2024.parquet',index=False)
    old.evaluate_legacy_2024(root/'legacy2024.parquet',root/'original_run',root/'original2024',baseline_model_path=root/'v1.joblib')
    rows(['2022-03-01','2022-04-01'],regions=('greater_london','sioux_falls'),new=True).to_parquet(root/'new.parquet',index=False)
    return root


def run(root,output,monkeypatch):
    monkeypatch.setattr(old,'build_estimators',lambda *args:(SmallRegressor(),SmallRegressor()))
    return more.run_experiment(root/'original.parquet',root/'new.parquet',root/'original_run',root/'areas.json',
                              root/'v1.joblib',root/'protocol.md',output,original_2024_dir=root/'original2024')


@pytest.mark.parametrize('stamp',['2023-01-01T00:00:00Z','2024-05-01T12:00:00Z','2025-01-01T00:00:00Z'])
def test_only_new_fit_years_are_read_before_thermal_columns(monkeypatch,stamp):
    calls=[]
    def read(path,columns=None):
        calls.append(columns)
        assert columns==['datetime_utc']
        return pd.DataFrame({'datetime_utc':[stamp]})
    monkeypatch.setattr(pd,'read_parquet',read)
    with pytest.raises(ValueError,match='thermal columns were not loaded'):
        more.load_additions('never-read-thermal.parquet')
    assert calls==[['datetime_utc']]


@pytest.mark.parametrize('change,match',[
    ({'research_admissibility_reason':'no_actual_station_pair'},'passed research admission'),
    ({'region_id':'cabauw'},'Cabauw'),
    ({'solar_elevation_deg':-20.0},'additions must be daytime'),
    ({'cohort_origin':'legacy'},'expanded Landsat'),
    ({'label_product':'ecostress_v2'},'expanded Landsat'),
    ({'ndvi':np.nan},'incomplete base40'),
    ({'longitude':0.0},'geographic coordinates'),
    ({'grid_row':800},'invalid pilot grid'),
    ({'datetime_utc':'2023-03-01T12:00:00Z'},'2023 or later'),
])
def test_defensive_new_fitting_admission(change,match):
    source=old.prepare_input(rows(['2021-01-01']))
    added=rows(['2022-03-01'],regions=('greater_london',),new=True)
    for key,value in change.items():added[key]=value
    with pytest.raises(ValueError,match=match):more.validate_additions(added,source,areas())


def test_recomputed_whole_cell_buffer_rejects_false_safe_flags():
    source=old.prepare_input(rows(['2021-01-01']))
    added=rows(['2022-03-01'],regions=('greater_london',),new=True).iloc[:1].copy()
    added['grid_row']=389;added['grid_col']=150
    lon,lat=Transformer.from_crs(32630,4326,always_xy=True).transform(515050,5741050)
    added['longitude']=lon;added['latitude']=lat
    # The centre is 1050m from the reserved block: its 100m cell reaches the 1km buffer.
    added['spatial_holdout']=False;added['in_holdout_buffer']=False
    with pytest.raises(ValueError,match='Reserved or buffer'):
        more.validate_additions(added,source,areas())


def test_nullable_missing_admission_reason_is_not_a_pass():
    source=old.prepare_input(rows(['2021-01-01']))
    added=rows(['2022-03-01'],regions=('greater_london',),new=True)
    added['research_admissibility_reason']=pd.Series(pd.NA,index=added.index,dtype='string')
    with pytest.raises(ValueError,match='passed research admission'):
        more.validate_additions(added,source,areas())


def test_existing_pilot_dates_rejected_even_with_changed_ids():
    source=old.prepare_input(rows(['2021-01-01']))
    added=rows(['2021-01-01'],regions=('greater_london',),new=True)
    added['sample_id']='alternative-'+added.sample_id;added['acquisition_id']='alternative-scene'
    with pytest.raises(ValueError,match='pilot-date overlaps'):
        more.validate_additions(added,source,areas())


def test_new_acquisition_and_sample_ids_cannot_overlap_original():
    source=old.prepare_input(rows(['2021-01-01']))
    added=rows(['2022-03-01'],regions=('greater_london',),new=True)
    added['acquisition_id']=source.iloc[0].acquisition_id
    with pytest.raises(ValueError,match='identities overlap'):
        more.validate_additions(added,source,areas())


def test_multiple_same_date_scenes_are_not_more_independent_dates():
    source=old.prepare_input(rows(['2021-01-01']))
    added=rows(['2022-03-01'],regions=('greater_london',),new=True)
    added.loc[0,'acquisition_id']='another-orbit'
    with pytest.raises(ValueError,match='one acquisition per pilot-date'):
        more.validate_additions(added,source,areas())


def test_cap_preserves_all_original_rows_and_only_limits_additions():
    source=old.prepare_input(rows(['2021-01-01','2021-02-01']))
    fitting=source.loc[source.split.eq('fit')].sort_values('sample_id')
    added=old.prepare_input(rows(['2022-03-01'],regions=('greater_london',),new=True))
    fit,chosen=more.append_fitting(fitting,added,maximum=len(fitting)+2)
    assert len(chosen)==2
    assert set(fitting.sample_id).issubset(fit.sample_id)
    assert set(fit.loc[fit.phase.eq('night'),'sample_id'])==set(fitting.loc[fitting.phase.eq('night'),'sample_id'])
    again,_=more.append_fitting(fitting,added.sample(frac=1,random_state=12),maximum=len(fitting)+2)
    assert fit.sample_id.tolist()==again.sample_id.tolist()
    # The fixed algorithm still equalizes pilot/phase groups after new daytime dates.
    weighted=fit.assign(weight=old.balanced_weights(fit))
    totals=weighted.groupby(['region_id','phase']).weight.sum()
    assert np.allclose(totals,totals.iloc[0])


def test_tampered_original_input_rejected_before_label_read(reference,tmp_path,monkeypatch):
    changed=tmp_path/'changed.parquet';changed.write_bytes(b'not-the-original-input')
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pytest.fail('Must check input hash before reading labels'))
    with pytest.raises(ValueError,match='Original paired input hash changed'):
        more.load_reference(reference/'original_run',changed,reference/'v1.joblib')


def test_one_model_fit_freezes_before_eval_and_preserves_all_evaluation_rows(reference,tmp_path,monkeypatch):
    output=tmp_path/'experiment'
    normal_predict=more.predict_all
    def predict(frame,*bundles):
        assert (output/'fit_freeze.json').exists()
        assert not (output/'results.json').exists()
        return normal_predict(frame,*bundles)
    monkeypatch.setattr(more,'predict_all',predict)
    monkeypatch.setattr(old,'choose_candidate',lambda *a:pytest.fail('No feature selection is allowed'))
    result=run(reference,output,monkeypatch)
    manifest=json.loads((output/'manifest.json').read_text())
    reference_scores=json.loads((reference/'original_run/results.json').read_text())['candidates']['A']['evaluation']
    assert result['auto_promotion'] is False
    model=joblib.load(output/'model.joblib')
    assert model['features']==list(old.BASE_FEATURES)
    assert model['model'].fitting_rows==manifest['fit_rows']
    assert manifest['fit_rows']==manifest['original_fit_rows']+manifest['new_chosen_rows']
    for split in more.SPLITS:
        got=result['evaluation'][split]
        assert got['row_sha256']==reference_scores[split]['model']['overall']['sample_id_sha256']
        assert got['weight_sha256']==manifest['fixed_evaluation'][split]['weight_sha256']
        assert {m['overall']['sample_id_sha256'] for m in got['metrics'].values()}=={got['row_sha256']}
    more.verify_experiment(output,reference/'original_run',reference/'v1.joblib')


def test_2024_last_is_same_rows_and_never_fits_or_changes_model(reference,tmp_path,monkeypatch):
    output=tmp_path/'experiment';run(reference,output,monkeypatch)
    before=old.sha(output/'model.joblib')
    monkeypatch.setattr(old,'build_estimators',lambda *a:pytest.fail('2024 must never fit'))
    result=more.evaluate_legacy_2024(reference/'legacy2024.parquet',output,reference/'original_run',
                                    reference/'original2024',reference/'v1.joblib',tmp_path/'legacy')
    expected=json.loads((reference/'original2024/results.json').read_text())
    assert result['row_sha256']==expected['metrics']['candidate']['overall']['sample_id_sha256']
    assert old.sha(output/'model.joblib')==before
    assert result['metrics']['A']['overall']['mae_c']==expected['metrics']['candidate']['overall']['mae_c']
    with pytest.raises(FileExistsError):
        more.evaluate_legacy_2024(reference/'legacy2024.parquet',output,reference/'original_run',
                                 reference/'original2024',reference/'v1.joblib',tmp_path/'legacy')


def test_model_tampering_blocks_2024_before_thermal_reads(reference,tmp_path,monkeypatch):
    output=tmp_path/'experiment';run(reference,output,monkeypatch)
    (output/'model.joblib').write_bytes(b'changed')
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pytest.fail('2024 labels must not be opened'))
    with pytest.raises(ValueError,match='frozen artifact changed'):
        more.evaluate_legacy_2024(reference/'legacy2024.parquet',output,reference/'original_run',
                                 reference/'original2024',reference/'v1.joblib',tmp_path/'legacy')


def test_2024_reference_report_cannot_change_after_fit(reference,tmp_path,monkeypatch):
    output=tmp_path/'experiment';run(reference,output,monkeypatch)
    changed=tmp_path/'changed2024';shutil.copytree(reference/'original2024',changed)
    report=json.loads((changed/'results.json').read_text());report['input_rows']+=1
    (changed/'results.json').write_text(json.dumps(report))
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pytest.fail('2024 labels must not be opened'))
    with pytest.raises(ValueError,match='2024 report or input changed'):
        more.evaluate_legacy_2024(reference/'legacy2024.parquet',output,reference/'original_run',
                                 changed,reference/'v1.joblib',tmp_path/'legacy')


def test_saved_prediction_reuse_aligns_ids_and_checks_each_value(tmp_path):
    data=pd.DataFrame({'sample_id':['one','two']})
    recorded=pd.DataFrame({'sample_id':['two','one'],'candidate':['A','A'],
                           'predicted_lst_c':[22.0,11.0],'v1_lst_c':[24.0,13.0]})
    path=tmp_path/'predictions.parquet';recorded.to_parquet(path,index=False)
    predicted={'A':np.array([11.0,22.0]),'v1':np.array([13.0,24.0])}
    result=more.reuse_reference_predictions(data,predicted,path)
    np.testing.assert_array_equal(result['A'],[11.0,22.0])
    # An unchanged overall mean cannot hide opposing errors on individual rows.
    predicted['A']=np.array([12.0,21.0])
    with pytest.raises(ValueError,match='saved row values'):
        more.reuse_reference_predictions(data,predicted,path)


@pytest.mark.parametrize('changed',['more_days_train.py','option_b_train.py'])
def test_changed_evaluation_code_blocks_2024_before_label_read(reference,tmp_path,monkeypatch,changed):
    output=tmp_path/'experiment';run(reference,output,monkeypatch)
    sha=old.sha
    monkeypatch.setattr(old,'sha',lambda path:'changed-code' if Path(path).name==changed else sha(path))
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pytest.fail('2024 labels must not be opened'))
    with pytest.raises(ValueError,match='changed after fitting'):
        more.evaluate_legacy_2024(reference/'legacy2024.parquet',output,reference/'original_run',
                                 reference/'original2024',reference/'v1.joblib',tmp_path/'legacy')
