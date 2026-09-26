"""Independent native quality, time, geometry and composite-cohort guards."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
import pytest
from shapely.geometry import box
from lst_pilot import multisensor_context as context
from lst_pilot import coarse_join
from lst_pilot import option_b_train as old


def fixture(tmp_path):
    area={"id":"greater_london","epsg":32630,"extent_m":[500000,5700000,580000,5780000]}
    lon,lat=Transformer.from_crs(32630,4326,always_xy=True).transform(515000,5745000)
    targets=pd.DataFrame({"sample_id":["a","extra_QA_only"],"region_id":[area['id']]*2,
        "datetime_utc":["2021-05-02T12:00:00Z"]*2,"latitude":[lat]*2,"longitude":[lon]*2,"epsg":[32630]*2})
    p=box(514500,5744500,515500,5745500);support=p.buffer(650)
    native=pd.DataFrame([{ "native_cell_id":"MOD21:cell1","region_id":area['id'],"acquisition_id":"acq",
        "granule_start_utc":"2021-05-02T10:00:00Z","granule_end_utc":"2021-05-02T10:05:00Z",
        "source_sha256":"a"*64,"product":"MOD21","version":"061","native_qa_valid":True,
        "qc_raw":2<<14,"qc_mandatory":0,"qc_data":0,"qc_cloud":0,"qc_lst_error":2,"oceanpix_raw":0,
        "lst_c":20.,"lst_error_k":1.,"view_zenith_deg":0.,"footprint_epsg":32630,
        "native_footprint_wkb":p.wkb,"native_footprint_area_m2":p.area,
        "context_support_wkb":support.wkb,"context_support_margin_m":650.,"context_eligible":True,"context_fit_eligible":True}])
    targets.to_parquet(tmp_path/'targets.parquet',index=False);native.to_parquet(tmp_path/'native.parquet',index=False)
    manifest={"complete":True,"records":[{"status":"processed","table_path":str(tmp_path/'native.parquet'),"table_sha256":old.sha(tmp_path/'native.parquet')} ]}
    (tmp_path/'context.json').write_text(json.dumps(manifest))
    coarse_join.run(tmp_path/'targets.parquet',tmp_path/'context.json',tmp_path/'join',True)
    spec={"target_path":str(tmp_path/'targets.parquet'),"context_manifest_path":str(tmp_path/'context.json'),"join_dir":str(tmp_path/'join')}
    return targets.iloc[:1].copy(),native,area,spec


def test_extra_QA_rows_and_absent_model_rows_never_change_model_cohort(tmp_path):
    target,native,area,spec=fixture(tmp_path)
    missing=target.copy();missing.sample_id='absent_from_join'
    frame=pd.concat([missing,target],ignore_index=True)
    result,audit=context.attach(frame,[spec],{area['id']:area},for_fitting=True)
    assert result.sample_id.tolist()==['absent_from_join','a']
    assert result.coarse_context_eligible.tolist()==[False,True]
    assert np.isnan(result.coarse_lst_c.iloc[0]) and result.coarse_lst_c.iloc[1]==20
    assert audit['rows_absent_from_joins']==1 and audit['rows']==2


def test_repeated_identical_batches_preserve_one_row(tmp_path):
    target,native,area,spec=fixture(tmp_path)
    result,_=context.attach(target,[spec,spec],{area['id']:area},for_fitting=True)
    assert len(result)==1 and result.coarse_context_eligible.iloc[0]


def test_join_tamper_rejected_before_native_value_decode(tmp_path,monkeypatch):
    target,native,area,spec=fixture(tmp_path)
    table=Path(spec['join_dir'])/'coarse_context.parquet'
    changed=pd.read_parquet(table);changed.loc[0,'coarse_lst_c']=999;changed.to_parquet(table,index=False)
    original=pd.read_parquet
    def read(path,*args,**kwargs):
        assert Path(path)!=tmp_path/'native.parquet'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',read)
    with pytest.raises(ValueError,match='not bound'):context.attach(target,spec,{area['id']:area},for_fitting=True)


@pytest.mark.parametrize('start,end',[('2021-05-02T12:01Z','2021-05-02T12:06Z'),('2021-05-01T11:59Z','2021-05-01T12:04Z'),('2021-05-02T10:05Z','2021-05-02T10:00Z')])
def test_entire_interval_must_be_causal_and_no_more_than24hours(tmp_path,start,end):
    target,native,area,spec=fixture(tmp_path)
    data=pd.DataFrame({'sample_id':['a'],'coarse_context_eligible':[True], 'source_granule_start':[start],'source_granule_end':[end]})
    with pytest.raises(ValueError,match='future, stale'):context.check_intervals(data,target)


@pytest.mark.parametrize('field,value,match', [('qc_raw',(2<<14)|16,'raw native quality'),('context_fit_eligible',False,'geographic eligibility'),('lst_error_k',2.,'raw QA'),('view_zenith_deg',31.,'raw QA')])
def test_claimed_match_requires_independent_native_QA(tmp_path,field,value,match):
    target,native,area,spec=fixture(tmp_path)
    row=native.iloc[0].copy();row[field]=value
    joined=pd.read_parquet(Path(spec['join_dir'])/'coarse_context.parquet').iloc[0]
    with pytest.raises(ValueError,match=match):context.verify_native_row(row,joined,target.iloc[0],area,True)


def test_native_expanded_support_not_just_target_center_clears_reserved_blocks(tmp_path):
    target,native,area,spec=fixture(tmp_path)
    row=native.iloc[0].copy();joined=pd.read_parquet(Path(spec['join_dir'])/'coarse_context.parquet').iloc[0].copy()
    # Keep the safe center but enlarge support into an independently held-out block.
    p=box(509000,5739000,521000,5751000);support=p.buffer(6150)
    for key,value in {'native_footprint_wkb':p.wkb,'native_footprint_area_m2':p.area,'context_support_wkb':support.wkb,'context_support_margin_m':6150}.items():row[key]=value
    for a,b in [('coarse_footprint_wkb','native_footprint_wkb'),('coarse_native_area_m2','native_footprint_area_m2'),('coarse_support_wkb','context_support_wkb'),('coarse_support_margin_m','context_support_margin_m')]:joined[a]=row[b]
    with pytest.raises(ValueError,match='withheld block'):context.verify_native_row(row,joined,target.iloc[0],area,True)


def test_2024_context_refused_before_context_values_read(tmp_path,monkeypatch):
    target,native,area,spec=fixture(tmp_path)
    all_targets=pd.read_parquet(spec['target_path']);all_targets.datetime_utc='2024-05-02T12:00:00Z';all_targets.to_parquet(spec['target_path'],index=False)
    manifest_path=Path(spec['join_dir'])/'manifest.json';manifest=json.loads(manifest_path.read_text());manifest['input_sha256']=old.sha(spec['target_path']);manifest['for_fitting']=False;manifest_path.write_text(json.dumps(manifest))
    target=all_targets.iloc[:1];original=pd.read_parquet
    def read(path,columns=None,**kwargs):
        if Path(path).name=='coarse_context.parquet':raise AssertionError('Context values must not be read')
        return original(path,columns=columns,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',read)
    with pytest.raises(ValueError,match='target year'):context.attach(target,spec,{area['id']:area},for_fitting=False)


def test_composite_input_freeze_detects_replaced_source_before_evaluation_read(tmp_path):
    target,native,area,spec=fixture(tmp_path)
    hashes=context.freeze_inputs({'fit':[spec],'old_evaluation':[spec]})
    (tmp_path/'context.json').write_text('{}')
    with pytest.raises(ValueError,match='before evaluation'):context.verify_inputs(hashes)


def test_raw_native_timestamps_checked_before_thermal_values_even_if_join_claims_causal(tmp_path,monkeypatch):
    target,native,area,spec=fixture(tmp_path)
    native.granule_start_utc='2025-05-02T10:00:00Z';native.granule_end_utc='2025-05-02T10:05:00Z'
    native.to_parquet(tmp_path/'native.parquet',index=False)
    source=json.loads((tmp_path/'context.json').read_text());source['records'][0]['table_sha256']=old.sha(tmp_path/'native.parquet');(tmp_path/'context.json').write_text(json.dumps(source))
    table=Path(spec['join_dir'])/'coarse_context.parquet';joined=pd.read_parquet(table)
    joined.coarse_context_table_sha256=old.sha(tmp_path/'native.parquet');joined.to_parquet(table,index=False)
    manifest_path=Path(spec['join_dir'])/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest.update(context_manifest_sha256=old.sha(tmp_path/'context.json'),output_sha256=old.sha(table));manifest_path.write_text(json.dumps(manifest))
    original=pd.read_parquet;calls=[]
    def read(path,columns=None,**kwargs):
        if Path(path)==tmp_path/'native.parquet':
            calls.append(columns)
            assert columns==['native_cell_id','granule_start_utc','granule_end_utc']
        return original(path,columns=columns,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',read)
    with pytest.raises(ValueError,match='future, stale'):context.attach(target,spec,{area['id']:area},for_fitting=True)
    assert len(calls)==1


@pytest.mark.parametrize('for_fitting,extra_year',[(True,2023),(True,2025),(False,2024),(False,2025)])
def test_unselected_target_years_rejected_before_any_join_temperature_decode(tmp_path,monkeypatch,for_fitting,extra_year):
    target,native,area,spec=fixture(tmp_path)
    all_targets=pd.read_parquet(spec['target_path'])
    all_targets.loc[all_targets.sample_id.eq('extra_QA_only'),'datetime_utc']=f'{extra_year}-05-02T12:00:00Z'
    all_targets.to_parquet(spec['target_path'],index=False)
    manifest_path=Path(spec['join_dir'])/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest.update(input_sha256=old.sha(spec['target_path']),for_fitting=for_fitting);manifest_path.write_text(json.dumps(manifest))
    original=pd.read_parquet;thermal_reads=[]
    def read(path,columns=None,**kwargs):
        if Path(path).name in ('coarse_context.parquet','native.parquet') and columns is None:
            thermal_reads.append(str(path));raise AssertionError('Unused reserved-year temperature columns must not be decoded')
        return original(path,columns=columns,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',read)
    with pytest.raises(ValueError,match='target year'):context.attach(target,spec,{area['id']:area},for_fitting=for_fitting)
    assert not thermal_reads


@pytest.mark.parametrize('for_fitting,extra_year',[(True,2023),(True,2025),(False,2024),(False,2025)])
def test_unselected_native_rows_cannot_hide_reserved_year_temperatures(tmp_path,monkeypatch,for_fitting,extra_year):
    target,native,area,spec=fixture(tmp_path)
    extra=native.copy();extra.native_cell_id='unused_future_native_cell'
    extra.granule_start_utc=f'{extra_year}-05-02T10:00:00Z';extra.granule_end_utc=f'{extra_year}-05-02T10:05:00Z'
    pd.concat([native,extra],ignore_index=True).to_parquet(tmp_path/'native.parquet',index=False)
    source=json.loads((tmp_path/'context.json').read_text());source['records'][0]['table_sha256']=old.sha(tmp_path/'native.parquet')
    (tmp_path/'context.json').write_text(json.dumps(source))
    table=Path(spec['join_dir'])/'coarse_context.parquet';joined=pd.read_parquet(table)
    joined.coarse_context_table_sha256=old.sha(tmp_path/'native.parquet');joined.to_parquet(table,index=False)
    manifest_path=Path(spec['join_dir'])/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest.update(context_manifest_sha256=old.sha(tmp_path/'context.json'),output_sha256=old.sha(table),for_fitting=for_fitting)
    manifest_path.write_text(json.dumps(manifest))
    original=pd.read_parquet;native_reads=[]
    def read(path,columns=None,**kwargs):
        if Path(path)==tmp_path/'native.parquet':
            native_reads.append(columns)
            assert columns is not None and 'lst_c' not in columns
        return original(path,columns=columns,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',read)
    with pytest.raises(ValueError,match='reserved-year metadata'):context.attach(target,spec,{area['id']:area},for_fitting=for_fitting)
    assert native_reads==[['native_cell_id','granule_start_utc','granule_end_utc']]


def test_unselected_join_future_interval_rejected_before_join_temperature_decode(tmp_path,monkeypatch):
    target,native,area,spec=fixture(tmp_path)
    table=Path(spec['join_dir'])/'coarse_context.parquet';joined=pd.read_parquet(table)
    joined.loc[joined.sample_id.eq('extra_QA_only'),'source_granule_start']='2023-05-02T10:00:00Z'
    joined.loc[joined.sample_id.eq('extra_QA_only'),'source_granule_end']='2023-05-02T10:05:00Z'
    joined.to_parquet(table,index=False)
    manifest_path=Path(spec['join_dir'])/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['output_sha256']=old.sha(table);manifest_path.write_text(json.dumps(manifest))
    original=pd.read_parquet;thermal_reads=[]
    def read(path,columns=None,**kwargs):
        if Path(path)==table and columns is None:thermal_reads.append(str(path));raise AssertionError('Forbidden full decode')
        return original(path,columns=columns,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',read)
    with pytest.raises(ValueError,match='future, stale'):context.attach(target,spec,{area['id']:area},for_fitting=True)
    assert not thermal_reads
