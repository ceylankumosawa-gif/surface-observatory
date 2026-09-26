"""Batch-union contracts using actual admission and local synthetic checkpoints.

No acquisition, model fitting, protected sources or remote API are involved.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
import pytest

from lst_pilot import multisensor_combine as combine
from lst_pilot import multisensor_pair as pair


DEPENDENCIES=("option_b_features.py","option_b_parallel.py","more_days_pair.py","option_b_cohort.py","legacy_isd.py")


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def signed_outputs(batch,parts,signature):
    completion={"signature_sha256":pair.features._sha(batch/'signature.json'),
                "input_rows":sum(len(p) for p in parts),"outputs":{}}
    for kind,frame in zip(combine.KINDS,parts):
        path=batch/(kind+'.parquet');frame.to_parquet(path,index=False)
        completion['outputs'][kind]={"path":str(path.resolve()),"sha256":pair.features._sha(path),"rows":len(frame)}
    write(batch/'completion.json',completion)
    return completion


def fixture(tmp_path,*,legacy_zero=False):
    root=tmp_path/'root';batch=tmp_path/'batch';batch.mkdir()
    area={"id":"boulder","epsg":32630,"extent_m":[500000,5700000,520000,5720000],"grid_shape":[200,200]}
    areas_path=root/'pilot/areas_resolved.json';write(areas_path,{"areas":[area]})
    script=root/'reports/night_replacement/audit_option_b_stations.py';script.parent.mkdir(parents=True);script.write_text('# synthetic audit identity\n')
    lon,lat=Transformer.from_crs(32630,4326,always_xy=True).transform(500550,5719450)
    rows=[]
    for sample,year,complete in [('fit_source',2021,True),('eval_source',2023,True),('excluded_source',2022,False)]:
        stamp=pd.Timestamp(f'{year}-06-01T12:00:00Z')
        row={name:('Dfb' if name=='climate_class' else 1.) for name in pair.features.BASE_FEATURES}
        row.update(sample_id=sample,region_id='boulder',acquisition_id=sample+'_acq',datetime_utc=stamp,
            latitude=lat,longitude=lon,epsg=32630,grid_row=5,grid_col=5,lst_c=25.,
            cohort_origin='expanded',label_product='ecostress_v2',label_source_sha256='a'*64,
            source_screen_pass=True,native_fit_support_pass=True,fresh_2023=year==2023,freshness_audit_sha256='b'*64,
            day_night='day',solar_elevation_deg=30.,air_temperature_c=20.,era5_snow_water_equivalent_m=0.,
            worldcover_land_fraction=1.,worldcover_water_fraction=0.,station_pair_available=True,
            station_age_minutes=20.,station_distance_km=10.,verified_station_report_status='exact_cached_report_verified',
            optical_earliest_source_utc=stamp-pd.Timedelta(days=5),optical_latest_source_utc=stamp-pd.Timedelta(days=2))
        if not complete or legacy_zero:row['ndvi']=np.nan
        rows.append(row)
    data=pd.DataFrame(rows)
    source=batch/'source_samples.parquet';source.parent.mkdir(parents=True,exist_ok=True);data.to_parquet(source,index=False)
    features=batch/'features/features.parquet';features.parent.mkdir();data.to_parquet(features,index=False)
    audited=batch/'station_audit/features_station_audited.parquet';audited.parent.mkdir();data.to_parquet(audited,index=False)
    write(batch/'station_audit/station_source_audit.json',{
        'reused_darwin_adapter':False,'input_sha256':pair.features._sha(features),'output_sha256':pair.features._sha(audited),
        'audit_code_sha256':pair.features._sha(script),'orchestration_sha256':pair.features._sha(pair.more_days_pair.__file__)})
    dependency_hashes={name:pair.features._sha(Path(pair.__file__).with_name(name)) for name in DEPENDENCIES}
    signature={'version':'multisensor-fine-pair-v1' if legacy_zero else pair.VERSION,
        'samples_sha256':pair.features._sha(source),'registry_sha256':'b'*64,'areas_sha256':pair.features._sha(areas_path),
        'source_sha256':'c'*64 if legacy_zero else pair.features._sha(pair.__file__),
        'dependencies_sha256':dependency_hashes,'raw_station_audit_script_sha256':pair.features._sha(script),'old_id_sources':{}}
    write(batch/'signature.json',signature)
    write(batch/'features/parallel_signature.json',{'input_sha256':pair.features._sha(source),'areas_sha256':pair.features._sha(areas_path),
        'builder_sha256':dependency_hashes['option_b_features.py'],'launcher_sha256':dependency_hashes['option_b_parallel.py'],
        'workers':1,'preserve_existing_base':False,'max_optical_scenes':16})
    feature_manifest={'input_path':str(source.resolve()),'input_sha256':pair.features._sha(source),
        'output_path':str(features.resolve()),'output_sha256':pair.features._sha(features),'rows':len(data),
        'complete_input_processed':True,'shards':[],'research_only':True}
    write(batch/'features/manifest.json',feature_manifest)
    fit,evaluation,excluded,_=pair.source_admission(data,{area['id']:area})
    signed_outputs(batch,[fit,evaluation,excluded],signature)
    return root,batch,data


def replace_output_and_hash(batch,kind,change):
    path=batch/(kind+'.parquet');frame=pd.read_parquet(path);frame=change(frame)
    frame.to_parquet(path,index=False)
    completion=json.loads((batch/'completion.json').read_text())
    completion['outputs'][kind].update(sha256=pair.features._sha(path),rows=len(frame))
    write(batch/'completion.json',completion)


def replace_signature_and_hash(batch,change):
    signature=json.loads((batch/'signature.json').read_text());change(signature);write(batch/'signature.json',signature)
    completion=json.loads((batch/'completion.json').read_text());completion['signature_sha256']=pair.features._sha(batch/'signature.json')
    write(batch/'completion.json',completion)


def test_complete_batch_uses_actual_station_checkpoint_and_admission(tmp_path):
    root,batch,data=fixture(tmp_path)
    result=combine.combine([batch],tmp_path/'combined',root)
    assert result['total_source_rows']==3
    assert {key:row['rows'] for key,row in result['outputs'].items()}=={'new_fine_fit':1,'new_fine_evaluation':1,'excluded':1}
    assert pd.read_parquet(tmp_path/'combined/new_fine_fit.parquet').sample_id.tolist()==['fit_source']
    assert pd.read_parquet(tmp_path/'combined/new_fine_evaluation.parquet').sample_id.tolist()==['eval_source']


@pytest.mark.parametrize('field',['lst_c','ndvi','air_temperature_c','native_fit_support_pass'])
def test_self_consistent_output_hash_does_not_excuse_changed_label_or_predictor(tmp_path,field):
    root,batch,_=fixture(tmp_path)
    def change(frame):
        frame.loc[frame.index[0],field]=False if field=='native_fit_support_pass' else 77.
        return frame
    replace_output_and_hash(batch,'new_fine_fit',change)
    with pytest.raises((ValueError,AssertionError)):
        combine.combine([batch],tmp_path/'rejected',root)
    assert not (tmp_path/'rejected').exists()


def test_fit_evaluation_swap_with_valid_hashes_ids_and_blank_admission_is_rejected(tmp_path):
    root,batch,_=fixture(tmp_path)
    fit=pd.read_parquet(batch/'new_fine_fit.parquet');evaluation=pd.read_parquet(batch/'new_fine_evaluation.parquet')
    replace_output_and_hash(batch,'new_fine_fit',lambda _:evaluation)
    replace_output_and_hash(batch,'new_fine_evaluation',lambda _:fit)
    with pytest.raises((ValueError,AssertionError)):
        combine.combine([batch],tmp_path/'rejected',root)
    assert not (tmp_path/'rejected').exists()


@pytest.mark.parametrize('change',['area','dependency','missing_dependency','station_script'])
def test_stale_signature_sources_rejected_even_with_updated_completion_hash(tmp_path,change):
    root,batch,_=fixture(tmp_path)
    def alter(signature):
        if change=='area':signature['areas_sha256']='d'*64
        elif change=='dependency':signature['dependencies_sha256']['option_b_features.py']='d'*64
        elif change=='missing_dependency':signature['dependencies_sha256'].pop('option_b_cohort.py')
        else:signature['raw_station_audit_script_sha256']='d'*64
    replace_signature_and_hash(batch,alter)
    with pytest.raises((ValueError,AssertionError)):
        combine.combine([batch],tmp_path/'rejected',root)
    assert not (tmp_path/'rejected').exists()


@pytest.mark.parametrize('change',['input_hash','output_hash','input_path','output_path','incomplete'])
def test_feature_manifest_must_bind_exact_batch_input_and_output(tmp_path,change):
    root,batch,_=fixture(tmp_path)
    path=batch/'features/manifest.json';manifest=json.loads(path.read_text())
    if change=='input_hash':manifest['input_sha256']='e'*64
    elif change=='output_hash':manifest['output_sha256']='e'*64
    elif change=='input_path':manifest['input_path']=str(tmp_path/'unrelated_source.parquet')
    elif change=='output_path':manifest['output_path']=str(tmp_path/'unrelated_features.parquet')
    else:manifest['complete_input_processed']=False
    write(path,manifest)
    with pytest.raises((ValueError,AssertionError)):
        combine.combine([batch],tmp_path/'rejected',root)
    assert not (tmp_path/'rejected').exists()


def test_2025_source_rejected_using_metadata_before_full_thermal_read(tmp_path,monkeypatch):
    root,batch,_=fixture(tmp_path);source=batch/'source_samples.parquet'
    data=pd.read_parquet(source);data['datetime_utc']=pd.Timestamp('2025-06-01T12:00:00Z');data.to_parquet(source,index=False)
    original=pd.read_parquet;calls=[]
    def read(path,columns=None,**kwargs):
        if Path(path)==source:
            calls.append(columns)
            assert columns is not None and 'lst_c' not in columns
        return original(path,columns=columns,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',read)
    with pytest.raises(ValueError):combine.combine([batch],tmp_path/'rejected',root)
    assert calls and all(columns is not None for columns in calls)


def test_first_engineering_v1_allowed_only_when_recomputed_admission_is_empty(tmp_path):
    root,batch,_=fixture(tmp_path,legacy_zero=True)
    result=combine.combine([batch],tmp_path/'combined',root)
    assert result['outputs']['new_fine_fit']['rows']==result['outputs']['new_fine_evaluation']['rows']==0
    assert result['outputs']['excluded']['rows']==3


def test_legacy_adapter_cannot_supply_an_admitted_row(tmp_path):
    root,batch,_=fixture(tmp_path)
    replace_signature_and_hash(batch,lambda s:s.update(version='multisensor-fine-pair-v1',source_sha256='c'*64))
    with pytest.raises(ValueError):combine.combine([batch],tmp_path/'rejected',root)
