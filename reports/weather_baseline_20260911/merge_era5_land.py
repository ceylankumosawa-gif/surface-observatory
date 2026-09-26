"""Hash-bound merge of native four-field ARCO and standard CDS GRIB141 SWE.

No fetching, training, interpolation, land substitution or target decoding.
The complete flag requires every requested row's same-cell/same-hour inputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS=['region_id','era5_land_valid_time_utc','era5_land_grid_latitude','era5_land_grid_longitude']
FOUR=['era5_land_skin_temperature_c','era5_land_air_temperature_c',
      'era5_land_soil_temperature_0_7cm_c','era5_land_soil_moisture_0_7cm_m3_m3']
SWE='era5_land_snow_water_equivalent_m'
PROOF=['era5_land_swe_source_sha256','era5_land_swe_param_id','era5_land_swe_expver',
       'era5_land_swe_units','era5_land_swe_step_type']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def normalized_keys(frame):
    f=frame.copy()
    if not isinstance(f.era5_land_valid_time_utc.dtype,pd.DatetimeTZDtype):
        raise ValueError('Native valid times must be explicitly timezone-aware.')
    f['era5_land_valid_time_utc']=pd.to_datetime(f.era5_land_valid_time_utc,utc=True)
    if f.era5_land_valid_time_utc.isna().any() or not f.era5_land_valid_time_utc.eq(f.era5_land_valid_time_utc.dt.floor('h')).all():
        raise ValueError('Require exact hourly valid timestamps.')
    for name in KEYS[2:]:
        x=f[name].to_numpy(float)
        if not np.isfinite(x).all() or not np.allclose(x*10,np.rint(x*10),atol=1e-6,rtol=0):
            raise ValueError('Source coordinates are not native0.1-degree cells.')
        f['_'+name]=np.rint(x*10).astype(np.int64)
    if (np.abs(f.era5_land_grid_latitude)>90).any() or (np.abs(f.era5_land_grid_longitude)>180).any():
        raise ValueError('Source longitude must be normalized to [-180,180].')
    if f.region_id.isna().any():raise ValueError('Region identity missing.')
    return f


def merge_frames(four,snow,requests,verified_source_hashes=None):
    for name in FOUR+['sample_id','era5_land_four_fields_complete']:
        if name not in four:raise ValueError('Four-field schema incomplete.')
    for name in [SWE]+PROOF:
        if name not in snow:raise ValueError('True SWE/source proof incomplete.')
    a=normalized_keys(four);s=normalized_keys(snow);r=normalized_keys(requests)
    moisture=FOUR[-1]
    a['era5_land_soil_moisture_raw_0_7cm_m3_m3']=a[moisture]
    near_zero=a[moisture].ge(-1e-12)&a[moisture].lt(0)
    a['era5_land_soil_moisture_zero_normalized']=near_zero
    a['era5_land_four_fields_complete_raw']=a.era5_land_four_fields_complete
    strict_bool=a.era5_land_four_fields_complete.map(lambda x:isinstance(x,(bool,np.bool_)))
    original_true=a.era5_land_four_fields_complete.map(lambda x:isinstance(x,(bool,np.bool_)) and bool(x))
    a.loc[near_zero,moisture]=0.
    numeric=np.isfinite(a[FOUR].to_numpy(float)).all(axis=1)
    for c in FOUR[:-1]:numeric &= a[c].between(-173.15,126.85).to_numpy()
    numeric &= a[moisture].between(0,1).to_numpy()
    # Only the recorded near-zero condition can resolve a previously false flag.
    a['era5_land_four_fields_complete']=numeric & strict_bool & (original_true|near_zero)
    s['era5_land_snow_water_equivalent_raw_m']=s[SWE]
    snow_near_zero=s[SWE].ge(-1e-12)&s[SWE].lt(0)
    s['era5_land_snow_water_equivalent_zero_normalized']=snow_near_zero
    s.loc[snow_near_zero,SWE]=0.
    decoded=['decoded_latitude','decoded_longitude','decoded_validity_date','decoded_validity_time']
    if any(c not in s for c in decoded):raise ValueError('Decoded SWE native geometry/time proof absent.')
    if not np.allclose(s.decoded_latitude,s.era5_land_grid_latitude,atol=1e-7,rtol=0) or not np.allclose(s.decoded_longitude,s.era5_land_grid_longitude,atol=1e-7,rtol=0):
        raise ValueError('Decoded SWE cells differ from requested native cells.')
    stamp=pd.to_datetime(s.decoded_validity_date.astype(int).astype(str).str.zfill(8)+s.decoded_validity_time.astype(int).astype(str).str.zfill(4),format='%Y%m%d%H%M',utc=True)
    if not stamp.eq(s.era5_land_valid_time_utc).all():raise ValueError('Decoded SWE valid time differs from requested hour.')
    keys=KEYS[:2]+['_'+x for x in KEYS[2:]]
    if a.sample_id.isna().any() or a.sample_id.duplicated().any() or s.duplicated(keys).any() or r.duplicated(keys).any():
        raise ValueError('Duplicate or missing source identities.')
    if not (pd.to_numeric(s.era5_land_swe_param_id,errors='coerce')==141).all():
        raise ValueError('SWE must be verified GRIB parameter141.')
    if not s.era5_land_swe_expver.astype(str).str.strip().str.lstrip('0').eq('1').all():
        raise ValueError('SWE must be consolidated ERA5-Land expver0001.')
    if not s.era5_land_swe_units.isin(['m','m of water equivalent']).all() or not s.era5_land_swe_step_type.eq('instant').all():
        raise ValueError('SWE units or instantaneous valid-time proof failed.')
    if not s.era5_land_swe_source_sha256.astype(str).str.fullmatch('[0-9a-f]{64}').all():
        raise ValueError('Missing source-response hash.')
    if verified_source_hashes is not None and not s.era5_land_swe_source_sha256.isin(verified_source_hashes).all():
        raise ValueError('SWE row references an unverified raw response.')
    target=r[keys].sort_values(keys).reset_index(drop=True)
    actual=s[keys].sort_values(keys).reset_index(drop=True)
    if not target.equals(actual):raise ValueError('SWE cells/hours differ from exact frozen request set.')
    acells=a[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    if not acells.equals(target):raise ValueError('Four-field cells/hours differ from native request set.')
    # Native numerical roundoff is tolerated only after proving the0.1-degree lattice.
    add=[c for c in s if c not in a and c not in keys]
    out=a.merge(s[keys+add],on=keys,how='left',validate='many_to_one',sort=False)
    if not out.sample_id.equals(a.sample_id):raise ValueError('Merge changed row identities/order.')
    finite=np.isfinite(out[FOUR+[SWE]].to_numpy(float)).all(axis=1)
    valid_snow=out[SWE].ge(0)&out[SWE].le(10)
    strict_four=out.era5_land_four_fields_complete.map(lambda x:isinstance(x,(bool,np.bool_)) and bool(x))
    out['era5_land_complete']=finite&valid_snow&strict_four
    out['era5_land_status']=np.where(out.era5_land_complete,'native_cell_available','native_cell_missing_or_invalid_no_land_substitution')
    return out.drop(columns=['_'+x for x in KEYS[2:]])


def run(arco,swe_values,swe_completion,output):
    arco=Path(arco).resolve();swe_values=Path(swe_values).resolve();swe_completion=Path(swe_completion).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new immutable merge directory.')
    ac=json.loads((arco/'completion_4fields.json').read_text())
    sc=json.loads(swe_completion.read_text())
    paths={'arco_plan':arco/'plan.json','arco_preflight':arco/'preflight_4fields.json',
           'arco_completion':arco/'completion_4fields.json','arco_values':arco/'values_4fields.parquet',
           'arco_ledger':arco/'transfer_ledger.json','arco_points':arco/'points.parquet',
           'native_cells':arco/'native_cells.parquet','swe_completion':swe_completion,'swe_values':swe_values,
           'swe_plan':swe_completion.parent/'plan.json','swe_sources':swe_completion.parent/'sources.json',
           'swe_state':swe_completion.parent/'state.json','swe_native_cells':swe_completion.parent/'native_cells.parquet'}
    hashes={k:sha(p) for k,p in paths.items()}
    if ac['status']!='four_fields_complete_SWE_pending' or ac['output_sha256']!=hashes['arco_values'] or ac['preflight_sha256']!=hashes['arco_preflight'] or ac['plan_sha256']!=hashes['arco_plan'] or ac['native_cells_sha256']!=hashes['native_cells']:
        raise ValueError('Four-field completion identity failed.')
    ap=json.loads(paths['arco_plan'].read_text())
    if ac['ledger_sha256']!=hashes['arco_ledger'] or ap['points_sha256']!=hashes['arco_points']:
        raise ValueError('Four-field ledger or original point identity failed.')
    if sc.get('status')!='complete' or sc.get('output_sha256')!=hashes['swe_values']:
        raise ValueError('SWE completion identity failed.')
    if sc['request_plan_sha256']!=hashes['swe_plan'] or sc['sources_sha256']!=hashes['swe_sources'] or sc['state_sha256']!=hashes['swe_state'] or sc['native_cells_sha256']!=hashes['swe_native_cells']:
        raise ValueError('SWE request, native cells, source metadata or transfer state changed.')
    source_files=sc.get('source_files')
    if not isinstance(source_files,list) or not source_files:
        raise ValueError('SWE completion must bind raw source response files.')
    verified_sources=set()
    for item in source_files:
        p=Path(item['path'])
        if not p.is_absolute():p=swe_completion.parent/p
        if not p.is_file() or sha(p)!=item['sha256']:
            raise ValueError('SWE raw source response checksum failed.')
        verified_sources.add(item['sha256'])
    output.mkdir(parents=True)
    plan={'delivery_format':'arco4_cds_swe_merge_v2','source_sha256':sha(__file__),'inputs':{k:{'path':str(paths[k]),'sha256':v} for k,v in hashes.items()},
          'swe_verified_source_hashes':sorted(verified_sources),
          'availability_policy':{'soil_moisture_zero_interval':'[-1e-12,0)','snow_water_equivalent_zero_interval':'[-1e-12,0)','raw_retained':True,'larger_negative_invalid':True,'secondary_mask_uses_all_five_fields':True},
          'original_full_cohort_primary_stopped':True,
          'all_five_fields_required':True,'same_cell_hour_required':True,'no_new_thermal_labels':True,
          'no_interpolation':True,'no_land_substitution':True,'retrospective_reanalysis':True,
          'release_caveat':'Historical ARCO and standard-CDS consolidated ERA5-Land; different delivery objects and packaging metadata retained. Exact binary release equality is not inferred.'}
    save(output/'plan.json',plan)
    values=merge_frames(pd.read_parquet(paths['arco_values']),pd.read_parquet(swe_values),pd.read_parquet(paths['native_cells']),verified_sources)
    values.to_parquet(output/'values.parquet',index=False)
    if any(sha(p)!=hashes[k] for k,p in paths.items()) or sha(__file__)!=plan['source_sha256']:
        raise ValueError('Bound source changed during merge.')
    result={'delivery_format':'arco4_cds_swe_merge_v2','status':'complete','rows':len(values),'available_rows':int(values.era5_land_complete.sum()),
            'plan_sha256':sha(output/'plan.json'),'output_sha256':sha(output/'values.parquet'),
            'source_sha256':sha(__file__),'arco_completion_sha256':hashes['arco_completion'],
            'swe_completion_sha256':hashes['swe_completion'],'input_hashes':hashes,
            'all_five_fields_available':bool(values.era5_land_complete.all()),
            'soil_moisture_zero_normalized_rows':int(values.era5_land_soil_moisture_zero_normalized.sum()),
            'soil_moisture_zero_epsilon_m3_m3':1e-12,
            'snow_water_equivalent_zero_normalized_rows':int(values.era5_land_snow_water_equivalent_zero_normalized.sum()),
            'snow_water_equivalent_zero_epsilon_m':1e-12,
            'original_full_cohort_primary_stopped':True,
            'no_new_thermal_labels':True,'no_training':True,
            'coarse_modeled_skin_not_observed_fine_labels':True}
    save(output/'completion.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arco',required=True);p.add_argument('--swe-values',required=True);p.add_argument('--swe-completion',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.arco,a.swe_values,a.swe_completion,a.output)))
