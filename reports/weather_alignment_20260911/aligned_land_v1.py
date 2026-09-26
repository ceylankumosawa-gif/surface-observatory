"""Versioned, source-bound retrospective interpolation of instantaneous ERA5-Land.

Inventory reads metadata only and never uses HTTP. Extraction requires a frozen
protocol hash. Existing sources, caches and target air temperature are untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

OLD = Path(__file__).resolve().parents[1] / 'weather_baseline_20260911'
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value
b = module('alignment_frozen_arco_decoder', OLD/'era5_land.py')
v3 = module('alignment_frozen_arco_v3', OLD/'era5_land_v3.py')
IDENTITY = b.COLUMNS
KEYS = ['region_id', 'era5_land_valid_time_utc', 'era5_land_grid_latitude', 'era5_land_grid_longitude']
GROUPS = ('skin', 'air', 'soil_temperature', 'soil_water')
FOUR = [b.VARIABLES[g]['output'] for g in GROUPS]
SWE = 'era5_land_snow_water_equivalent_m'
FIELDS = FOUR + [SWE]
FORMAT = 'bracketed_instantaneous_land_v1'
DOC = 'https://confluence.ecmwf.int/pages/viewpage.action?pageId=177471794'

def require(value, message):
    if not value: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text())
def binding(path): return {'path': str(Path(path).resolve()), 'sha256': b.sha(path)}
def check(item):
    require(b.sha(item['path']) == item['sha256'], 'Bound source checksum changed')
    return Path(item['path'])

def time_brackets(frame, period='fit'):
    """Requested labels stay within period; only their exact ceil may cross year."""
    require(period in ('fit','evaluation'),'Unknown source period')
    out = b.prepare_points(frame, period)[IDENTITY].copy()
    out['floor_utc'] = out.datetime_utc.dt.floor('h')
    out['ceil_utc'] = out.datetime_utc.dt.ceil('h')
    out['interpolation_alpha'] = (out.datetime_utc-out.floor_utc).dt.total_seconds()/3600
    out['later_valid_time_used'] = out.interpolation_alpha.gt(0)
    upper='2023-01-01T00:00:00Z' if period=='fit' else '2025-01-01T00:00:00Z'
    require(out.ceil_utc.le(pd.Timestamp(upper)).all(), 'Endpoint exceeds authorized covariate boundary')
    return out

def endpoint_requests(points):
    frames = []
    for name in ('floor', 'ceil'):
        a = points.copy()
        if name == 'ceil': a = a.loc[a.later_valid_time_used]
        a['era5_land_valid_time_utc'] = a[name+'_utc']
        frames.append(a[['region_id','era5_land_valid_time_utc','latitude','longitude']])
    return pd.concat(frames,ignore_index=True).drop_duplicates().reset_index(drop=True)

class CachedHTTP:
    """Read-only verified parent objects; cannot make a network request."""
    def __init__(self, directory, ledger_sha):
        self.root = Path(directory)
        require(b.sha(self.root/'transfer_ledger.json') == ledger_sha, 'ARCO ledger changed')
        self.ledger = read(self.root/'transfer_ledger.json'); self.used = {}
    def item(self, url):
        record = self.ledger['objects'].get(url)
        if not record or record.get('status') != 'complete': return None
        path = self.root/'http_cache'/hashlib.sha256(url.encode()).hexdigest()
        require(path.is_file() and b.sha(path)==record['sha256'], 'Cached ARCO object missing/changed')
        return dict(binding(path), bytes=path.stat().st_size, url=url)
    def get(self, url):
        item = self.item(url)
        if item is None: raise FileNotFoundError('Object absent from read-only cache; no request made')
        self.used[url] = item
        return Path(item['path']).read_bytes()

def source_positions(client, requests):
    grid = None; results = {}; native = None
    for group in GROUPS:
        reader = b.ZarrReader(client, group)
        name, attrs = v3.checked_variable(reader, group)
        lat, la = reader.coordinate('latitude'); lon, lo = reader.coordinate('longitude')
        tv, ta = reader.coordinate('time'); times = b.decode_times(tv,ta)
        require(la.get('units') in ('degrees_north','degree_north') and lo.get('units') in ('degrees_east','degree_east'), 'Unverified native coordinate units')
        require(not times.has_duplicates and times.is_monotonic_increasing, 'Ambiguous native valid-time axis')
        if grid is None: grid = (lat,lon)
        else: require(all(np.array_equal(x,y) for x,y in zip(grid,(lat,lon))), 'Native spatial axes differ')
        ila=b.nearest_indexes(lat,requests.latitude); ilo=b.nearest_indexes(lon,requests.longitude,longitude=True)
        wanted=pd.DatetimeIndex(requests.era5_land_valid_time_utc); it=times.get_indexer(wanted)
        require((it>=0).all() and np.array_equal(times[it].asi8,wanted.asi8), 'Required exact endpoint hour absent')
        positions=np.column_stack([it,ila,ilo]); meta,_=reader.array(name)
        chunks=sorted(set(map(tuple,positions//np.asarray(meta['chunks']))))
        urls=[reader.url+'/'+name+'/'+meta.get('dimension_separator','.').join(map(str,c)) for c in chunks]
        results[group]={'reader':reader,'name':name,'attributes':attrs,'positions':positions,'urls':urls,
                        'array':meta,'metadata_sha256':reader.metadata_sha}
        if native is None:
            native=requests.copy();native[KEYS[2]]=lat[ila];native[KEYS[3]]=lon[ilo]
    return native, results

def swe_cached_coverage(cells, plan):
    """Metadata request containment only; extraction revalidates actual GRIB."""
    owners=np.full(len(cells), '', dtype=object)
    for job in plan['jobs']:
        req=job['request']; north,west,south,east=req['area']
        t=cells[KEYS[1]]
        mask=(cells.region_id.eq(job['region_id']) & t.dt.year.eq(int(req['year'][0])) &
              t.dt.month.eq(int(req['month'][0])) & t.dt.day.isin([int(x) for x in req['day']]) &
              t.dt.strftime('%H:%M').isin(req['time']) & cells[KEYS[2]].between(south-1e-7,north+1e-7) &
              cells[KEYS[3]].between(west-1e-7,east+1e-7))
        require(not np.any(mask & (owners!='')), 'Overlapping SWE source jobs require explicit deduplication')
        owners[mask]=job['id']
    return owners

def inventory(input_path, arco, swe, output, period='fit'):
    output=Path(output).resolve();arco=Path(arco).resolve();swe=Path(swe).resolve()
    require(not output.exists(), 'Use a new immutable inventory directory')
    before=b.sha(input_path)
    points=time_brackets(pd.read_parquet(input_path,columns=IDENTITY),period)
    require(b.sha(input_path)==before, 'Input changed while reading metadata')
    ac=read(arco/'completion_4fields.json'); ap=read(arco/'plan.json');sc=read(swe/'completion.json');sp=read(swe/'plan.json')
    require(ac['status']=='four_fields_complete_SWE_pending' and sc['status']=='complete', 'Parent source incomplete')
    require(b.sha(arco/'plan.json')==ac['plan_sha256'] and b.sha(swe/'plan.json')==sc['request_plan_sha256'], 'Parent plan changed')
    require(ap['period']==period and sp['period']==period, 'Parent source period differs')
    require(b.sha(OLD/'era5_land_v3.py')==ac['source_sha256'] and b.sha(OLD/'era5_land.py')==ac['base_source_sha256'], 'Frozen ARCO decoder changed')
    require(b.sha(OLD/'cds_swe.py')==sc['source_sha256'], 'Frozen SWE decoder changed')
    client=CachedHTTP(arco,ac['ledger_sha256'])
    native,info=source_positions(client,endpoint_requests(points))
    cells=native[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    owners=swe_cached_coverage(cells,sp)
    state=read(swe/'state.json');require(b.sha(swe/'state.json')==sc['state_sha256'], 'SWE parent state changed')
    source_hashes={x['sha256'] for x in sc['source_files']}
    for jobid in sorted(set(owners)-{''}):
        raw=state['jobs'][jobid]['raw']
        require(state['jobs'][jobid]['status']=='downloaded' and raw['sha256'] in source_hashes and b.sha(raw['path'])==raw['sha256'], 'SWE raw cache proof failed')
    object_rows=[]
    for group,data in info.items():
        for url in data['urls']:
            item=client.item(url)
            object_rows.append({'group':group,'url':url,'cached':item is not None,'cache':item,
                                'decoded_bytes':int(np.prod(data['array']['chunks'])*np.dtype(data['array']['dtype']).itemsize)})
    output.mkdir(parents=True)
    points.to_parquet(output/'points.parquet',index=False)
    native.to_parquet(output/'endpoint_requests.parquet',index=False)
    cells.to_parquet(output/'native_cells.parquet',index=False)
    cells.loc[owners==''].to_parquet(output/'missing_swe_native_cells.parquet',index=False)
    cells.assign(cached_swe_job_id=owners).to_parquet(output/'swe_coverage.parquet',index=False)
    paths={'input':binding(input_path),'arco_plan':binding(arco/'plan.json'),'arco_completion':binding(arco/'completion_4fields.json'),
           'arco_ledger':binding(arco/'transfer_ledger.json'),'swe_plan':binding(swe/'plan.json'),'swe_completion':binding(swe/'completion.json'),
           'swe_state':binding(swe/'state.json'),'decoder':binding(OLD/'era5_land.py'),'arco_v3':binding(OLD/'era5_land_v3.py'),
           'swe_decoder':binding(OLD/'cds_swe.py'),'batched_swe_decoder':binding(Path(__file__).with_name('cds_swe_batched_v1.py')),
           'batched_swe_decoder_v2':binding(Path(__file__).with_name('cds_swe_batched_v2.py'))}
    plan={'format':FORMAT,'source_sha256':b.sha(__file__),'inputs':paths,'arco_directory':str(arco),'swe_directory':str(swe),
          'outputs':{p.name:binding(p) for p in output.glob('*.parquet')},'metadata_objects':client.used,'arco_objects':object_rows,
          'period':period,'target_years':([2021,2022] if period=='fit' else [2021,2022,2023,2024]),
          'weather_year_edge':'Only exact required ceil(target) may enter Jan1 2023 or Jan1 2025 00:00; never corresponding thermal labels.',
          'retrospective_interpolation':True,'no_nearest_land_fill':True,'normalization_epsilon':1e-12,
          'target_air_A_unchanged':True,'documentation':DOC,'requests_made':0,
          'field_units':{field:('degC' if field.endswith('_c') else ('m water equivalent' if field==SWE else 'm3/m3')) for field in FIELDS},
          'raw_endpoints_definition':'Source physical values after explicit K-to-degC conversion, before tiny-negative normalization; raw source objects remain hash-bound.',
          'limits':{'transfer_bytes':64*1024**2,'object_bytes':64*1024**2,'requests':100,'wall_seconds':1800,'decoded_chunk_bytes':64*1024**2}}
    b.save(output/'plan.json',plan)
    summary={'status':'metadata_inventory_complete_no_values_extracted','target_rows':len(points),'target_hours':points.floor_utc.nunique(),
             'exact_hour_rows':int((~points.later_valid_time_used).sum()),'endpoint_native_cell_hours':len(cells),
             'endpoint_hours':cells[KEYS[1]].nunique(),'year_edge_endpoints':int(cells[KEYS[1]].dt.year.eq(2023 if period=='fit' else 2025).sum()),
             'arco_chunks':len(object_rows),'arco_cached_chunks':sum(x['cached'] for x in object_rows),
             'arco_cached_compressed_bytes':sum(x['cache']['bytes'] for x in object_rows if x['cached']),
             'arco_missing_chunks':sum(not x['cached'] for x in object_rows),
             'arco_decoded_bytes_total':sum(x['decoded_bytes'] for x in object_rows),
             'swe_cached_native_cell_hours':int((owners!='').sum()),'swe_missing_native_cell_hours':int((owners=='').sum()),
             'swe_missing_region_months':int(cells.loc[owners==''].assign(month=cells.loc[owners=='',KEYS[1]].dt.strftime('%Y-%m'))[['region_id','month']].drop_duplicates().shape[0]),
             'new_HTTP_requests':0,'new_download_bytes':0,'plan_sha256':b.sha(output/'plan.json')}
    b.save(output/'inventory.json',summary)
    return summary

def normalized_endpoint(frame):
    out=frame.copy(); complete=np.ones(len(out),dtype=bool)
    for field in FIELDS:
        x=pd.to_numeric(out[field],errors='coerce').to_numpy(float,copy=True)
        out[field+'_raw']=x.copy()
        repair=np.zeros(len(x),dtype=bool)
        if field in (FOUR[-1],SWE): repair=(x>=-1e-12)&(x<0);x[repair]=0
        out[field]=x;out[field+'_zero_normalized']=repair
        lo,hi=(-173.15,126.85) if field.endswith('_c') else ((0,10) if field==SWE else (0,1))
        complete &= np.isfinite(x)&(x>=lo)&(x<=hi)
    out['endpoint_complete']=complete
    return out

def interpolate(points, endpoints, period='fit'):
    """All-five common mask; no numerical gap filling across bad endpoints."""
    expected=time_brackets(points,period)
    for column in ('floor_utc','ceil_utc','interpolation_alpha','later_valid_time_used'):
        require(points[column].reset_index(drop=True).equals(expected[column]), 'Target bracket/alpha was altered')
    require(isinstance(endpoints[KEYS[1]].dtype,pd.DatetimeTZDtype), 'Endpoint time must be explicitly timezone-aware')
    require(endpoints[KEYS[1]].eq(endpoints[KEYS[1]].dt.floor('h')).all(), 'Endpoint times must be exact native hours')
    require(not endpoints.duplicated(KEYS).any(), 'Duplicate native endpoints')
    native=points.copy()
    require(all(c in native for c in KEYS[2:]), 'Target native cell unavailable')
    normalized=normalized_endpoint(endpoints)
    key=KEYS[:1]+KEYS[2:]
    for side in ('floor','ceil'):
        right=normalized.rename(columns={KEYS[1]:side+'_utc',**{c:side+'__'+c for c in normalized if c not in KEYS}})
        native=native.merge(right,on=key+[side+'_utc'],how='left',validate='many_to_one',sort=False)
    require(native.sample_id.tolist()==points.sample_id.tolist(), 'Alignment changed target order')
    alpha=native.interpolation_alpha.to_numpy(float)
    valid=native['floor__endpoint_complete'].eq(True).to_numpy() & ((alpha==0)|native['ceil__endpoint_complete'].eq(True).to_numpy())
    for field in FIELDS:
        lo=native['floor__'+field].to_numpy(float);hi=native['ceil__'+field].to_numpy(float)
        # Masked assignment avoids NaN*0 on exact-hour targets.
        value=lo.copy();between=alpha>0;value[between]=lo[between]+alpha[between]*(hi[between]-lo[between])
        value[~valid]=np.nan;native[field]=value
    native['era5_land_complete']=valid
    native['era5_land_status']=np.where(valid,'retrospective_native_alignment_available','required_endpoint_missing_or_invalid')
    native['era5_land_valid_time_utc']=native.datetime_utc
    native['retrospective_interpolation']=True
    return native

def extract(directory, protocol, protocol_sha, extra_swe=None):
    """Use cached ARCO only. Missing SWE is supplied by a separately frozen CDS job."""
    directory=Path(directory);plan=read(directory/'plan.json')
    require(b.sha(protocol)==protocol_sha, 'Explicit frozen study protocol checksum required')
    require(plan['source_sha256']==b.sha(__file__), 'Alignment source changed after inventory')
    require(not (directory/'completion.json').exists(), 'Completed aligned delivery is immutable')
    for item in [*plan['inputs'].values(),*plan['outputs'].values()]:check(item)
    period=plan['period']
    points=pd.read_parquet(directory/'points.parquet');time_brackets(points,period)
    requests=pd.read_parquet(directory/'endpoint_requests.parquet')
    cache=CachedHTTP(plan['arco_directory'],plan['inputs']['arco_ledger']['sha256'])
    native,info=source_positions(cache,requests[['region_id',KEYS[1],'latitude','longitude']])
    for group,data in info.items():
        value=data['reader'].read(data['name'],data['positions'])
        if 'GRIB_missingValue' in data['attributes']:
            value[np.isclose(value,float(data['attributes']['GRIB_missingValue']),rtol=1e-6,atol=0)]=np.nan
        native[b.VARIABLES[group]['output']]=value+b.VARIABLES[group]['offset']
    four=native[KEYS+FOUR].drop_duplicates(KEYS)
    cds=module('alignment_frozen_cds_decoder',OLD/'cds_swe.py')
    coverage=pd.read_parquet(directory/'swe_coverage.parquet');swe_dir=Path(plan['swe_directory'])
    sp=read(swe_dir/'plan.json');state=read(swe_dir/'state.json');snow=[]
    for job in sp['jobs']:
        sub=coverage.loc[coverage.cached_swe_job_id.eq(job['id']),KEYS]
        if len(sub):
            raw=state['jobs'][job['id']]['raw'];require(b.sha(raw['path'])==raw['sha256'],'Cached SWE response changed')
            snow.append(cds.parse_job(raw['path'],job,sub)[0])
    extra_bindings={}
    missing=coverage.loc[coverage.cached_swe_job_id.eq(''),KEYS]
    if len(missing):
        require(extra_swe is not None,'Required SWE endpoints pending; no interpolation or fill permitted')
        ext=Path(extra_swe);done=read(ext/'completion.json');ep=read(ext/'plan.json')
        extra_decoder=cds
        batch_format=ep.get('format')
        if batch_format in ('cds_swe_pilot_year_v1','cds_swe_pilot_year_v2'):
            source_key='batched_swe_decoder_v2' if batch_format.endswith('_v2') else 'batched_swe_decoder'
            extra_decoder=module('alignment_batched_swe_decoder',check(plan['inputs'][source_key]))
            check(ep['parent_source'])
        require(done['status']=='complete' and done['source_sha256']==b.sha(extra_decoder.__file__), 'New SWE delivery incomplete/source changed')
        require(ep['period']==period and ep['new_thermal_labels_read'] is False and ep['reserved_2025_opened'] is False, 'Unexpected extra weather scope')
        for filename,field in [('plan.json','request_plan_sha256'),('state.json','state_sha256'),('native_cells.parquet','native_cells_sha256'),('sources.json','sources_sha256'),('values.parquet','output_sha256')]:
            require(b.sha(ext/filename)==done[field], 'New SWE delivery changed');extra_bindings[filename]=binding(ext/filename)
        req=pd.read_parquet(ext/'native_cells.parquet')
        allowed=[2021,2022] if period=='fit' else [2021,2022,2023,2024]
        edge=pd.Timestamp('2023-01-01T00:00Z' if period=='fit' else '2025-01-01T00:00Z')
        require((req[KEYS[1]].dt.year.isin(allowed)|req[KEYS[1]].eq(edge)).all(), 'Unapproved extra endpoint years')
        membership=missing.merge(req,on=KEYS,how='left',validate='one_to_one',indicator=True)
        require(membership['_merge'].eq('both').all(), 'New SWE request set omits required endpoints')
        # A separately bound fitting-weather screen may share the rectangular
        # request. Only this delivery's exact missing endpoint keys are emitted.
        es=read(ext/'state.json')
        for job in ep['jobs']:
            if batch_format in ('cds_swe_pilot_year_v1','cds_swe_pilot_year_v2'):sub=extra_decoder.job_keys(missing,job)
            else:sub=missing[(missing.region_id==job['region_id'])&(missing[KEYS[1]].dt.strftime('%Y-%m')==job['id'][-7:])]
            if not len(sub):continue
            raw=es['jobs'][job['id']]['raw'];require(b.sha(raw['path'])==raw['sha256'],'New SWE raw response changed')
            snow.append(extra_decoder.parse_job(raw['path'],job,sub)[0]);extra_bindings[raw['path']]=binding(raw['path'])
    snow=pd.concat(snow,ignore_index=True)
    endpoint=four.merge(snow,on=KEYS,how='left',validate='one_to_one',sort=False)
    require(len(endpoint)==len(coverage) and endpoint.era5_land_swe_source_sha256.notna().all(),'Missing SWE endpoint proof')
    # Coordinate association comes from the exact unchanged native grid, not the values.
    coordinates=native[['region_id','latitude','longitude']+KEYS[2:]].drop_duplicates()
    target=points.merge(coordinates,on=['region_id','latitude','longitude'],how='left',validate='many_to_one',sort=False)
    result=interpolate(target,endpoint,period)
    normalized_endpoint(endpoint).to_parquet(directory/'native_endpoints.parquet',index=False)
    result.to_parquet(directory/'values.parquet',index=False)
    for item in [*plan['inputs'].values(),*plan['outputs'].values(),*extra_bindings.values()]:check(item)
    done={'format':FORMAT,'status':'complete','rows':len(result),'available_rows':int(result.era5_land_complete.sum()),
          'plan_sha256':b.sha(directory/'plan.json'),'source_sha256':b.sha(__file__),'protocol':binding(protocol),
          'output_sha256':b.sha(directory/'values.parquet'),'native_endpoints_sha256':b.sha(directory/'native_endpoints.parquet'),
          'cached_arco_used':cache.used,'extra_swe':extra_bindings,'new_HTTP_requests':0,'new_download_bytes':0,
          'target_air_A_unchanged':True,'thermal_labels_read':False,'retrospective_later_hour_used':True,'no_2025_target_rows':True,
          'weather_2025_endpoint_rows':int(endpoint[KEYS[1]].dt.year.eq(2025).sum()),'period':period}
    b.save(directory/'completion.json',done);return done

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('inventory')
    for name in ('input','arco','swe','output'):a.add_argument('--'+name,required=True)
    a.add_argument('--period',choices=['fit','evaluation'],default='fit')
    a=s.add_parser('extract')
    for name in ('directory','protocol','protocol-sha'):a.add_argument('--'+name,required=True)
    a.add_argument('--extra-swe')
    args=vars(p.parse_args());command=args.pop('command')
    if command=='inventory':args['input_path']=args.pop('input')
    print(json.dumps((inventory if command=='inventory' else extract)(**args)))
if __name__=='__main__':main()
