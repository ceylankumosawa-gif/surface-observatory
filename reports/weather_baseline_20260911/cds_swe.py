"""Bounded standard-CDS ERA5-Land SWE retrieval; native weather keys only.

The ARCO snow-depth product is GRIB3066, not water equivalent. This retrieves
GRIB141 from the standard catalogue, retaining original native cell/hour keys.
No thermal labels are read. Run with the project Python; auxiliary packages are
loaded from an isolated research runtime, never installed in the web runtime.
"""
from __future__ import annotations
import argparse
import calendar
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
import warnings

RUNTIME = Path('/opt/lst-pilot/runs/weather_baseline_20260911_v1/cds_runtime/lib')
for p in RUNTIME.glob('python*/site-packages'):
    sys.path.append(str(p))
import numpy as np
import pandas as pd
import requests
from ecmwf.datastores import Client
import eccodes as ec

API = 'https://cds.climate.copernicus.eu/api'
DATASET = 'reanalysis-era5-land'
KEYS = ['region_id', 'era5_land_valid_time_utc', 'era5_land_grid_latitude', 'era5_land_grid_longitude']
VALUE = 'era5_land_snow_water_equivalent_m'
DOC = 'https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,data):
    p=Path(path);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');t.replace(p)

def read(path):return json.loads(Path(path).read_text())
def log(**x):print(json.dumps(x),flush=True)
class SafeValidationError(ValueError):pass
def require(x,msg):
    if not x:raise SafeValidationError(msg)

def make_plan(native,output,period):
    out=Path(output);require(not out.exists(),'Use a new immutable output directory')
    d=pd.read_parquet(native,columns=KEYS)
    require(not d.duplicated(KEYS).any() and d[KEYS].notna().all().all(),'Invalid native keys')
    d[KEYS[1]]=pd.to_datetime(d[KEYS[1]],utc=True)
    years=[2021,2022] if period=='fit' else [2021,2022,2023,2024]
    require(d[KEYS[1]].dt.year.isin(years).all(),'Unapproved requested year')
    require((d[KEYS[1]]==d[KEYS[1]].dt.floor('h')).all(),'Non-hour native time')
    require(np.allclose(d[KEYS[2:]].to_numpy()*10,np.rint(d[KEYS[2:]].to_numpy()*10),atol=1e-7,rtol=0),'Non-native grid')
    require(d[KEYS[2]].between(-90,90).all() and d[KEYS[3]].between(-180,180).all(),'Native coordinates must use normalized longitude')
    d['_month']=d[KEYS[1]].dt.strftime('%Y-%m')
    jobs=[]
    for (region,ym),g in d.groupby(['region_id','_month'],sort=True):
        lat=g[KEYS[2]].round(1);lon=g[KEYS[3]].round(1)
        north,south,west,east=float(lat.max()),float(lat.min()),float(lon.min()),float(lon.max())
        require(east-west<3 and north-south<3,'Unexpected large rectangle')
        # At least a 2x2 grid; exact selected cells are retained when decoding.
        if north==south:north=round(north+.1,1)
        if east==west:east=round(east+.1,1)
        days=sorted(g[KEYS[1]].dt.strftime('%d').unique().tolist())
        hours=sorted(g[KEYS[1]].dt.strftime('%H:%M').unique().tolist())
        req={'variable':['snow_depth_water_equivalent'], 'year':[ym[:4]],'month':[ym[5:]],'day':days,'time':hours,
             'area':[north,west,south,east],'data_format':'grib','download_format':'unarchived'}
        # One real cross-delivery check, declared before any source values.
        if not jobs:req['variable']+=['skin_temperature','2m_temperature']
        ncell=(round((north-south)*10)+1)*(round((east-west)*10)+1)
        jobs.append({'id':f'{region}_{ym}','region_id':region,'request':req,'wanted_native_rows':len(g),
                     'returned_fields':len(days)*len(hours)*len(req['variable']),
                     'grid_cells':ncell,'weather_extras_from_rectangular_day_hour_request':True})
    require(len(jobs)<=200,'Too many CDS jobs')
    out.mkdir(parents=True);d.drop(columns='_month').to_parquet(out/'native_cells.parquet',index=False)
    plan={'dataset':DATASET,'period':period,'years':years,'native_input_path':str(Path(native).resolve()),
          'native_input_sha256':sha(native),'native_cells_sha256':sha(out/'native_cells.parquet'),
          'source_sha256':sha(__file__),'documentation':DOC,'rows':len(d),'jobs':jobs,
          'selection':'exact native 0.1 degree cell and floor UTC hour; no interpolation/nearest land substitution',
          'limits':{'api_requests':1500,'download_bytes':64*1024**2,'object_bytes':8*1024**2,'max_active_jobs':3,'wall_seconds':1800},
          'new_thermal_labels_read':False,'reserved_2025_opened':False,
          'first_job_extra_variables':'skt235/t2m167 retained separately to compare ARCO and CDS delivery',
          'estimated_unpacked_scalar_bytes':sum(j['returned_fields']*j['grid_cells']*8 for j in jobs)}
    save(out/'plan.json',plan);log(plan='ready',jobs=len(jobs),rows=len(d),scalar_bytes=plan['estimated_unpacked_scalar_bytes'])

def continue_plan(previous,output):
    previous=Path(previous);out=Path(output);before=read(previous/'plan.json');state=read(previous/'state.json')
    require(state['request_plan_sha256']==sha(previous/'plan.json'),'Previous request plan changed')
    require(sha(previous/'native_cells.parquet')==before['native_cells_sha256'],'Previous native keys changed')
    require(not any(x['status']=='submitting' for x in state['jobs'].values()),'Ambiguous previous submission')
    make_plan(previous/'native_cells.parquet',out,before['period'])
    after=read(out/'plan.json');require(before['jobs']==after['jobs'],'Continuation changes requested jobs')
    for entry in state['jobs'].values():
        if entry['status']=='downloaded':require(sha(entry['raw']['path'])==entry['raw']['sha256'],'Previous downloaded bytes changed')
    elapsed=time.time()-state['started_epoch']
    after['continuation']={'previous_directory':str(previous.resolve()),'previous_plan_sha256':sha(previous/'plan.json'),
                           'previous_state_sha256':sha(previous/'state.json'),'previous_source_sha256':before['source_sha256'],
                           'previous_elapsed_seconds':elapsed,'reason':'Versioned source-validation continuation; exact same requests, carried jobs/charges, raw negative SWE preserved for explicit merger domain policy.'}
    save(out/'plan.json',after)
    state['request_plan_sha256']=sha(out/'plan.json');state['prior_elapsed_seconds']=state.get('prior_elapsed_seconds',0)+elapsed
    state['started_epoch']=time.time();save(out/'state.json',state)
    log(continued_jobs=len(state['jobs']),carried_api_requests=state['api_requests'],carried_download_bytes=state['download_charged_bytes'])

class BoundSession(requests.Session):
    def __init__(self,out,plan,state):
        super().__init__();self.trust_env=False;self.out=out;self.plan=plan;self.state=state
    def request(self,method,url,**kwargs):
        u=urlparse(url)
        require(u.scheme=='https' and u.netloc=='cds.climate.copernicus.eu' and u.path.startswith('/api/'),'Unexpected authenticated endpoint')
        require(self.state['api_requests']<self.plan['limits']['api_requests'],'API request budget reached')
        require(time.time()-self.state['started_epoch']<self.plan['limits']['wall_seconds'],'Continuation required after wall-time ceiling')
        self.state['api_requests']+=1;save(self.out/'state.json',self.state)
        kwargs['allow_redirects']=False
        return super().request(method,url,**kwargs)

def client(out,plan,state):
    credential=Path('/var/lib/lst-data/copernicus/credential.json')
    require(not credential.stat().st_mode&0o077,'Credentials must be private')
    token=read(credential)['token']
    logging.disable(logging.CRITICAL);warnings.filterwarnings('ignore')
    return Client(url=API,key=token,session=BoundSession(out,plan,state),timeout=45,maximum_tries=1,
                  progress=False,cleanup=False,log_callback=lambda *a,**k:None)

def download(result,out,plan,state,job):
    size=result.content_length;u=urlparse(result.location)
    require(u.scheme=='https' and (u.hostname.endswith('.ecmwf.int') or u.hostname.endswith('.climate.copernicus.eu')),'Unrecognized CDS result host')
    require(0<size<=plan['limits']['object_bytes'],'Result exceeds per-object budget')
    require(state['download_charged_bytes']+size<=plan['limits']['download_bytes'],'Download budget reached')
    state['download_charged_bytes']+=size;save(out/'state.json',state)
    target=out/(job['id']+'.grib');temp=target.with_suffix('.part')
    s=requests.Session();s.trust_env=False
    with s.get(result.location,stream=True,allow_redirects=False,timeout=(15,60)) as r:
        require(r.status_code==200,'CDS result download failed')
        total=0
        with temp.open('wb') as f:
            for b in r.iter_content(65536):
                total+=len(b);require(total<=size,'Result exceeded declared size');f.write(b)
        require(total==size,'Incomplete result download')
        proof={'bytes':total,'sha256':sha(temp),'host':u.hostname,'etag':r.headers.get('ETag'),'last_modified':r.headers.get('Last-Modified')}
    temp.replace(target);proof['path']=str(target.resolve());return proof

def step(output,submit_limit=None):
    out=Path(output);plan=read(out/'plan.json')
    require(sha(__file__)==plan['source_sha256'],'Source changed after request plan')
    state=read(out/'state.json') if (out/'state.json').exists() else {'request_plan_sha256':sha(out/'plan.json'),'started_epoch':time.time(),'api_requests':0,'download_charged_bytes':0,'jobs':{}}
    require(state['request_plan_sha256']==sha(out/'plan.json'),'Request plan changed')
    c=client(out,plan,state)
    for job in plan['jobs']:
        entry=state['jobs'].get(job['id'])
        if not entry or entry['status']=='downloaded':continue
        if entry['status']=='submitting':raise RuntimeError('Ambiguous submit; reconcile job before resubmitting')
        r=c.get_remote(entry['request_id']);status=r.status
        entry['status']=status;save(out/'state.json',state)
        require(status in ('accepted','running','successful'),'CDS job failed or rejected')
        if status=='successful':
            entry['raw']=download(r.get_results(),out,plan,state,job)
            entry['status']='downloaded';save(out/'state.json',state)
            log(downloaded=job['id'],bytes=entry['raw']['bytes'])
    active=sum(j['status']!='downloaded' for j in state['jobs'].values())
    n=0
    for job in plan['jobs']:
        if active>=plan['limits']['max_active_jobs'] or (submit_limit is not None and n>=submit_limit):break
        if job['id'] in state['jobs']:continue
        state['jobs'][job['id']]={'status':'submitting'};save(out/'state.json',state)
        r=c.submit(DATASET,job['request'])
        state['jobs'][job['id']]={'status':'accepted','request_id':r.request_id};save(out/'state.json',state)
        active+=1;n+=1;log(submitted=job['id'])
    done=sum(j['status']=='downloaded' for j in state['jobs'].values())
    log(downloaded_jobs=done,total_jobs=len(plan['jobs']),active=active,api_requests=state['api_requests'],download_bytes=state['download_charged_bytes'])
    return done==len(plan['jobs'])

def parse_job(path,job,keys):
    records=[];cross=[];messages=[];seen=set()
    expected={(int(t.strftime('%Y%m%d')),int(t.strftime('%H%M'))) for t in keys[KEYS[1]]}
    req=job['request'];year=int(req['year'][0]);month=int(req['month'][0])
    require(year in [2021,2022,2023,2024],'Unapproved requested weather year')
    params={'snow_depth_water_equivalent':141,'skin_temperature':235,'2m_temperature':167}
    requested={(params[v],year*10000+month*100+int(d),int(h.replace(':','')))
               for v in req['variable'] for d in req['day'] for h in req['time']
               if 1<=int(d)<=calendar.monthrange(year,month)[1]}
    with Path(path).open('rb') as f:
        while (h:=ec.codes_grib_new_from_file(f)) is not None:
            try:
                names=['paramId','shortName','units','expver','stepType','dataDate','dataTime','step','validityDate','validityTime','gridType','iDirectionIncrementInDegrees','jDirectionIncrementInDegrees','numberOfPoints','bitsPerValue','referenceValue','binaryScaleFactor','decimalScaleFactor']
                p={k:ec.codes_get(h,k) for k in names}
                require(p['paramId'] in ([141,235,167] if len(job['request']['variable'])==3 else [141]),'Unplanned GRIB parameter')
                require(int(p['expver'])==1 and p['stepType']=='instant','Wrong release or step type')
                require(p['gridType']=='regular_ll' and abs(p['iDirectionIncrementInDegrees']-.1)<1e-7 and abs(p['jDirectionIncrementInDegrees']-.1)<1e-7,'Wrong native grid')
                require(p['units'] in (['m','m of water equivalent'] if p['paramId']==141 else ['K']),'Wrong GRIB units')
                valid=(int(p['validityDate']),int(p['validityTime']))
                ident=(p['paramId'],*valid);require(ident in requested,'Unrequested GRIB identity')
                require(ident not in seen,'Duplicate GRIB field');seen.add(ident)
                stamp=pd.to_datetime(f'{valid[0]:08d}{valid[1]:04d}',format='%Y%m%d%H%M',utc=True)
                require(stamp.year in [2021,2022,2023,2024],'Unapproved weather year')
                lat=ec.codes_get_array(h,'latitudes');lon=ec.codes_get_array(h,'longitudes');vals=ec.codes_get_values(h)
                if ec.codes_get(h,'bitmapPresent'):
                    bitmap=ec.codes_get_array(h,'bitmap').astype(bool);vals=np.where(bitmap,vals,np.nan)
                cells={(round(float(a),1),round((float(b)+180)%360-180,1)):i for i,(a,b) in enumerate(zip(lat,lon))}
                require(len(cells)==len(vals),'Duplicate native grid coordinates')
                messages.append(p)
                if valid not in expected:continue
                for _,key in keys.loc[keys[KEYS[1]]==stamp].iterrows():
                    ix=cells.get((round(key[KEYS[2]],1),round(key[KEYS[3]],1)))
                    require(ix is not None,'Requested exact native cell absent')
                    require(abs(float(lat[ix])-key[KEYS[2]])<1e-7 and abs((float(lon[ix])+180)%360-180-key[KEYS[3]])<1e-7,'Native cell mismatch')
                    row={k:key[k] for k in KEYS}
                    row.update({'era5_land_swe_source_sha256':sha(path),'era5_land_swe_param_id':p['paramId'],
                                'era5_land_swe_expver':'0001','era5_land_swe_units':p['units'],'era5_land_swe_step_type':p['stepType'],
                                'decoded_latitude':float(lat[ix]),'decoded_longitude':(float(lon[ix])+180)%360-180,
                                'decoded_validity_date':valid[0],'decoded_validity_time':valid[1]})
                    if p['paramId']==141:row[VALUE]=float(vals[ix]);records.append(row)
                    else:row['crosscheck_temperature_c']=float(vals[ix]-273.15);cross.append(row)
            finally:ec.codes_release(h)
    require(len(messages)==job['returned_fields'],'Unexpected GRIB field count')
    require(seen==requested,'Full requested GRIB identity set differs')
    require(len(records)==len(keys),'Incomplete SWE cell/hour extraction')
    result=pd.DataFrame(records)
    require(not result.duplicated(KEYS).any(),'Duplicate extracted keys')
    # Preserve source numbers unchanged. The separately frozen merger owns the
    # tiny-negative zero repair and marks larger negatives physically invalid.
    return result,pd.DataFrame(cross),messages

def decode(output,probe=False):
    out=Path(output);plan=read(out/'plan.json');state=read(out/'state.json')
    require(sha(__file__)==plan['source_sha256'] and sha(out/'plan.json')==state['request_plan_sha256'],'Frozen request changed')
    require(sha(out/'native_cells.parquet')==plan['native_cells_sha256'],'Native keys changed')
    keys=pd.read_parquet(out/'native_cells.parquet');frames=[];cross=[];proofs=[]
    for job in plan['jobs'][:1] if probe else plan['jobs']:
        entry=state['jobs'][job['id']];require(entry['status']=='downloaded','CDS job incomplete')
        raw=entry['raw'];require(sha(raw['path'])==raw['sha256'],'GRIB bytes changed')
        month=job['id'][-7:]
        sub=keys[(keys.region_id==job['region_id'])&(keys[KEYS[1]].dt.strftime('%Y-%m')==month)]
        f,c,p=parse_job(raw['path'],job,sub);frames.append(f);cross.append(c)
        proofs.append({'job':job['id'],'request_id':entry['request_id'],'raw':raw,'messages':p})
    result=pd.concat(frames,ignore_index=True)
    name='probe_values.parquet' if probe else 'values.parquet'
    result.to_parquet(out/name,index=False)
    pd.concat(cross,ignore_index=True).to_parquet(out/('probe_crosscheck.parquet' if probe else 'crosscheck.parquet'),index=False)
    save(out/('probe_sources.json' if probe else 'sources.json'),proofs)
    if not probe:
        require(len(result)==plan['rows'],'Final row count mismatch')
        merged=keys.merge(result,on=KEYS,how='outer',validate='one_to_one',indicator=True)
        require((merged['_merge']=='both').all(),'Final keys mismatch')
        save(out/'completion.json',{'status':'complete','rows':len(result),'finite_rows':int(result[VALUE].notna().sum()),
              'output_sha256':sha(out/name),'request_plan_sha256':sha(out/'plan.json'),'source_sha256':sha(__file__),
              'native_cells_sha256':sha(out/'native_cells.parquet'),'raw_responses':proofs,
              'source_files':[{'path':x['raw']['path'],'sha256':x['raw']['sha256']} for x in proofs],
              'sources_sha256':sha(out/'sources.json'),'state_sha256':sha(out/'state.json'),
              'raw_negative_rows':int((result[VALUE]<0).sum()),'raw_negative_values_preserved':True,
              'reserved_2025_opened':False,'thermal_labels_decoded':False,'download_bytes':state['download_charged_bytes'],
              'api_requests':state['api_requests'],'elapsed_seconds':time.time()-state['started_epoch'],
              'prior_elapsed_seconds':state.get('prior_elapsed_seconds',0)})
    log(decoded='probe' if probe else 'complete',rows=len(result),finite=int(result[VALUE].notna().sum()),snow_min=float(result[VALUE].min()),snow_max=float(result[VALUE].max()))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','continue','step','probe','decode']);p.add_argument('--output',required=True)
    p.add_argument('--previous')
    p.add_argument('--native');p.add_argument('--period',choices=['fit','evaluation'],default='fit');p.add_argument('--submit-limit',type=int)
    a=p.parse_args()
    try:
        if a.action=='plan':make_plan(a.native,a.output,a.period)
        elif a.action=='continue':continue_plan(a.previous,a.output)
        elif a.action=='step':step(a.output,a.submit_limit)
        else:decode(a.output,probe=a.action=='probe')
    except Exception as e:
        # The API client's exception body may include signed URLs; do not log it.
        log(status='stopped',error_type=type(e).__name__)
        if isinstance(e,SafeValidationError):log(check=str(e))
        sys.exit(1)
if __name__=='__main__':main()
