"""Bounded NOAA source cohort; no model fitting or LST-label admission."""
from __future__ import annotations
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from lst_pilot import reference

ROOT=Path('/opt/lst-pilot')
HERE=Path(__file__).resolve().parent
STATIONS={'bon':'Bondville_IL','dra':'Desert_Rock_NV','fpk':'Fort_Peck_MT','gwn':'Goodwin_Creek_MS',
          'psu':'Penn_State_PA','sxf':'Sioux_Falls_SD','tbl':'Boulder_CO'}
BASE='https://gml.noaa.gov/aftp/data/radiation/surfrad'
MAX_REQUESTS=32
MAX_BYTES=20*2**20


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data): Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def parse(path, expected_day, station):
    data=reference.parse_day(path,keep_raw=True)
    lines=Path(path).read_text().strip().splitlines()
    if len(data)!=len(lines)-2 or not 1<=len(data)<=1440: raise ValueError('Invalid or silently lost minute rows')
    if data.timestamp_utc.duplicated().any() or not data.timestamp_utc.dt.strftime('%Y-%m-%d').eq(expected_day).all():
        raise ValueError('Unexpected or duplicate minute times')
    if not data.day_of_year.eq(pd.Timestamp(expected_day).dayofyear).all(): raise ValueError('Day-of-year mismatch')
    if data.station_id.nunique()!=1 or data.station_id.iloc[0]!=station: raise ValueError('Station identity mismatch')
    lat,lon=data.attrs['header_latitude'],data.attrs['header_longitude']
    if not (24<lat<50 and -125<lon<-65): raise ValueError('Station header coordinates do not match western-hemisphere US network')
    for key in ['longwave_up_w_m2','longwave_down_w_m2']:
        flags=data[key+'_qc']
        if not np.isfinite(flags).all() or not flags.eq(flags.astype(int)).all(): raise ValueError('Malformed thermal QC')
        if data.loc[flags.ne(0),key].notna().any(): raise ValueError('QC failure survived masking')
    for epsilon in (.95,.97,.99,1.):
        field='radiometric_e'+str(epsilon).replace('.','p')+'_c'
        data[field]=reference.radiometric_temperature_c(data.longwave_up_w_m2,data.longwave_down_w_m2,epsilon)
    low=data.radiometric_e0p95_c.to_numpy();high=data.radiometric_e0p99_c.to_numpy()
    data['emissivity_sensitivity_min_c']=np.minimum(low,high)
    data['emissivity_sensitivity_max_c']=np.maximum(low,high)
    data['phase']=np.where(data.is_daylight,'day','night')
    data['station_latitude']=lat;data['station_longitude']=lon
    data['relative_humidity_bin']=np.where(data.relative_humidity_percent.ge(80),'high_humidity',np.where(data.relative_humidity_percent.le(50),'low_humidity','intermediate_or_missing'))
    data['training_eligible']=False;data['temperature_truth_validated']=False
    data['cloud_condition']='unclassified_all_sky';data['wet_ground_observed']=False
    data['source_sha256']=sha(path)
    return data


def run(output):
    started=time.monotonic()
    if output.exists(): raise ValueError('Use a new immutable source run')
    output.mkdir(parents=True);(output/'raw').mkdir()
    metadata=[]
    for name in ['surfrad_metadata_v1','surfrad_metadata_v2']:
        path=ROOT/'runs/global_release_20260918/accuracy'/name/'requests.json'
        entries=json.loads(path.read_text())
        for entry in entries:
            if sha(entry['path'])!=entry['sha256']:raise ValueError('Metadata receipt changed')
        metadata.extend(entries)
    ledger={'requests':len(metadata),'received_bytes':sum(e['bytes'] for e in metadata),'records':metadata.copy()}
    plan=[]
    for station,directory in STATIONS.items():
        for month in (1,4,7,10):
            d=date(2021,month,15);filename=f'{station}{d:%y}{d.timetuple().tm_yday:03d}.dat'
            plan.append({'station':station,'utc_date':d.isoformat(),'url':f'{BASE}/{directory}/2021/{filename}','filename':filename})
    save(output/'plan.json',{'records':plan,'maximum_requests':MAX_REQUESTS,'maximum_bytes':MAX_BYTES,
        'protocol':{'path':str(HERE/'SURFRAD_PLAN.md'),'sha256':sha(HERE/'SURFRAD_PLAN.md')},
        'source_sha256':sha(__file__),'decoder_sha256':sha(reference.__file__),'metadata_records':metadata,
        'selection':'7 fixed stations x Jan/Apr/Jul/Oct15,2021; no temperature/cloud/error selection','training_eligible':False})
    save(output/'ledger.json',ledger)
    frames=[];results=[]
    for item in plan:
        if ledger['requests']>=MAX_REQUESTS or time.monotonic()-started>600:raise ValueError('Campaign limit reached')
        receipt={**item,'status':'attempting','bytes':0};ledger['requests']+=1;ledger['records'].append(receipt);save(output/'ledger.json',ledger)
        try:
            with requests.get(item['url'],stream=True,timeout=(10,45),allow_redirects=False) as response:
                receipt['http_status']=response.status_code
                if response.status_code!=200:raise ValueError('Expected exact official source HTTP200')
                parts=[]
                for block in response.iter_content(65536):
                    receipt['bytes']+=len(block);ledger['received_bytes']+=len(block)
                    save(output/'ledger.json',ledger)
                    if receipt['bytes']>2*2**20 or ledger['received_bytes']>MAX_BYTES:raise ValueError('Transfer cap reached')
                    parts.append(block)
            raw=output/'raw'/item['filename'];raw.write_bytes(b''.join(parts))
            receipt['sha256']=sha(raw);receipt['path']=str(raw)
            frame=parse(raw,item['utc_date'],item['station']);frames.append(frame)
            receipt['status']='complete'
            results.append({**item,'rows':len(frame),'valid_thermal_minutes':int(frame.radiometric_e0p97_c.notna().sum()),
                            'latitude':float(frame.station_latitude.iloc[0]),'longitude':float(frame.station_longitude.iloc[0]),
                            'source_sha256':sha(raw)})
        except (ValueError,requests.RequestException) as error:
            receipt['status']='failed';receipt['error']=str(error)[:300]
        save(output/'ledger.json',ledger)
        if ledger['received_bytes']>=MAX_BYTES:raise ValueError('Cumulative transfer limit')
        time.sleep(.25)
    if frames:
        data=pd.concat(frames,ignore_index=True);data.attrs={};data.to_parquet(output/'minutes.parquet',index=False)
        summary=[]
        for (station,phase,humidity),g in data.groupby(['station_id','phase','relative_humidity_bin']):
            valid=g.radiometric_e0p97_c.notna();values=g.loc[valid]
            summary.append({'station_id':station,'phase':phase,'relative_humidity_bin':humidity,'minutes':len(g),
                'valid_thermal_minutes':int(valid.sum()),'utc_dates':int(g.timestamp_utc.dt.date.nunique()),
                'radiometric_e0p97_mean_c':float(values.radiometric_e0p97_c.mean()) if len(values) else None,
                'air_10m_mean_c':float(values.air_temperature_10m_c.mean()) if values.air_temperature_10m_c.notna().any() else None,
                'emissivity_sensitivity_mean_width_c':float((values.emissivity_sensitivity_max_c-values.emissivity_sensitivity_min_c).mean()) if len(values) else None})
        pd.DataFrame(summary).to_csv(output/'phase_humidity_summary.csv',index=False)
    save(output/'source_summary.json',results)
    save(output/'completion.json',{'status':'complete' if len(frames)==28 else 'partial','successful_days':len(frames),
        'planned_days':28,'stations':len(set(r['station'] for r in results)), 'requests':ledger['requests'],
        'received_bytes':ledger['received_bytes'],'wall_seconds':time.monotonic()-started,'training_eligible':False,
        'source_truth_status':'emissivity-dependent footprint diagnostic; not validated100mLST',
        'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file()},'reserved_2025_opened':False})
    print(json.dumps(json.loads((output/'completion.json').read_text())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output.resolve())
