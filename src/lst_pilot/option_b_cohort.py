"""Deterministic research row sampling and permanent geographic exclusions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin
from .ecostress import digest, save_json

VERSION = 'option-b-cohort-20260909-v3'


def coordinate_flags(x,y,area):
    left,bottom,right,top=area['extent_m']
    br=np.floor((y-bottom)/10000).astype(int);bc=np.floor((x-left)/10000).astype(int)
    reserved=(br+2*bc)%5==0
    buffer=np.zeros(np.shape(x),bool)
    for r in range(int(np.ceil((top-bottom)/10000))):
        for c in range(int(np.ceil((right-left)/10000))):
            if (r+2*c)%5:continue
            x0,x1=left+c*10000,min(right,left+(c+1)*10000)
            y0,y1=bottom+r*10000,min(top,bottom+(r+1)*10000)
            # A retained 100 m cell's complete footprint must clear the buffer.
            dx=np.maximum(np.maximum(x0-(x+50),(x-50)-x1),0)
            dy=np.maximum(np.maximum(y0-(y+50),(y-50)-y1),0)
            buffer |= np.hypot(dx,dy)<=1000
    return br,bc,reserved,buffer & ~reserved


def spatial_flags(frame, areas):
    """Pilot-origin 10 km rectangles; whole cell footprints clear a 1 km buffer."""
    data=frame.copy()
    data['spatial_holdout']=False;data['in_holdout_buffer']=False;data['block_id']='unreserved_pilot'
    for region_id,idx in data.groupby('region_id').groups.items():
        area=areas[region_id];left,bottom,right,top=area['extent_m']
        x=left+(data.loc[idx,'grid_col'].to_numpy()+.5)*100
        y=top-(data.loc[idx,'grid_row'].to_numpy()+.5)*100
        br,bc,reserved,buffer=coordinate_flags(x,y,area)
        data.loc[idx,'block_id']=[f'{region_id}_r{r:02d}_c{c:02d}' for r,c in zip(br,bc)]
        if region_id not in ('greater_london','sioux_falls'):continue
        data.loc[idx,'spatial_holdout']=reserved
        data.loc[idx,'in_holdout_buffer']=buffer
    return data


def choose_night_positions(valid, key, maximum=400, safe=None, reserved=None):
    """One tile per quadrant; rank safe support, allocate 80 safe/20 held-out."""
    split_sampling=safe is not None
    if safe is None:safe=valid
    if reserved is None:reserved=np.zeros_like(valid)
    h,w=valid.shape;tiles=[]
    for r in range(0,h,128):
        for c in range(0,w,128):
            rr,cc=np.where(valid[r:r+128,c:c+128]);rr=rr+r;cc=cc+c
            if len(rr):
                quadrant=(int((r+min(128,h-r)/2)>=h/2),int((c+min(128,w-c)/2)>=w/2))
                tiles.append((quadrant,r,c,rr,cc,int(safe[rr,cc].sum())))
    selected=[]
    for quadrant in sorted({t[0] for t in tiles}):
        options=[t for t in tiles if t[0]==quadrant]
        selected.append(min(options,key=lambda t:(-t[5],-len(t[3]),t[1],t[2])))
    out=[];audit=[]
    for _,r,c,rr,cc,_ in selected:
        seed=int(hashlib.sha256(f'2708:{key}:{r}:{c}'.encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed)
        if split_sampling:
            safe_ix=rng.permutation(np.flatnonzero(safe[rr,cc]))[:maximum//5]
            held_ix=rng.permutation(np.flatnonzero(reserved[rr,cc]))[:maximum//20]
            ix=np.concatenate([safe_ix,held_ix])
        else:ix=rng.permutation(len(rr))[:maximum//4]
        out.extend(zip(rr[ix],cc[ix]));audit.append({'row':r,'col':c,'valid_cells':len(rr),'sampled_cells':len(ix),
                       'sampled_safe':int(safe[rr[ix],cc[ix]].sum()),'sampled_reserved':int(reserved[rr[ix],cc[ix]].sum())})
    return np.asarray(out,dtype=int).reshape(-1,2),audit


def night_frame(record, area):
    if not record.get('source_screen_pass') or not record.get('qualifying_independent_date'):
        raise ValueError('Require a substantive source-screened independent acquisition.')
    if not record.get('geolocation',{}).get('accepted') or record.get('obstruction',{}).get('status') not in ('NotListed','ListedNoObstruction'):
        raise ValueError('Incomplete source geolocation/known-obstruction screen.')
    if record.get('obstruction',{}).get('obstructed') is True:raise ValueError('Confirmed obstruction is excluded.')
    stamp=pd.Timestamp(record['time_start'])
    if stamp.year not in (2021,2022,2023):raise ValueError('Reserved test year.')
    path=Path(record['raster_path'])
    if digest(path)!=record['raster_sha256']:raise ValueError('Source raster hash changed.')
    with rasterio.open(path) as src:
        if src.crs.to_epsg()!=area['epsg'] or src.shape!=tuple(area['grid_shape']):raise ValueError('Pilot grid mismatch.')
        if src.transform!=from_origin(area['extent_m'][0],area['extent_m'][3],100,100):raise ValueError('Pilot origin mismatch.')
        arrays={name:src.read(i+1) for i,name in enumerate(src.descriptions)}
    valid=np.isfinite(arrays['lst_c']) & (arrays['valid_fraction']>=.9) & (arrays['max_source_lst_error_k']<=2)
    rows,cols=np.indices(valid.shape)
    _,_,reserved,buffer=coordinate_flags(area['extent_m'][0]+(cols+.5)*100,
                                         area['extent_m'][3]-(rows+.5)*100,area)
    positions,windows=choose_night_positions(valid,record['title'],safe=valid & ~reserved & ~buffer,reserved=valid & reserved)
    rr,cc=positions.T
    x=area['extent_m'][0]+(cc+.5)*100;y=area['extent_m'][3]-(rr+.5)*100
    lon,lat=Transformer.from_crs(area['epsg'],4326,always_xy=True).transform(x,y)
    frame=pd.DataFrame({'region_id':area['id'],'datetime_utc':stamp,'grid_row':rr,'grid_col':cc,
                         'pixel_x':x,'pixel_y':y,'epsg':area['epsg'],'pixel_epsg':area['epsg'],
                         'longitude':lon,'latitude':lat,'lst_c':arrays['lst_c'][rr,cc],
                         'label_valid_fraction':arrays['valid_fraction'][rr,cc],
                         'max_source_lst_error_k':arrays['max_source_lst_error_k'][rr,cc]})
    frame['pixel_id']=[f'{area["id"]}:{r}:{c}' for r,c in positions]
    frame['acquisition_id']=record['acquisition_group'];frame['scene_id']=record['title']
    frame['sample_id']=[hashlib.sha256(f'ecostress_v2:{record["acquisition_group"]}:{p}'.encode()).hexdigest()[:32] for p in frame.pixel_id]
    frame['label_product']='ecostress_v2';frame['cohort_origin']='expanded';frame['day_night']='night'
    frame['label_condition']='clear_sky_nighttime';frame['source_screen_pass']=True
    frame['label_source_sha256']=record['raster_sha256'];frame['label_source']=record['title']
    frame['registration_status']='positive GEO flag; independent 100 m registration validation still pending'
    frame['obstruction_status']=record['obstruction']['status']
    frame['obstruction_limitation']='Complete official known-obstruction screen; does not prove every pixel unobstructed'
    return frame,windows


def sample_nights(manifest,areas,output):
    source=json.loads(Path(manifest).read_text());frames=[];audit=[];dates=set()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Frozen manifest order retains calendar ranks. One granule per pilot/date.
    for record in source['records']:
        key=(record['pilot_id'],record['utc_date'])
        if key in dates or not record.get('qualifying_independent_date'):continue
        frame,windows=night_frame(record,areas[record['pilot_id']]);dates.add(key);frames.append(frame)
        audit.append({'pilot_id':key[0],'date':key[1],'title':record['title'],'rows':len(frame),'tiles':windows,
                      'source_sha256':record['raster_sha256']})
    if not frames:raise ValueError('No substantive source-screened nights available.')
    data=spatial_flags(pd.concat(frames,ignore_index=True),areas)
    if data.sample_id.duplicated().any():raise ValueError('Duplicate sample identities.')
    data.to_parquet(output/'night_input.parquet',index=False)
    first=data[data.acquisition_id.eq(data.acquisition_id.iloc[0])]
    tile=(first.grid_row//128).astype(str)+':'+(first.grid_col//128).astype(str)
    first.loc[tile.eq(tile.iloc[0])].head(6).to_parquet(output/'night_smoke.parquet',index=False)
    save_json(output/'sampling_manifest.json',{'version':VERSION,'source_manifest_sha256':digest(manifest),
              'source_code_sha256':digest(__file__),'rows':len(data),'acquisitions':audit,
              'output_sha256':digest(output/'night_input.parquet')})
    print(json.dumps({'rows':len(data),'dates':len(dates),'output':str(output/'night_input.parquet')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--night-manifest',required=True);parser.add_argument('--areas',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();areas={a['id']:a for a in json.loads(Path(args.areas).read_text())['areas']}
    sample_nights(args.night_manifest,areas,args.output)


if __name__=='__main__':main()
