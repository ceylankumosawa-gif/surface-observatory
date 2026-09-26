"""Bounded calendar-stratified Landsat expansion with independent predictors."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
import planetary_computer
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin, array_bounds
from rasterio.warp import Resampling

from . import raster, satellite
from .ecostress import save_json, digest

VERSION = 'option-b-day-20260909-v1'
PERIODS = [('2021-01-01/2022-01-01', 8), ('2022-01-01/2023-01-01', 8),
           ('2023-01-01/2023-07-01', 4), ('2023-07-01/2024-01-01', 4)]
CENTRES = ((.4375,.4375),(.4375,.5625),(.5625,.4375),(.5625,.5625),
           (.125,.125),(.125,.875),(.875,.125),(.875,.875))


def fixed_windows(region):
    """Four central and four outer 2 km windows, fixed without temperatures."""
    height, width = region['grid_shape']
    left, _, _, top = region['extent_m']
    for i, (yr, xc) in enumerate(CENTRES):
        rr = max(0, min(height-20, round(height*yr)-10))
        cc = max(0, min(width-20, round(width*xc)-10))
        transform = from_origin(left+cc*100, top-rr*100, 100, 100)
        yield f'w{i}', raster.RasterGrid(region['epsg'], transform, 20, 20,
                array_bounds(20,20,transform), None, None, np.ones((20,20),bool), rr, cc)


def aggregate_labels(raw, src_transform, src_crs, grid):
    """Independent optical values are computed before the separate label mask."""
    features, _ = raster.aggregate_optical(raw, src_transform, src_crs, grid)
    good = satellite.qa_valid(raw['qa_pixel'],raw['qa_radsat']) & (raw['lwir11']>0)
    good &= (raw['qa']*.01 <= 3)
    fraction = raster._warp(good,src_transform,src_crs,grid)
    observed = raster._warp(np.where(good,satellite.scale_temperature(raw['lwir11']),np.nan),
                            src_transform,src_crs,grid,np.nan)
    error = raster._warp(np.where(good,raw['qa']*.01,np.nan),src_transform,src_crs,grid,np.nan,Resampling.max)
    snow = raster._warp(np.where(good,(raw['qa_pixel']>>5)&1,np.nan),src_transform,src_crs,grid,np.nan)
    return features, {'lst_c':np.where(fraction>=.8,observed,np.nan),
                      'label_valid_fraction':fraction,'max_source_lst_error_k':error,
                      'snow_fraction':snow}


def sample_scene(scene,region,per_window=60):
    required=[*satellite.SR_BANDS,'qa_pixel','qa_radsat','lwir11','qa']
    if not set(required).issubset(scene.get('assets',{})):
        raise ValueError('Scene lacks independent optical or temperature-quality assets.')
    stamp=pd.Timestamp(scene['properties']['datetime'])
    if stamp.year not in (2021,2022,2023):
        raise ValueError('Expansion refuses reserved test years.')
    frames=[];audit=[]
    with rasterio.Env(**raster.GDAL_ENV),ExitStack() as stack:
        sources={b:stack.enter_context(rasterio.open(planetary_computer.sign(scene['assets'][b]['href']))) for b in required}
        base=sources['red']
        if any((s.crs,s.transform,s.shape)!=(base.crs,base.transform,base.shape) for s in sources.values()):
            raise ValueError('Landsat asset grids are inconsistent.')
        for name,grid in fixed_windows(region):
            win=raster._native_window(base,grid)
            raw={b:s.read(1,window=win,boundless=True,fill_value=1 if b=='qa_pixel' else 0) for b,s in sources.items()}
            features,labels=aggregate_labels(raw,base.window_transform(win),base.crs,grid)
            valid=np.isfinite(labels['lst_c'])
            for key in satellite.SURFACE_FEATURES:valid &= np.isfinite(features[key])
            # Independent optical water screen; comprehensive land fractions follow.
            valid &= np.isfinite(features['water_fraction']) & (features['water_fraction']<=.05)
            choices=np.flatnonzero(valid)
            rng=np.random.default_rng(int(hashlib.sha256(f'2708:{region["id"]}:{scene["id"]}:{name}'.encode()).hexdigest()[:16],16))
            choices=rng.permutation(choices)[:per_window]
            audit.append({'window':name,'available_cells':int(valid.sum()),'sampled_cells':len(choices),
                          'grid_origin':[grid.row_offset,grid.col_offset]})
            if not len(choices):continue
            f=raster.feature_frame(region,grid,{**features,**labels},stamp).iloc[choices].copy()
            rr,cc=np.divmod(choices,grid.width)
            f['grid_row']=rr+grid.row_offset;f['grid_col']=cc+grid.col_offset
            f['window_id']=name;f['scene_id']=scene['id'];f['acquisition_id']=scene['id']
            f['label_product']='landsat_c2_l2';f['label_condition']='clear_sky_daytime'
            f['cohort_origin']='expanded';f['epsg']=region['epsg']
            f['sample_id']=[hashlib.sha256(f'landsat_c2_l2:{scene["id"]}:{p}'.encode()).hexdigest()[:32] for p in f.pixel_id]
            for key in ('optical_source_datetime_utc','optical_start_utc','optical_end_utc'):f[key]=stamp
            f['optical_source_id']=scene['id'];f['optical_source_count']=1;f['optical_age_days']=0.
            f['optical_predictor_qa_independence']='independent optical-only mask before thermal labels'
            f['label_source']='USGS Landsat C2 L2 Tier1, independently masked optical predictors'
            f['label_source_sha256']=hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest()
            f.drop(columns=['_raster_position'],inplace=True)
            frames.append(f)
    data=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    return data,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-scenes',type=int,default=48);p.add_argument('--runtime-minutes',type=int,default=45)
    args=p.parse_args();logging.basicConfig(level=logging.WARNING);satellite.configure_safe_logging()
    args.output.mkdir(parents=True,exist_ok=True)
    planpath=args.output/'plan.json'
    if planpath.exists():
        plan=json.loads(planpath.read_text())
        if plan['source_sha256']!=digest(__file__):raise ValueError('Daytime source changed; use a new run directory.')
    else:
        areas=json.loads((args.root/'pilot/areas_resolved.json').read_text())['areas']
        selected=[r for r in areas if r['id'] in ('greater_london','sioux_falls')]
        plan={'version':VERSION,'source_sha256':digest(__file__),'records':[],
              'selection':'Fixed calendar strata, then footprint and catalog cloud metadata. No temperature/error selection.',
              'windows':list(CENTRES),'per_window_samples':60,'max_scenes':args.max_scenes}
        for region in selected:
            for period,quota in PERIODS:
                items=satellite.search_scenes(region,period,args.output/'stac',quota,quota*6)
                plan['records'].append({'region':region,'period':period,'quota':quota,'items':items[:quota*3]})
        save_json(planpath,plan)
    logpath=args.output/'audit.json'
    log=json.loads(logpath.read_text()) if logpath.exists() else {'plan_sha256':digest(planpath),'records':[]}
    if log['plan_sha256']!=digest(planpath):raise ValueError('Frozen plan changed.')
    done={(r['region_id'],r['scene_id']):r for r in log['records']}
    started=time.monotonic()
    for group in plan['records']:
        region=group['region'];success=0
        for item in group['items']:
            key=(region['id'],item['id'])
            previous=done.get(key)
            if previous:
                success+=previous.get('rows',0)>=100
                if success>=group['quota']:break
                continue
            if sum(r.get('rows',0)>=100 for r in log['records'])>=args.max_scenes or time.monotonic()-started>=args.runtime_minutes*60:
                print('Bounded daytime run paused; completed checkpoints are reusable.',flush=True);return
            output=args.output/'scenes'/f'{region["id"]}_{item["id"]}.parquet'
            record={'region_id':region['id'],'scene_id':item['id'],'datetime_utc':item['properties']['datetime'],
                    'period':group['period'],'path':str(output)}
            try:
                frame,windows=sample_scene(item,region)
                output.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(output,index=False)
                record.update(status='complete',rows=len(frame),windows=windows,sha256=digest(output))
                success+=len(frame)>=100
            except Exception as error:
                record.update(status='failed',rows=0,error=satellite._safe_error(error))
            log['records'].append(record);done[key]=record
            log['elapsed_seconds']=time.monotonic()-started
            save_json(logpath,log)
            print(json.dumps({k:record[k] for k in ('region_id','scene_id','status','rows')}),flush=True)
            if success>=group['quota']:break
    log['completed']=True;save_json(logpath,log)


if __name__=='__main__':main()
