"""Two frozen same-Terra ASTER/MOD21 native-footprint diagnostics, not labels.

No network, fitting, temperature-based selection or geometric correction.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
import rasterio
import shapely
from shapely.affinity import translate
from shapely.ops import transform

from . import aster, ecostress as eco

CASES=[('greater_london','AST_08_00409082021111205_20251121071631'),
       ('sioux_falls','AST_08_00406152021173237_20250915193549')]
SHIFTS=[(-180,0),(-90,0),(90,0),(180,0),(0,-180),(0,-90),(0,90),(0,180)]


def aggregate_polygon(poly, values, valid, affine):
    inverse=~affine
    x,y=np.asarray(poly.exterior.coords).T
    cc,rr=inverse*(x,y)
    r0=max(0,int(np.floor(rr.min())));r1=min(values.shape[0],int(np.ceil(rr.max())))
    c0=max(0,int(np.floor(cc.min())));c1=min(values.shape[1],int(np.ceil(cc.max())))
    if r1<=r0 or c1<=c0:return None,0.,0
    if (r1-r0)*(c1-c0)>30000:raise ValueError('Unexpected coarse footprint size.')
    rows,cols=np.indices((r1-r0,c1-c0));rows=rows.ravel()+r0;cols=cols.ravel()+c0
    coords=[]
    for dx,dy in ((0,0),(1,0),(1,1),(0,1),(0,0)):
        sx,sy=affine*(cols+dx,rows+dy)
        coords.append(np.column_stack([sx,sy]))
    native=shapely.polygons(np.stack(coords,axis=1))
    weights=shapely.area(shapely.intersection(native,poly))
    weights=np.where(valid[rows,cols],weights,0.)
    fraction=float(weights.sum()/poly.area)
    if weights.sum()<=0:return None,fraction,0
    return float(np.average(values[rows,cols],weights=weights)),fraction,int((weights>0).sum())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);root=args.root;output=args.output
    if output.exists():raise ValueError('A new immutable diagnostic directory is required.')
    manifest_path=root/'runs/multisensor_20260910_v1/highres_engineering_v1/manifest.json'
    records=json.loads(manifest_path.read_text())['records']
    coarse_root=root/'runs/multisensor_20260910_v1/coarse/context_2021_2022_v1/native'
    planned=[];hashes={str(Path(__file__)):eco.digest(__file__),str(manifest_path):eco.digest(manifest_path),str(Path(aster.__file__)):eco.digest(aster.__file__)}
    for pilot,title in CASES:
        record=next(r for r in records if r['pilot_id']==pilot and r['title']==title)
        when=pd.Timestamp(record['time_start']);matches=[]
        for path in sorted((coarse_root/pilot).glob('MOD21*/native_cells.parquet')):
            if pq.ParquetFile(path).metadata.num_rows==0:continue
            temporal=pd.read_parquet(path,columns=['granule_start_utc','granule_end_utc'])
            if len(temporal) and pd.Timestamp(temporal.iloc[0,0])<=when<=pd.Timestamp(temporal.iloc[0,1]):matches.append(path)
        if len(matches)!=1:raise ValueError('Exactly one preexisting MOD21 same-swath counterpart required.')
        files={layer:root/'cache/aster_v4'/title/(layer+'.tif') for layer in aster.LAYERS}
        for layer,path in files.items():
            if eco.digest(path)!=record['downloads'][layer]['sha256']:raise ValueError('ASTER source hash changed.')
            hashes[str(path)]=eco.digest(path)
        hashes[str(matches[0])]=eco.digest(matches[0])
        planned.append({'pilot':pilot,'aster_title':title,'aster_time':when.isoformat(),'coarse_table':str(matches[0]),'native_files':{k:str(v) for k,v in files.items()}})
    output.mkdir(parents=True)
    plan={'cases':planned,'source_hashes':hashes,'minimum_aster_valid_native_area':.9,
          'shift_sensitivity_offsets_native_crs_m':SHIFTS,
          'coarse_gate':'native_qa_valid and at least99% footprint inside pilot; no temperature/error sorting',
          'purpose':'Same-swath descriptive sensor comparison only; not causal context at the ASTER instant, registration proof, cloud certification, calibration or fitting.'}
    eco.save_json(output/'plan.json',plan) # Before opening ASTER temperature arrays.
    summaries=[];comparisons=[]
    for case in planned:
        table=pd.read_parquet(case['coarse_table'])
        subset=table.loc[table.native_qa_valid & (table.pilot_overlap_fraction>=.99)].copy()
        with ExitStack() as stack:
            sources={k:stack.enter_context(rasterio.open(v)) for k,v in case['native_files'].items()}
            source=sources['SKT'];raw={k:s.read(1) for k,s in sources.items()}
            valid,kelvin,_=aster.native_valid(raw,source.transform)
            celsius=kelvin.astype('float64')-273.15
            project=Transformer.from_crs(int(subset.footprint_epsg.iloc[0]),source.crs,always_xy=True).transform if len(subset) else None
            accepted=[]
            for row in subset.itertuples():
                poly=transform(project,shapely.from_wkb(row.native_footprint_wkb))
                estimate,fraction,native_count=aggregate_polygon(poly,celsius,valid,source.transform)
                if estimate is None or fraction<.9:continue
                shifts={}
                for dx,dy in SHIFTS:
                    value,shift_fraction,_=aggregate_polygon(translate(poly,xoff=dx,yoff=dy),celsius,valid,source.transform)
                    shifts[f'{dx},{dy}']={'aster_c':value,'valid_fraction':shift_fraction,
                                         'supported':value is not None and shift_fraction>=.9}
                record={'pilot_id':case['pilot'],'aster_title':case['aster_title'],'aster_time_utc':case['aster_time'],
                        'mod21_native_cell_id':row.native_cell_id,'mod21_start_utc':row.granule_start_utc,'mod21_end_utc':row.granule_end_utc,
                        'mod21_c':float(row.lst_c),'aster_native_area_mean_c':estimate,'aster_minus_mod21_c':estimate-float(row.lst_c),
                        'aster_valid_area_fraction':fraction,'aster_contributing_native_cells':native_count,
                        'coarse_native_footprint_wkb':row.native_footprint_wkb,'coarse_footprint_epsg':int(row.footprint_epsg),
                        'shift_sensitivity_json':json.dumps(shifts),
                        'source_screen_pass':False,'training_eligible':False,'registration_verified':False,'independent_fine_cloud_proven':False}
                accepted.append(record)
        comparisons.extend(accepted)
        delta=np.array([r['aster_minus_mod21_c'] for r in accepted])
        common=[r for r in accepted if all(x['supported'] for x in json.loads(r['shift_sensitivity_json']).values())]
        sensitivity=[]
        for dx,dy in SHIFTS:
            key=f'{dx},{dy}'
            changes=np.array([json.loads(r['shift_sensitivity_json'])[key]['aster_c']-r['aster_native_area_mean_c'] for r in common])
            differences=np.array([json.loads(r['shift_sensitivity_json'])[key]['aster_c']-r['mod21_c'] for r in common])
            sensitivity.append({'offset_native_crs_m':[dx,dy],'common_supported_footprints':len(common),
                                'mean_change_from_unshifted_c':float(changes.mean()) if len(changes) else None,
                                'mean_abs_change_from_unshifted_c':float(np.abs(changes).mean()) if len(changes) else None,
                                'p95_abs_change_from_unshifted_c':float(np.quantile(np.abs(changes),.95)) if len(changes) else None,
                                'mean_aster_minus_mod21_c':float(differences.mean()) if len(differences) else None})
        summaries.append({'pilot_id':case['pilot'],'aster_title':case['aster_title'],'aster_time_utc':case['aster_time'],
                          'coarse_qa_and_pilot_footprints':len(subset),'paired_native_footprints':len(accepted),
                          'mean_aster_minus_mod21_c':float(delta.mean()) if len(delta) else None,
                          'median_aster_minus_mod21_c':float(np.median(delta)) if len(delta) else None,
                          'mean_absolute_difference_c':float(np.abs(delta).mean()) if len(delta) else None,
                          'difference_p05_p95_c':np.quantile(delta,[.05,.95]).tolist() if len(delta) else None,
                          'shift_sensitivity_on_common_supported_footprints':sensitivity,
                          'independent_dates':1,'fine_registration_verified':False,'independent_fine_cloud_proven':False})
    path=output/'native_footprint_comparison.parquet';pd.DataFrame(comparisons).to_parquet(path,index=False)
    result={'plan_sha256':eco.digest(output/'plan.json'),'table_path':str(path),'table_sha256':eco.digest(path),
            'summary':summaries,'network_requests':0,'training_eligible':False,
            'limitations':['Two fitting-year daytime engineering cases only; no night or extreme-weather validation.',
                           'MOD21 retrieval QA supplies coarser cloud context, not 90m cloud evidence.',
                           'Area means use ASTER reported rotated native grid without independent position correction.',
                           'Fixed cardinal90m/180m footprint perturbations measure coarse-aggregate sensitivity, not actual registration error; no shift is selected or applied.',
                           'MOD21 swath contains ASTER time but ends later: diagnostic only, never an available past-context predictor.',
                           'Differences cannot be attributed solely to one sensor or turned into a calibration from these two scenes.']}
    eco.save_json(output/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
