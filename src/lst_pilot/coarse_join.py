"""Sample-preserving causal native context joins; no fine-resolution labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import from_wkb
from shapely.geometry import Point
from shapely.strtree import STRtree
from .coarse_inventory import sha,save

CONTEXT_COLUMNS=['native_cell_id','region_id','acquisition_id','granule_start_utc','granule_end_utc',
    'native_footprint_wkb','native_footprint_area_m2','footprint_epsg','view_zenith_deg','product','lst_c','lst_error_k',
    'source_sha256','context_support_wkb','context_support_margin_m','context_eligible','context_fit_eligible']


def join_frame(samples,native,for_fitting=False,max_age_hours=24):
    required=['sample_id','region_id','datetime_utc','latitude','longitude','epsg']
    if any(k not in samples for k in required) or samples.sample_id.duplicated().any():raise ValueError('Unique sample_id and explicit time/location/CRS required.')
    samples=samples.reset_index(drop=True)
    if any(pd.Timestamp(t).tzinfo is None for t in samples.datetime_utc.unique()):raise ValueError('Explicit UTC-offset timestamps required.')
    result=samples[['sample_id']].copy();result['coarse_context_eligible']=False
    numeric=['coarse_lst_c','coarse_age_hours','coarse_age_min_hours','coarse_lst_error_k','coarse_native_area_m2','coarse_support_margin_m']
    for key in numeric:result[key]=np.nan
    strings=['coarse_product','source_granule_start','source_granule_end','coarse_native_id','coarse_acquisition_id',
             'coarse_source_sha256','coarse_context_table_sha256','coarse_footprint_wkb','coarse_support_wkb']
    for key in strings:result[key]=pd.Series(None,index=result.index,dtype=object)
    result['coarse_context_missing_reason']='no_QA_geometry_and_time_eligible_native_support'
    n=native.copy()
    n['_start']=pd.to_datetime(n.granule_start_utc,utc=True);n['_end']=pd.to_datetime(n.granule_end_utc,utc=True)
    gate='context_fit_eligible' if for_fitting else 'context_eligible';n=n.loc[n[gate]].copy()
    samples=samples.copy();samples['_target']=pd.to_datetime(samples.datetime_utc,utc=True)
    for (region,target,epsg),indices in samples.groupby(['region_id','_target','epsg'],sort=False).groups.items():
        candidates=n.loc[n.region_id.eq(region)&(n._end<=target)&(n._start>=target-pd.Timedelta(hours=max_age_hours))].copy()
        if len(candidates)==0:continue
        if not candidates.footprint_epsg.eq(epsg).all():raise ValueError('Context footprint CRS differs from sample pilot.')
        # Granule publication time is deliberately not treated as observation time.
        candidates=candidates.sort_values(['_end','view_zenith_deg','native_cell_id'],ascending=[False,True,True]).reset_index(drop=True)
        polygons=[from_wkb(x) for x in candidates.native_footprint_wkb];tree=STRtree(polygons)
        x,y=Transformer.from_crs(4326,int(epsg),always_xy=True).transform(samples.loc[indices,'longitude'].to_numpy(),samples.loc[indices,'latitude'].to_numpy())
        for index,xx,yy in zip(indices,x,y):
            if not np.isfinite(xx) or not np.isfinite(yy):continue
            point=Point(xx,yy);hits=[int(i) for i in tree.query(point) if polygons[int(i)].covers(point)]
            if not hits:continue
            row=candidates.iloc[min(hits)]
            values={'coarse_context_eligible':True,'coarse_context_missing_reason':'',
                'coarse_lst_c':row.lst_c,'coarse_age_hours':(target-row._start).total_seconds()/3600,
                'coarse_age_min_hours':(target-row._end).total_seconds()/3600,'coarse_lst_error_k':row.lst_error_k,
                'coarse_product':row['product'],'source_granule_start':row.granule_start_utc,'source_granule_end':row.granule_end_utc,
                'coarse_native_id':row.native_cell_id,'coarse_acquisition_id':row.acquisition_id,'coarse_native_area_m2':row.native_footprint_area_m2,
                'coarse_support_margin_m':row.context_support_margin_m,'coarse_source_sha256':row.source_sha256,
                'coarse_context_table_sha256':row.get('context_table_sha256',''),'coarse_footprint_wkb':row.native_footprint_wkb,
                'coarse_support_wkb':row.context_support_wkb}
            for key,value in values.items():result.at[index,key]=value
    return result


def run(input_path,manifest_path,output,for_fitting):
    # Intentionally read no fine-resolution thermal labels from the input.
    columns=['sample_id','region_id','datetime_utc','latitude','longitude','epsg']
    samples=pd.read_parquet(input_path,columns=columns);manifest=json.loads(Path(manifest_path).read_text());parts=[]
    for record in manifest['records']:
        if sha(record['table_path'])!=record['table_sha256']:raise ValueError('Context source checksum changed.')
        frame=pd.read_parquet(record['table_path'],columns=CONTEXT_COLUMNS)
        frame['context_table_sha256']=record['table_sha256'];parts.append(frame)
    if not parts:raise ValueError('No audited native context sources.')
    native=pd.concat(parts,ignore_index=True);joined=join_frame(samples,native,for_fitting)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);path=output/'coarse_context.parquet';joined.to_parquet(path,index=False)
    audit={'input_sha256':sha(input_path),'input_columns_read':columns,'context_manifest_sha256':sha(manifest_path),'join_source_sha256':sha(__file__),
        'rows':len(joined),'matched_rows':int(joined.coarse_context_eligible.sum()),'unique_native_cells':int(joined.coarse_native_id.nunique()),
        'for_fitting':for_fitting,'maximum_oldest_possible_age_hours':24,'time_rule':'Entire granule interval completed at target; start at or after target minus 24 hours',
        'age_semantics':'coarse_age_hours = target minus source start (oldest possible age); coarse_age_min_hours = target minus source end',
        'native_label_warning':'Repeated context value across fine samples is one shared native observation, not independent fine labels',
        'output_path':str(path.resolve()),'output_sha256':sha(path)}
    save(output/'manifest.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--context-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--for-fitting',action='store_true')
    a=p.parse_args();run(a.input,a.context_manifest,a.output,a.for_fitting)
