"""Conservative eligibility and causal native context, separate from LST labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from shapely import from_wkb
from shapely.geometry import box,Point
from shapely.ops import unary_union
from .coarse_inventory import sha,save,utc
from .coarse_lst import age_interval,partition

GEOMETRY_POLICY={
    'version':'native-context-geometry-v1',
    'native_cell_geometry':'Within-scan quadrilaterals from full-resolution geolocation centers',
    'padding':'Half longest measured native edge plus geolocation allowance times sec(view zenith)^2',
    'MOD03_allowance':'max(150 m, three times granule GEO_EST_RMS_ERROR); engineering proxy, not a certified probability bound',
    'VIIRS_allowance':'375 m published radial 3-sigma requirement; conservative design allowance, not per-granule measured accuracy',
    'view_limit_degrees':30,'heldout_buffer_m':1000,
    'native_coarse_labels_enabled':False,
    'coast_limitation':'Native land-mask class does not prove every subpixel or expanded support is land',
    'psf_limitation':'Expanded geometric support is conservative engineering context support, not a measured point-spread function',
    'viirs_official_source':'https://ntrs.nasa.gov/citations/20220013338',
    'mod03_official_source':'https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD03_c61',
}


def context_margin(polygon,view_deg,product,mod03_rms_m=None):
    if not np.isfinite(view_deg) or not 0<=view_deg<=30:return None
    if product=='MOD21':
        if mod03_rms_m is None or not np.isfinite(mod03_rms_m) or mod03_rms_m<0:return None
        geo=max(150.,3*mod03_rms_m)
    elif product=='VNP21':geo=375.
    else:raise ValueError('Unapproved native context product.')
    edge=np.linalg.norm(np.diff(np.asarray(polygon.exterior.coords),axis=0),axis=1).max()
    return .5*edge+geo/np.cos(np.radians(view_deg))**2


def annotate(frame,summary,area):
    out=frame.copy();supports=[];margins=[];eligible=[];fit=[];partitions=[];pilot=box(*area['extent_m'])
    if len(out)==0:
        # A CMR swath can intersect the pilot while no accepted native geometry
        # remains. Keep a typed, joinable empty table and explicit zero coverage.
        from .coarse_join import CONTEXT_COLUMNS
        for key in CONTEXT_COLUMNS:
            if key not in out:out[key]=pd.Series(dtype=object)
    try:rms=float(summary['metadata']['geolocation_rms_error_m'])
    except (ValueError,KeyError):rms=None
    for row in frame.itertuples():
        p=from_wkb(row.native_footprint_wkb);margin=context_margin(p,row.view_zenith_deg,row.product,rms)
        support=p.buffer(margin) if margin is not None else None
        okay=bool(row.native_qa_valid and support is not None and support.is_valid and pilot.covers(support))
        part=partition(p,area,margin) if margin is not None else 'geolocation_uncertainty_or_view_unsupported'
        margins.append(np.nan if margin is None else margin);supports.append(None if support is None else support.wkb)
        eligible.append(okay);fit.append(okay and part=='fit_safe_geometry_only' and area['id']!='cabauw');partitions.append(part)
    out['context_support_wkb']=supports;out['context_support_margin_m']=margins
    out['context_eligible']=pd.Series(eligible,index=out.index,dtype=bool)
    out['context_fit_eligible']=pd.Series(fit,index=out.index,dtype=bool);out['context_support_partition']=partitions
    out['coarse_label_training_eligible']=False
    out['context_geometry_policy']=GEOMETRY_POLICY['version']
    return out


def point_context(frame,target,longitude,latitude,epsg,for_fitting=False,max_age_hours=24):
    """A shared coarse observation context, never a new fine-resolution label.

    Membership uses the native cell; expanded support is solely a leakage and
    uncertainty guard. Overlap ties prefer smaller view angle, then stable ID.
    """
    from pyproj import Transformer
    x,y=Transformer.from_crs(4326,epsg,always_xy=True).transform(longitude,latitude);point=Point(x,y)
    eligible=frame.loc[frame.context_fit_eligible if for_fitting else frame.context_eligible]
    candidates=[]
    for row in eligible.itertuples():
        interval={'granule_start_utc':row.granule_start_utc,'granule_end_utc':row.granule_end_utc}
        age=age_interval(interval,target,max_age_hours)
        if age['available'] and from_wkb(row.native_footprint_wkb).covers(point):
            candidates.append((row,age))
    if not candidates:return {'context_available':False,'context_missing_reason':'no_QA_and_geometry_eligible_past_native_cell'}
    row,age=min(candidates,key=lambda x:(-utc(x[0].granule_end_utc).value,x[0].view_zenith_deg,x[0].native_cell_id))
    return {'context_available':True,'context_native_cell_id':row.native_cell_id,'context_source_acquisition_id':row.acquisition_id,
        'context_lst_c':row.lst_c,'context_lst_error_k':row.lst_error_k,'context_product':row.product,
        'context_source_sha256':row.source_sha256,'context_footprint_area_m2':row.native_footprint_area_m2,
        'context_support_margin_m':row.context_support_margin_m,**age,
        'context_resolution_statement':'Shared native-area observation; no independent 100 m temperature evidence'}


def run(manifest_path,areas_path,output):
    source=json.loads(Path(manifest_path).read_text());areas={a['id']:a for a in json.loads(Path(areas_path).read_text())['areas']}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    signature={'native_manifest_sha256':sha(manifest_path),'context_code_sha256':sha(__file__),'areas_sha256':sha(areas_path),'geometry_policy':GEOMETRY_POLICY}
    if (output/'signature.json').exists():
        if json.loads((output/'signature.json').read_text())!=signature:raise ValueError('Context signature changed.')
    else:save(output/'signature.json',signature)
    records=[]
    for record in source['records']:
        if record['status']!='processed':continue
        if sha(record['table_path'])!=record['table_sha256']:raise ValueError('Native table checksum changed.')
        summary=json.loads(Path(record['summary_path']).read_text());area=areas[record['region_id']]
        frame=annotate(pd.read_parquet(record['table_path']),summary,area)
        path=output/(record['region_id']+'_'+record['stem']+'.parquet');frame.to_parquet(path,index=False)
        union=lambda flag:unary_union([from_wkb(x) for x in frame.loc[flag,'native_footprint_wkb']]).area/box(*area['extent_m']).area
        result={'region_id':record['region_id'],'stem':record['stem'],'product':record['product'],'phase':record['phase'],
            'rows':len(frame),'context_eligible_rows':int(frame.context_eligible.sum()),'context_fit_eligible_rows':int(frame.context_fit_eligible.sum()),
            'context_coverage_fraction':union(frame.context_eligible),'context_fit_coverage_fraction':union(frame.context_fit_eligible),
            'table_path':str(path.resolve()),'table_sha256':sha(path),'native_table_sha256':record['table_sha256']}
        records.append(result);print(json.dumps(result),flush=True)
    save(output/'manifest.json',{'signature':signature,'records':records,'coarse_label_training_eligible':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-manifest',required=True);p.add_argument('--areas',default='pilot/areas_resolved.json');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.native_manifest,a.areas,a.output)
