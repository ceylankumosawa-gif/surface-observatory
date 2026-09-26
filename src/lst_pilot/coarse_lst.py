"""Native MOD21.061/VNP21.002 engineering evidence, never 100 m labels.

Geolocation-center quadrilaterals estimate native cell geometry, not the optical
point-spread function. A half-nadir-pixel margin is an engineering exclusion,
not a claimed geolocation confidence interval. All rows stay ineligible for fit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from .coarse_inventory import PRODUCTS, sha, save, utc, region_bbox

QA_RULES={
    'mandatory_codes':[0], 'data_quality_codes':[0], 'cloud_codes':[0],
    'lst_error_class_codes':[2,3], 'lst_error_max_k':1.5, 'view_zenith_max_deg':30,
    'land_mask_value':0, 'plausible_lst_kelvin':[150,400],
    'mod03_gflags_value':0, 'scan_edge_detector_rows_excluded':True,
    'native_support_guard_margin_fraction_nadir_pixel':0.5,
    'native_footprints_are_psf_estimates':False,
    'fit_enabled':False,
}
SOURCES={
    'modis_guide':'https://lpdaac.usgs.gov/documents/1398/MOD21_User_Guide_V61.pdf',
    'viirs_guide':'https://lpdaac.usgs.gov/documents/1662/VNP21_User_Guide_V2.pdf',
    'mod03_spec':'https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MOD03_c61',
}


def text(value):
    a=np.asarray(value)
    if a.size==1:value=a.reshape(-1)[0]
    return value.decode() if isinstance(value,bytes) else str(value)


def attr(attributes, name):
    key=next((k for k in attributes if k.lower()==name.lower()),None)
    if key is not None:return text(attributes[key]).strip('"')
    for value in attributes.values():
        if not isinstance(value,(str,bytes)):continue
        m=re.search(r'\bOBJECT\s*=\s*'+re.escape(name)+r'\s+(.*?)\bEND_OBJECT\s*=\s*'+re.escape(name),text(value),re.S|re.I)
        if m:
            found=re.search(r'\bVALUE\s*=\s*(.*)',m.group(1),re.S)
            if found:return found.group(1).strip().strip('"')
    raise ValueError('Required native metadata absent: '+name)


def validate_interval(attributes,record):
    beginning=attr(attributes,'RangeBeginningDate')+'T'+attr(attributes,'RangeBeginningTime')+'Z'
    ending=attr(attributes,'RangeEndingDate')+'T'+attr(attributes,'RangeEndingTime')+'Z'
    if utc(beginning)!=utc(record['granule_start_utc']) or utc(ending)!=utc(record['granule_end_utc']):
        raise ValueError('Native file observation interval differs from frozen CMR interval.')
    if attr(attributes,'LocalGranuleID')!=record['stem']+PRODUCTS[record['product']]['suffix']:
        raise ValueError('Native LocalGranuleID mismatch.')


def validate_scale(attributes,scale,fill=None):
    scale_value=next((attributes[k] for k in ('scale_factor','_Scale') if k in attributes),None)
    offset=next((attributes[k] for k in ('add_offset','_Offset') if k in attributes),None)
    if scale_value is None or offset is None or not np.isclose(float(np.asarray(scale_value).item()),scale) or float(np.asarray(offset).item())!=0:
        raise ValueError('Native scaled variable metadata differs from pinned product specification.')
    if fill is not None and float(np.asarray(attributes['_FillValue']).item())!=fill:
        raise ValueError('Native fill value differs from pinned specification.')


def decode(raw):
    q=np.asarray(raw['QC'],dtype=np.uint16)
    result={name:((q>>shift)&3) for name,shift in [('qc_mandatory',0),('qc_data',2),('qc_cloud',4),
         ('qc_convergence',6),('qc_opacity',8),('qc_emissivity_mmd',10),('qc_emissivity_error',12),('qc_lst_error',14)]}
    lst=np.asarray(raw['LST'],dtype=float)*.02;err=np.asarray(raw['LST_err'],dtype=float)*.04
    view=np.asarray(raw['View_angle'],dtype=float)*.5
    result.update(lst_c=np.where(raw['LST']!=0,lst-273.15,np.nan),lst_error_k=np.where(raw['LST_err']!=0,err,np.nan),view_zenith_deg=view)
    gates={'qa_mandatory_ok':result['qc_mandatory']==0,'qa_data_ok':result['qc_data']==0,
        'qa_cloud_clear':result['qc_cloud']==0,'qa_error_class_ok':result['qc_lst_error']>=2,
        'qa_error_ok':(raw['LST_err']>0)&(err<=1.5),'qa_view_ok':(raw['View_angle']<=60),
        'qa_land':raw['oceanpix']==0,'qa_lst_range':(raw['LST']!=0)&(lst>=150)&(lst<=400)}
    result.update(gates);result['native_qa_valid']=np.logical_and.reduce(list(gates.values()))
    return result


def age_interval(record,target,max_age_hours=24):
    target=utc(target);start=utc(record['granule_start_utc']);end=utc(record['granule_end_utc'])
    if start>end:raise ValueError('Invalid source interval.')
    if end>target:return {'available':False,'reason':'source_interval_has_future_observations'}
    oldest=(target-start).total_seconds()/3600;newest=(target-end).total_seconds()/3600
    return {'available':oldest<=max_age_hours,'reason':'' if oldest<=max_age_hours else 'source_interval_older_than_window',
            'age_min_hours':newest,'age_max_hours':oldest,'source_available_at_utc':end.isoformat(),
            'causality':'Observation-valid-time only; retrospective production latency is not as-issued availability'}


def native_shape(thermal,latitude,longitude):
    if thermal.shape!=latitude.shape or latitude.shape!=longitude.shape or thermal.ndim!=2:
        raise ValueError('Require full native-resolution matching geolocation; coarse/tie-point interpolation is prohibited.')


def footprint(x,y,row,col,scan_rows,nominal):
    # Adjacent rows at a scan boundary need a scan-aware sensor model. Exclude
    # those detector rows rather than average centers from different scans.
    if row%scan_rows in (0,scan_rows-1) or row<1 or col<1 or row>=x.shape[0]-1 or col>=x.shape[1]-1:return None
    xx=x[row-1:row+2,col-1:col+2];yy=y[row-1:row+2,col-1:col+2]
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():return None
    corners=[(float(xx[a:a+2,b:b+2].mean()),float(yy[a:a+2,b:b+2].mean())) for a,b in [(0,0),(0,1),(1,1),(1,0)]]
    p=Polygon(corners);edges=np.linalg.norm(np.diff(np.asarray(corners+[corners[0]]),axis=0),axis=1)
    if not p.is_valid or p.area<.25*nominal**2 or p.area>4*nominal**2 or np.min(edges)<.4*nominal or np.max(edges)>2.5*nominal:return None
    if not p.covers(Polygon(corners).centroid) or abs(p.convex_hull.area-p.area)>1e-5:return None
    return p


def reserved_polygons(area):
    if area['id'] not in ('greater_london','sioux_falls'):return []
    left,bottom,right,top=area['extent_m']
    return [box(left+c*10000,bottom+r*10000,min(right,left+(c+1)*10000),min(top,bottom+(r+1)*10000))
        for r in range(int(np.ceil((top-bottom)/10000))) for c in range(int(np.ceil((right-left)/10000))) if (r+2*c)%5==0]


def partition(polygon,area,margin_m):
    if area['id']=='cabauw':return 'cabauw_reference_only'
    support=polygon.buffer(margin_m)
    if not box(*area['extent_m']).covers(support):return 'pilot_boundary'
    reserved=reserved_polygons(area)
    if any(polygon.intersects(p) for p in reserved):return 'spatial_holdout'
    if any(support.distance(p)<=1000 for p in reserved):return 'holdout_buffer'
    return 'fit_safe_geometry_only'


def read_source(record,thermal_path,geo_path=None):
    names=['LST','LST_err','QC','View_angle','oceanpix']
    raw={};field_attrs={};geo_attrs={}
    if record['product']=='VNP21':
        import h5py
        with h5py.File(thermal_path) as f:
            attributes=dict(f.attrs);datasets={}
            def collect(n,o):
                if isinstance(o,h5py.Dataset):datasets.setdefault(n.split('/')[-1].lower(),[]).append(n)
            f.visititems(collect)
            for name in names+['latitude','longitude']:
                matches=datasets.get(name.lower(),[])
                if len(matches)!=1:raise ValueError('Native variable missing/ambiguous: '+name)
                variable=f[matches[0]];raw[name]=variable[:];field_attrs[name]=dict(variable.attrs)
        latitude=raw.pop('latitude');longitude=raw.pop('longitude');geo_valid=np.ones(latitude.shape,bool)
        geolocation_method='VNP21 native 750 m latitude/longitude from its recorded VNP03MOD input'
    else:
        from pyhdf.SD import SD,SDC
        if geo_path is None:raise ValueError('MOD21 requires original full-resolution MOD03 companion.')
        f=SD(str(thermal_path),SDC.READ)
        try:
            attributes=f.attributes()
            for name in names:
                s=f.select(name);raw[name]=s.get();field_attrs[name]=s.attributes()
        finally:f.end()
        companion=record['geolocation_companion']
        if companion['granule_start_utc']!=record['granule_start_utc'] or companion['granule_end_utc']!=record['granule_end_utc']:
            raise ValueError('MOD03 companion acquisition interval mismatch.')
        # Exact revision, not merely the same orbit/time, must be the actual input.
        input_pointer=attr(attributes,'InputPointer')
        if companion['stem']+'.hdf' not in input_pointer:
            raise ValueError('Selected MOD03 revision is not named by MOD21 InputPointer.')
        f=SD(str(geo_path),SDC.READ)
        try:
            geo_attrs=f.attributes();validate_interval(geo_attrs,companion)
            latitude=f.select('Latitude').get();longitude=f.select('Longitude').get();gflags=f.select('gflags').get()
        finally:f.end()
        raw['geolocation_flags']=gflags;geo_valid=gflags==0
        geolocation_method='MOD03 exact input acquisition/revision, full native 1 km centers; sparse MOD21 tie points ignored'
    validate_interval(attributes,record)
    validate_scale(field_attrs['LST'],.02,0);validate_scale(field_attrs['LST_err'],.04,0);validate_scale(field_attrs['View_angle'],.5)
    native_shape(raw['LST'],latitude,longitude)
    if any(v.shape!=latitude.shape for v in raw.values()):raise ValueError('Native QA/thermal grid mismatch.')
    geo_valid&=np.isfinite(latitude)&np.isfinite(longitude)&(abs(latitude)<=90)&(abs(longitude)<=180)
    raw['latitude']=latitude;raw['longitude']=longitude;raw['geolocation_valid']=geo_valid
    return raw,{'geolocation_method':geolocation_method,'native_shape':list(latitude.shape),
        'input_pointer':attr(attributes,'InputPointer'),'production_datetime':attr(attributes,'ProductionDateTime'),
        'native_affine':None,'native_crs':'EPSG:4326 geolocation arrays; non-affine swath',
        'geolocation_rms_error_m':text(geo_attrs.get('GEO_EST_RMS_ERROR','not provided'))}


def process(download,area,output):
    record=download['record'];source=download['thermal_asset'];geo=download.get('geolocation_asset')
    for asset in [source]+([geo] if geo else []):
        if sha(asset['path'])!=asset['sha256']:raise ValueError('Native source SHA changed.')
    raw,metadata=read_source(record,source['path'],geo['path'] if geo else None)
    west,south,east,north=region_bbox(area)
    candidates=raw['geolocation_valid']&(raw['longitude']>=west-.1)&(raw['longitude']<=east+.1)&(raw['latitude']>=south-.1)&(raw['latitude']<=north+.1)
    rr,cc=np.where(candidates);nominal=PRODUCTS[record['product']]['native_m'];scan_rows=10 if record['product']=='MOD21' else 16
    transform=Transformer.from_crs(4326,area['epsg'],always_xy=True)
    # Transform only a contiguous geographic subset with its native neighbors.
    pilot=box(*area['extent_m']);rows=[];polys=[];accepted=[];invalid_geometry=0
    if len(rr):
        r0=max(0,int(rr.min())-1);r1=min(raw['latitude'].shape[0],int(rr.max())+2);c0=max(0,int(cc.min())-1);c1=min(raw['latitude'].shape[1],int(cc.max())+2)
        x,y=transform.transform(raw['longitude'][r0:r1,c0:c1],raw['latitude'][r0:r1,c0:c1])
        # Keep global detector indices in the scan gate, and use local indices only for array access.
        decoded=decode({k:raw[k][rr,cc] for k in ['LST','LST_err','QC','View_angle','oceanpix']})
        for i,(r,c) in enumerate(zip(rr,cc)):
            if r%scan_rows in (0,scan_rows-1):invalid_geometry+=1;continue
            # Local offset may change modulo; footprint scan check is handled above.
            p=footprint(x,y,int(r-r0),int(c-c0),10**9,nominal)
            if p is None:invalid_geometry+=1;continue
            if not p.intersects(pilot):continue
            neighborhood=raw['geolocation_valid'][r-1:r+2,c-1:c+2]
            if neighborhood.shape!=(3,3) or not neighborhood.all():invalid_geometry+=1;continue
            valid=bool(decoded['native_qa_valid'][i]);polys.append(p.intersection(pilot))
            if valid:accepted.append(polys[-1])
            rows.append({'native_cell_id':record['stem']+f':r{r}:c{c}','region_id':area['id'],
                'product':record['product'],'version':record['version'],'sensor':PRODUCTS[record['product']]['sensor'],
                'acquisition_id':record['acquisition_key'],'granule_id':record['granule_id'],'cmr_revision':record['cmr_revision'],
                'granule_start_utc':record['granule_start_utc'],'granule_end_utc':record['granule_end_utc'],
                'source_available_at_utc':record['granule_end_utc'],'native_row':int(r),'native_col':int(c),
                'latitude':float(raw['latitude'][r,c]),'longitude':float(raw['longitude'][r,c]),'footprint_epsg':area['epsg'],
                'native_footprint_wkb':p.wkb,'native_footprint_area_m2':p.area,'pilot_overlap_fraction':p.intersection(pilot).area/p.area,
                'support_guard_margin_m':nominal*.5,'support_partition':partition(p,area,nominal*.5),
                'native_qa_valid':valid,'native_valid_fraction':float(valid),'native_valid_fraction_definition':'Whole native retrieval gate, not measured subpixel cloud fraction',
                **{k:(v[i].item() if hasattr(v[i],'item') else v[i]) for k,v in decoded.items()},
                'qc_raw':int(raw['QC'][r,c]),'oceanpix_raw':int(raw['oceanpix'][r,c]),
                'source_sha256':source['sha256'],'source_bytes':source['bytes'],
                'geolocation_source_sha256':geo['sha256'] if geo else source['sha256'],
                'training_eligible':False,'purpose':'native context or aggregate calibration engineering only',
                'phase_at_pilot':record['phase_at_pilot']})
    output=Path(output);output.mkdir(parents=True,exist_ok=True);table=output/'native_cells.parquet'
    frame=pd.DataFrame(rows);frame.to_parquet(table,index=False)
    covered=unary_union(polys).area if polys else 0;valid_covered=unary_union(accepted).area if accepted else 0
    summary={'record':record,'source_assets':{'thermal':source,'geolocation':geo},'metadata':metadata,'qa_rules':QA_RULES,'official_sources':SOURCES,
        'footprint_method':'Quadrilaterals from adjacent full-native geolocation centers within one scan; not physical PSF support',
        'footprint_limitations':'Native optical footprint and registration uncertainty require independent validation before any fitting use',
        'candidate_geolocation_centers':len(rr),'geometry_rejected_candidates':invalid_geometry,'rows':len(frame),
        'native_qa_valid_rows':int(frame.native_qa_valid.sum()) if len(frame) else 0,
        'support_partition_counts':dict(Counter(frame.support_partition)) if len(frame) else {},
        'valid_support_partition_counts':dict(Counter(frame.loc[frame.native_qa_valid,'support_partition'])) if len(frame) else {},
        'qa_gate_failures':{k:int((~frame[k]).sum()) for k in frame if k.startswith('qa_')},
        'pilot_native_geometry_coverage_fraction':covered/pilot.area,'pilot_valid_coverage_fraction':valid_covered/pilot.area,
        'valid_fraction_of_geometric_coverage':valid_covered/covered if covered else None,
        'native_uncertainty_kind':'Product per-pixel LST_err in Kelvin; not total bias or registration uncertainty',
        'table_path':str(table.resolve()),'table_sha256':sha(table),'reader_sha256':sha(__file__),'training_eligible':False}
    save(output/'summary.json',summary);return summary


def run(root,areas_path):
    root=Path(root);source=root/'download_manifest.json';downloads=json.loads(source.read_text())
    areas={a['id']:a for a in json.loads(Path(areas_path).read_text())['areas']}
    signature={'download_manifest_sha256':sha(source),'reader_sha256':sha(__file__),'areas_sha256':sha(areas_path),'qa_rules':QA_RULES}
    sigpath=root/'native_reader_signature.json'
    if sigpath.exists():
        if json.loads(sigpath.read_text())!=signature:raise ValueError('Frozen native reader signature changed.')
    else:save(sigpath,signature)
    results=[]
    for item in downloads['records']:
        if item['status']!='downloaded':continue
        record=item['record'];output=root/'native'/record['region_id']/record['stem']
        try:
            result=process(item,areas[record['region_id']],output)
            results.append({'status':'processed','region_id':record['region_id'],'product':record['product'],'stem':record['stem'],
                'phase':record['phase_at_pilot'],'rows':result['rows'],'valid_rows':result['native_qa_valid_rows'],
                'pilot_valid_coverage_fraction':result['pilot_valid_coverage_fraction'],'summary_path':str((output/'summary.json').resolve()),
                'table_path':result['table_path'],'table_sha256':result['table_sha256']})
        except ValueError as exc:results.append({'status':'rejected','stem':record['stem'],'error':str(exc)})
        print(json.dumps(results[-1]),flush=True)
    save(root/'native_manifest.json',{'signature':signature,'records':results,'training_eligible':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--areas',default='pilot/areas_resolved.json')
    a=p.parse_args();run(a.root,a.areas)
