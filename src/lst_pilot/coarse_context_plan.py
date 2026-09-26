"""Anonymous target-specific past-swath planning; never downloads thermal data."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pandas as pd
from .coarse_inventory import PRODUCTS, normalize, fetch, companion, sha, save, utc, solar_phase, region_bbox
from .coarse_lst import age_interval

ASTER_COMPARISONS=[
    {'region_id':'greater_london','datetime_utc':'2021-09-08T11:12:05.659Z','scopes':['ASTER_same_swath_diagnostic_only']},
    {'region_id':'sioux_falls','datetime_utc':'2021-06-15T17:32:37.269Z','scopes':['ASTER_same_swath_diagnostic_only']},
]


def select_past(records,target):
    eligible=[r for r in records if age_interval(r,target)['available']]
    eligible.sort(key=lambda r:(utc(r['granule_end_utc']).value,int(r['production_tag']),r['cmr_revision']),reverse=True)
    # Latest complete native interval is chosen using metadata alone. Zero native
    # QA coverage remains missing; we do not substitute future/seasonal evidence.
    return eligible[0] if eligible else None


def plan(target_path,areas_path,output):
    output=Path(output)
    if (output/'context_metadata_plan.json').exists():raise FileExistsError('Metadata context plan already frozen.')
    source=json.loads(Path(target_path).read_text());areas={a['id']:a for a in json.loads(Path(areas_path).read_text())['areas']}
    targets=[x for x in source['targets'] if x['region_id'] in ('greater_london','sioux_falls')]
    def query(spec):
        target,product,diagnostic=spec;t=utc(target['datetime_utc']);area=areas[target['region_id']]
        if t.year not in (2021,2022,2023):raise ValueError('No new 2024/2025 context metadata in this pass.')
        key=f'{target["region_id"]}_{product}_{t.strftime("%Y%m%dT%H%M%S")}_{"diagnostic" if diagnostic else "past"}'
        path=output/'public_metadata'/f'{key}.json'
        begin=t if diagnostic else t-pd.Timedelta(hours=24)
        data=fetch({'collection_concept_id':PRODUCTS[product]['collection_id'],'temporal':begin.isoformat()+','+t.isoformat(),
                    'bounding_box':','.join(map(str,region_bbox(area)))},path)
        records=[normalize(i) for i in data['items']]
        if diagnostic:
            candidates=[r for r in records if utc(r['granule_start_utc'])<=t<=utc(r['granule_end_utc'])]
            selected=max(candidates,key=lambda r:(r['production_tag'],r['cmr_revision'])) if candidates else None
        else:selected=select_past(records,t)
        if selected:
            selected={**selected,'region_id':target['region_id'],'phase_at_pilot':solar_phase(selected['granule_start_utc'],selected['granule_end_utc'],area),
                'source_metadata_path':str(path.resolve()),'source_metadata_sha256':sha(path)}
        return {'target':target,'product':product,'same_swath_diagnostic_only':diagnostic,'metadata_candidates':len(records),
            'past24h_candidate_count':sum(age_interval(r,t)['available'] for r in records),'selected':selected,
            'temporal_join':age_interval(selected,t) if selected else {'available':False,'reason':'no_matching_source'},
            'thermal_download_enabled':False,'requires_new_2023_protocol':t.year==2023}
    tasks=[(t,p,False) for t in targets for p in ('MOD21','VNP21')]+[(t,'MOD21',True) for t in ASTER_COMPARISONS]
    with ThreadPoolExecutor(max_workers=4) as pool:joins=list(pool.map(query,tasks))
    unique={}
    for join in joins:
        if join['selected']:
            r=join['selected'];unique[(r['region_id'],r['stem'])]=r
    # Public full-resolution companions are inventoried before any size budget is proposed.
    for r in unique.values():
        if r['product']=='MOD21':r['geolocation_companion']=companion(r,output)
    rows=list(unique.values());fit=[r for r in rows if utc(r['granule_start_utc']).year<=2022]
    total=lambda records:sum(r['estimated_source_bytes']+r.get('geolocation_companion',{}).get('estimated_source_bytes',0) for r in records)
    result={'version':'target-native-context-metadata-v1','reference_targets_sha256':sha(target_path),'areas_sha256':sha(areas_path),
        'planner_sha256':sha(__file__),'metadata_only':True,'protected_downloads_enabled':False,
        'selection':'Latest fully completed granule interval in preceding 24 hours for each target/sensor; no thermal QA selection',
        'zero_qa_coverage_policy':'Missing context, never future or reference-year substitution',
        'aster_diagnostic_policy':'Overlapping same-Terra granule only; not a causal feature when source end is later than target',
        'targets':len(targets),'joins':joins,'unique_sources':rows,'unique_source_count':len(rows),
        'estimated_all_sources_bytes':total(rows),'estimated_2021_2022_sources_bytes':total(fit),
        'sources_2021_2022':len(fit),'sources_2023_protocol_pending':len(rows)-len(fit)}
    save(output/'context_metadata_plan.json',result)
    print(json.dumps({k:result[k] for k in ['targets','unique_source_count','estimated_all_sources_bytes','estimated_2021_2022_sources_bytes','sources_2021_2022','sources_2023_protocol_pending']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--targets',required=True);p.add_argument('--areas',default='pilot/areas_resolved.json');p.add_argument('--output',required=True)
    a=p.parse_args();plan(a.targets,a.areas,a.output)
