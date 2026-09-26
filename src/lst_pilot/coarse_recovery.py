"""Frozen earlier-pass recovery selected from metadata and zero coverage only."""
from __future__ import annotations
import argparse
from collections import defaultdict,deque,Counter
import json
from pathlib import Path
from .coarse_inventory import normalize,identity,sha,save,utc,companion,solar_phase
from .coarse_lst import age_interval


def balanced_select(candidates,maximum=40):
    """Rotate pilot/fit-eval, then seasons, then year/product within season."""
    buckets=defaultdict(lambda:defaultdict(lambda:defaultdict(deque)))
    for c in sorted(candidates,key=lambda c:(c['target']['datetime_utc'],c['candidate_rank'],c['record']['stem'])):
        target=c['target'];t=utc(target['datetime_utc']);stage='fit' if t.year<=2022 else 'eval'
        buckets[(target['region_id'],stage)][(t.month-1)//3][(t.year,c['record']['product'])].append(c)
    outer=[(r,s) for s in ('fit','eval') for r in ('greater_london','sioux_falls')]
    season_cursor=defaultdict(int);inner_cursor=defaultdict(int);selected=[];used=set();target_counts=Counter()
    while len(selected)<maximum:
        progress=False
        for group in outer:
            if len(selected)>=maximum:break
            # Each ticket visits a season, rotating even when a season is empty.
            chosen=None
            for _ in range(4):
                season=season_cursor[group]%4;season_cursor[group]+=1
                options=buckets[group][season];available=sorted(options)
                years=sorted({k[0] for k in available});products=sorted({k[1] for k in available})
                keys=[]
                # Alternate years immediately, instead of exhausting both
                # products of the earliest year before reaching the next year.
                for step in range(len(years)*len(products)):
                    k=(years[(step+season)%len(years)],products[(step//len(years)+step%len(years)+season)%len(products)])
                    if k in options and k not in keys:keys.append(k)
                keys.extend(k for k in available if k not in keys)
                if not keys:continue
                for _ in range(len(keys)):
                    k=keys[inner_cursor[(group,season)]%len(keys)];inner_cursor[(group,season)]+=1
                    while options[k]:
                        c=options[k].popleft();physical=identity(c['record']['stem'])['acquisition_key'];target_key=(c['target']['region_id'],c['target']['datetime_utc'],c['record']['product'])
                        if physical in used or target_counts[target_key]>=2:continue
                        chosen=c;used.add(physical);target_counts[target_key]+=1;break
                    if chosen:break
                if chosen:break
            if chosen:selected.append(chosen);progress=True
        if not progress:break
    return selected


def plan(metadata_paths,context_paths,areas_path,shared,output,addendum):
    shared=Path(shared);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'context_metadata_plan.json').exists():raise FileExistsError('Recovery queue already frozen.')
    areas={a['id']:a for a in json.loads(Path(areas_path).read_text())['areas']};coverage={};provenance=[]
    for p in context_paths:
        d=json.loads(Path(p).read_text());provenance.append({'path':str(Path(p).resolve()),'sha256':sha(p)})
        for r in d['records']:
            if r.get('status','processed')=='processed':coverage[(r['region_id'],r['stem'])]=r['context_eligible_rows']
    collected=set()
    for path in (shared/'assets').iterdir():
        try:collected.add(identity(path.name)['acquisition_key'])
        except ValueError:pass
    candidates=[];zero_targets=[];seen=set()
    for metadata in metadata_paths:
        data=json.loads(Path(metadata).read_text())
        for join in data['joins']:
            if join['same_swath_diagnostic_only'] or not join['selected']:continue
            original=join['selected'];target=join['target'];key=(target['region_id'],target['datetime_utc'],join['product'])
            if key in seen or coverage.get((original['region_id'],original['stem']))!=0:continue
            seen.add(key);zero_targets.append({'target':target,'product':join['product'],'initial_stem':original['stem']})
            source=Path(original['source_metadata_path'])
            if sha(source)!=original['source_metadata_sha256']:raise ValueError('Frozen public metadata checksum changed.')
            records=[normalize(i) for i in json.loads(source.read_text())['items']]
            earlier=[r for r in records if r['product']==join['product'] and age_interval(r,target['datetime_utc'])['available']
                and utc(r['granule_end_utc'])<utc(original['granule_end_utc']) and r['acquisition_key'] not in collected]
            earlier.sort(key=lambda r:(r['granule_end_utc'],r['production_tag'],r['cmr_revision']),reverse=True)
            # Exactly the first two earlier metadata-ranked physical passes may
            # enter the queue; no preview temperature/cloud selection occurs.
            unique=[];seen_physical=set()
            for r in earlier:
                if r['acquisition_key'] in seen_physical:continue
                seen_physical.add(r['acquisition_key']);unique.append(r)
            for rank,r in enumerate(unique[:2],1):
                record={**r,'region_id':target['region_id'],'source_metadata_path':str(source.resolve()),'source_metadata_sha256':sha(source),
                    'phase_at_pilot':solar_phase(r['granule_start_utc'],r['granule_end_utc'],areas[target['region_id']])}
                candidates.append({'target':target,'initial_stem':original['stem'],'candidate_rank':rank,'record':record})
    selected=balanced_select(candidates,40)
    for s in selected:
        r=s['record']
        if r['product']=='MOD21':r['geolocation_companion']=companion(r,output)
    records=[s['record'] for s in selected];total=sum(r['estimated_source_bytes']+r.get('geolocation_companion',{}).get('estimated_source_bytes',0) for r in records)
    audit={'version':'coarse-zero-coverage-recovery-v1','metadata_only':True,'protected_downloads_enabled':False,
        'source_code_sha256':sha(__file__),'areas_sha256':sha(areas_path),'recovery_addendum_path':str(Path(addendum).resolve()),'recovery_addendum_sha256':sha(addendum),
        'initial_context_manifests':provenance,'target_metadata_manifests':[{'path':str(Path(p).resolve()),'sha256':sha(p)} for p in metadata_paths],
        'initial_zero_context_target_products':zero_targets,'candidate_count':len(candidates),'max_additional_per_target_product':2,'max_new_thermal_sources':40,
        'selection':'Round-robin London/Sioux fit/eval, seasons, then year/product; chronological target order and earlier-pass rank; no residual or temperature-magnitude selection',
        'selected_target_associations':selected,'unique_sources':records,'unique_source_count':len(records),'estimated_all_sources_bytes':total,
        'protected_cap_scope':'Existing24GiB cumulative coarse ledger, including all phases','unchanged':'Native QA, expanded geography,24h causality and H numerical specification'}
    save(output/'context_metadata_plan.json',audit)
    print(json.dumps({'selected_sources':len(records),'estimated_bytes':total,'zero_target_products':len(zero_targets),'candidates':len(candidates),
        'strata':dict(Counter(f'{s["target"]["region_id"]}:{utc(s["target"]["datetime_utc"]).year}:Q{(utc(s["target"]["datetime_utc"]).month-1)//3+1}' for s in selected))}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata-plans',nargs='+',required=True);p.add_argument('--context-manifests',nargs='+',required=True);p.add_argument('--areas',default='pilot/areas_resolved.json');p.add_argument('--shared-root',required=True);p.add_argument('--output',required=True);p.add_argument('--addendum',required=True)
    a=p.parse_args();plan(a.metadata_plans,a.context_manifests,a.areas,a.shared_root,a.output,a.addendum)
