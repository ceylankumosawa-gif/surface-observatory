"""Freeze disjoint completed-source sample batches for the expanded campaign."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import ecostress as eco, highres_sample as sampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--exclude-snapshots',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New immutable sample output required.')
    plan_bytes=(args.source/'plan.json').read_bytes();manifest_bytes=(args.source/'manifest.json').read_bytes()
    plan,manifest=json.loads(plan_bytes),json.loads(manifest_bytes)
    if manifest['plan_sha256']!=hashlib.sha256(plan_bytes).hexdigest():raise ValueError('Plan mismatch.')
    excluded=set();prior_hashes={}
    for path in args.exclude_snapshots:
        path=path/'manifest_snapshot.json'
        prior_hashes[str(path)]=eco.digest(path)
        excluded.update(r['granule_id'] for r in json.loads(path.read_text())['records'])
    lookup={r['granule_concept_id']:r for rows in plan['queues'].values() for r in rows}
    frames,audit=[],[]
    for record in manifest['records']:
        if record['granule_id'] in excluded or not record.get('source_screen_pass') or 'native_fit_support' not in record:continue
        planned={**lookup[record['granule_id']], 'cmr_revision_id':record['cmr_meta']['revision-id']}
        row={**record,'status':'engineering_qa_complete'}
        frame,windows=sampler.sample_record(row,planned,plan['pilot_areas'][record['pilot_id']])
        if not frame.empty:
            frame['fresh_2023']=bool(planned['fresh_2023'])
            frame['freshness_audit_sha256']=plan['registry_sha256']
            frame['obstruction_status']=record['obstruction']['status']
            frames.append(frame)
        audit.append({'granule_id':record['granule_id'],'pilot_id':record['pilot_id'],'date':record['utc_date'],
            'phase':record['actual_phase'],'temporal_split':record['temporal_split'],'rows':len(frame),
            'qualifying_independent_date':record['qualifying_independent_date'],'windows':windows})
    args.output.mkdir(parents=True)
    (args.output/'plan_snapshot.json').write_bytes(plan_bytes)
    (args.output/'manifest_snapshot.json').write_bytes(manifest_bytes)
    result={'version':'ecostress-expanded-samples-20260910-v1','plan_sha256':hashlib.sha256(plan_bytes).hexdigest(),
        'source_snapshot_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'prior_snapshot_hashes':prior_hashes,
        'sampler_sha256':eco.digest(__file__),'sampling_helper_sha256':eco.digest(sampler.__file__),
        'records':audit,'rows':0,'outputs':{}}
    if frames:
        data=pd.concat(frames,ignore_index=True)
        if data.sample_id.duplicated().any():raise ValueError('Repeated physical sample identity in frozen source batch.')
        path=args.output/'source_screened_samples.parquet';data.to_parquet(path,index=False)
        result.update(rows=len(data),pilot_dates=len(data.assign(utc_date=data.datetime_utc.dt.strftime('%Y-%m-%d'))[['region_id','utc_date']].drop_duplicates()),
            phase_rows=data.day_night.value_counts().to_dict(),year_rows=data.datetime_utc.dt.year.value_counts().to_dict())
        result['outputs']['source_screened_samples']={'path':str(path),'sha256':eco.digest(path),'rows':len(data)}
    eco.save_json(args.output/'sampling_manifest.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
