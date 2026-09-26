"""Immutable metadata-only sample batches and serialized coarse phase setup."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from .coarse_inventory import sha,save
from .coarse_context_plan import plan
from .coarse_download import H_PROTOCOL_SHA256


def metadata(samples,expected_sha,output,areas):
    samples=Path(samples);output=Path(output)
    if sha(samples)!=expected_sha:raise ValueError('Parent-frozen sample source checksum changed.')
    columns=['sample_id','region_id','datetime_utc','latitude','longitude','epsg']
    frame=pd.read_parquet(samples,columns=columns)
    if frame.sample_id.duplicated().any():raise ValueError('Duplicate source sample IDs.')
    if not set(frame.region_id)<={'greater_london','sioux_falls'}:raise ValueError('Unplanned pilot; do not silently discard new target regions.')
    if not pd.to_datetime(frame.datetime_utc,utc=True).dt.year.isin([2021,2022,2023]).all():raise ValueError('No new 2024/2025 target sources.')
    target=output/'reference_targets.json';output.mkdir(parents=True,exist_ok=True)
    if target.exists():
        before=json.loads(target.read_text())
        if before['input_sha256']!=expected_sha or sha(output/'samples.parquet')!=before['sample_metadata_sha256']:
            raise ValueError('Existing metadata batch signature changed.')
    else:
        frame.to_parquet(output/'samples.parquet',index=False)
        times=frame[['region_id','datetime_utc']].drop_duplicates().copy()
        times['datetime_utc']=times.datetime_utc.map(lambda t:pd.Timestamp(t).isoformat())
        times['scopes']=[['new_fine_source_metadata_only']]*len(times)
        save(target,{'targets':times.to_dict('records'),'input_path':str(samples.resolve()),'input_sha256':expected_sha,
                     'input_columns_read':columns,'sample_metadata_sha256':sha(output/'samples.parquet')})
    if not (output/'plan/context_metadata_plan.json').exists():plan(target,areas,output/'plan')


def freeze(metadata_plan,output,shared,protocol):
    metadata_plan=Path(metadata_plan);output=Path(output);shared=Path(shared);protocol=Path(protocol).resolve()
    if sha(protocol)!=H_PROTOCOL_SHA256:raise ValueError('Exact predeclared H protocol required.')
    data=json.loads(metadata_plan.read_text())
    if not data['metadata_only'] or data['protected_downloads_enabled']:raise ValueError('Require independently frozen metadata selection.')
    if (output/'engineering_plan.json').exists():raise FileExistsError('Acquisition plan already frozen.')
    plan={'version':'coarse-post-H-source-batch-v1','metadata_source_path':str(metadata_plan.resolve()),'metadata_source_sha256':sha(metadata_plan),
        'H_protocol':{'path':str(protocol),'sha256':sha(protocol)},'areas_sha256':data['areas_sha256'],
        'max_total_protected_bytes':24*1024**3,'max_requests':1800,'max_download_workers':2,
        'max_file_bytes':{'MOD21':32*1024**2,'MOD03':64*1024**2,'VNP21':192*1024**2},'thermal_years':[2021,2022,2023],
        'estimated_source_bytes':data['estimated_all_sources_bytes'],'shared_budget_path':str((shared/'transfer_budget.json').resolve()),
        'working_storage_free_floor_bytes':150*1024**3,'download_serialization':'One process owns shared download.lock before loading the cumulative ledger',
        'records':data['unique_sources']}
    # Bound projected uncached payload by metadata before the hard streaming cap.
    assets=[asset for r in plan['records'] for asset in ([r,r['geolocation_companion']] if 'geolocation_companion' in r else [r])]
    needed=sum(a['estimated_source_bytes'] for a in assets if not any((shared/'assets').glob(a['stem']+'.*json')))
    ledger=json.loads((shared/'transfer_budget.json').read_text())
    if ledger['bytes_received']+needed>plan['max_total_protected_bytes']:raise ValueError('Projected batch exceeds cumulative coarse transfer cap.')
    import shutil
    if shutil.disk_usage(shared).free-needed<plan['working_storage_free_floor_bytes']:raise ValueError('Projected batch violates the 150 GiB free-space floor.')
    output.mkdir(parents=True,exist_ok=True);save(output/'engineering_plan.json',plan)
    (output/'assets').symlink_to((shared/'assets').resolve(),target_is_directory=True)
    (output/'transfer_budget.json').symlink_to((shared/'transfer_budget.json').resolve())
    print(json.dumps({'plan':str(output),'sources':len(plan['records']),'estimated_uncached_bytes':needed,'plan_sha256':sha(output/'engineering_plan.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    m=sub.add_parser('metadata');m.add_argument('--samples',required=True);m.add_argument('--expected-sha',required=True);m.add_argument('--output',required=True);m.add_argument('--areas',default='pilot/areas_resolved.json')
    f=sub.add_parser('freeze');f.add_argument('--metadata-plan',required=True);f.add_argument('--output',required=True);f.add_argument('--shared-root',required=True);f.add_argument('--protocol',required=True)
    a=p.parse_args()
    if a.mode=='metadata':metadata(a.samples,a.expected_sha,a.output,a.areas)
    else:freeze(a.metadata_plan,a.output,a.shared_root,a.protocol)
