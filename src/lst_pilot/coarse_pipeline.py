"""Two-worker immutable native/context checkpoints alongside bounded downloads."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
from pathlib import Path
import time
import pandas as pd
from shapely import from_wkb
from shapely.geometry import box
from shapely.ops import unary_union
from .coarse_inventory import sha,save
from .coarse_lst import process
from .coarse_context import annotate,GEOMETRY_POLICY


def job(args):
    download,area,root=args;root=Path(root);record=download['record'];key=record['region_id']+'_'+record['stem']
    checkpoint=root/'processing_checkpoints'/f'{key}.json';native_dir=root/'native'/record['region_id']/record['stem']
    try:
        summary_path=native_dir/'summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            if sha(summary['table_path'])!=summary['table_sha256'] or summary['reader_sha256']!=sha(importlib.import_module('lst_pilot.coarse_lst').__file__):
                raise ValueError('Native checkpoint signature/checksum changed.')
        else:summary=process(download,area,native_dir)
        context=annotate(pd.read_parquet(summary['table_path']),summary,area)
        path=root/'context'/f'{key}.parquet';path.parent.mkdir(parents=True,exist_ok=True);context.to_parquet(path,index=False)
        union=lambda mask:unary_union([from_wkb(x) for x in context.loc[mask,'native_footprint_wkb']]).area/box(*area['extent_m']).area
        result={'status':'processed','region_id':area['id'],'product':record['product'],'stem':record['stem'],'phase':record['phase_at_pilot'],
            'rows':len(context),'native_valid_rows':int(context.native_qa_valid.sum()) if 'native_qa_valid' in context else 0,
            'context_eligible_rows':int(context.context_eligible.sum()),'context_fit_eligible_rows':int(context.context_fit_eligible.sum()),
            'context_coverage_fraction':union(context.context_eligible),'context_fit_coverage_fraction':union(context.context_fit_eligible),
            'table_path':str(path.resolve()),'table_sha256':sha(path),'native_table_sha256':summary['table_sha256'],
            'native_table_path':summary['table_path'],'native_summary_path':str(summary_path.resolve())}
    except ValueError as exc:result={'status':'rejected','region_id':area['id'],'product':record['product'],'stem':record['stem'],'error':str(exc)}
    save(checkpoint,result);return result


def run(root,areas_path,workers=2):
    root=Path(root);plan=json.loads((root/'engineering_plan.json').read_text());areas={a['id']:a for a in json.loads(Path(areas_path).read_text())['areas']}
    signature={'plan_sha256':sha(root/'engineering_plan.json'),'areas_sha256':sha(areas_path),'worker_limit':workers,
        'code_sha256':{m:sha(importlib.import_module('lst_pilot.'+m).__file__) for m in ['coarse_pipeline','coarse_lst','coarse_context','coarse_join','coarse_inventory']},
        'geometry_policy':GEOMETRY_POLICY}
    path=root/'pipeline_signature.json'
    if path.exists():
        if json.loads(path.read_text())!=signature:raise ValueError('Native/context processing signature changed.')
    else:save(path,signature)
    pending={r['region_id']+'_'+r['stem']:r for r in plan['records']};finished={};active={};last_progress=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            for key in list(pending):
                checkpoint=root/'processing_checkpoints'/f'{key}.json'
                if checkpoint.exists():
                    result=json.loads(checkpoint.read_text())
                    if result['status']=='processed' and sha(result['table_path'])!=result['table_sha256']:raise ValueError('Context checkpoint checksum changed.')
                    finished[key]=result;del pending[key];last_progress=time.monotonic();continue
                download=root/'downloads'/f'{key}.json'
                if not download.exists() or len(active)>=workers:continue
                data=json.loads(download.read_text())
                if data['status']!='downloaded':
                    finished[key]={'status':'download_failed','stem':pending[key]['stem'],'region_id':pending[key]['region_id'],'error':data['error']}
                    del pending[key];continue
                future=pool.submit(job,(data,areas[pending[key]['region_id']],str(root)));active[future]=key;del pending[key]
            for future in list(active):
                if not future.done():continue
                key=active.pop(future);finished[key]=future.result();last_progress=time.monotonic()
                print(json.dumps({'completed':len(finished),'total':len(plan['records']),**finished[key]}),flush=True)
            save(root/'processing_progress.json',{'complete':not pending and not active,'finished':len(finished),'active':len(active),'pending':len(pending),'records':list(finished.values())})
            if time.monotonic()-last_progress>900:raise TimeoutError('No native/download progress for 15 minutes; preserve checkpoints for resumption.')
            if pending or active:time.sleep(5)
    ordered=[finished[r['region_id']+'_'+r['stem']] for r in plan['records']]
    save(root/'context_manifest.json',{'signature':signature,'complete':True,'records':ordered,'coarse_label_training_eligible':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--areas',default='pilot/areas_resolved.json');p.add_argument('--workers',type=int,default=2,choices=[1,2])
    a=p.parse_args();run(a.root,a.areas,a.workers)
