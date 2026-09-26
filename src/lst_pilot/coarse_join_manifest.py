"""Bind successful native context tables without hiding failed acquisitions."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from .coarse_inventory import sha,save,identity


def prepare(sources,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Join-input manifest already frozen.')
    records={};excluded=[];provenance=[]
    for source in sources:
        source=Path(source);data=json.loads(source.read_text())
        if data.get('complete') is False:raise ValueError('Native context acquisition is not complete.')
        provenance.append({'path':str(source.resolve()),'sha256':sha(source),'signature':data.get('signature')})
        for record in data['records']:
            status=record.get('status','processed')
            if status in ('download_failed','rejected'):
                excluded.append({'source_manifest_sha256':sha(source),**record});continue
            if status!='processed':raise ValueError('Unknown native source outcome; explicit review required.')
            if not record.get('table_path') or not record.get('table_sha256'):raise ValueError('Processed source lacks table proof.')
            if sha(record['table_path'])!=record['table_sha256']:raise ValueError('Processed source checksum changed.')
            key=(record['region_id'],identity(record['stem'])['acquisition_key'])
            if key in records:
                # Independently repeated ASTER diagnostic/source checkpoints may
                # have identical source data and output values in separate roots.
                if records[key]['table_sha256']!=record['table_sha256']:raise ValueError('Duplicate native source has inconsistent context output.')
                continue
            records[key]=record
    result={'version':'native-context-join-input-v1','source_manifests':provenance,'source_code_sha256':sha(__file__),
        'records':list(records.values()),'excluded_acquisitions':excluded,'excluded_status_counts':dict(Counter(x['status'] for x in excluded)),
        'coarse_label_training_eligible':False,'complete':True}
    save(output,result);print(json.dumps({'path':str(output),'sha256':sha(output),'processed_sources':len(records),'excluded_sources':len(excluded)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.sources,a.output)
