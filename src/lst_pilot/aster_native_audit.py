"""Read-only ASTER native QA / published checksum audit; no network or fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import rasterio

from . import aster, ecostress as eco


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('New immutable audit output required.')
    plan=json.loads((args.source/'plan.json').read_text())
    manifest=json.loads((args.source/'manifest.json').read_text())
    results={r['granule_id']:r for r in manifest['records']}
    records=[]
    for r in plan['records']:
        if r['product']!='aster_v4':continue
        result=results[r['granule_concept_id']]
        folder=args.root/'cache/aster_v4'/r['granule_title']
        assets={a['Name']:a for a in r['cmr_envelope']['umm']['DataGranule']['ArchiveAndDistributionInformation']}
        verified=[]
        for layer in aster.LAYERS:
            path=folder/(layer+'.tif');name=r['granule_title']+'_'+layer+'.tif'
            meta=assets.get(name,{})
            checksum=meta.get('Checksum',{})
            algorithm=re.sub('[^a-z0-9]','',checksum.get('Algorithm','').lower())
            computed=None
            if algorithm in ('sha512','sha256','md5'):
                h=hashlib.new(algorithm)
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
                computed=h.hexdigest()
            verified.append({'layer':layer,'published_algorithm':checksum.get('Algorithm'),
                'published_checksum_match':computed is not None and computed.lower()==checksum.get('Value','').lower(),
                'published_bytes_match':meta.get('SizeInBytes')==path.stat().st_size,
                'retained_sha256':eco.digest(path),'bytes':path.stat().st_size})
        with rasterio.open(folder/'SKT.tif') as s:
            valid=s.read(1)>0;native={'transform':list(s.transform),'shape':list(s.shape),'crs':str(s.crs),'dtype':s.dtypes[0],'scales':list(s.scales)}
        with rasterio.open(folder/'SKT_QA_DataPlane.tif') as s:q=s.read(1)
        with rasterio.open(folder/'SKT_QA_DataPlane2.tif') as s:q2=s.read(1);qa2_dtype=s.dtypes[0]
        histogram=lambda a:{str(x):int(n) for x,n in zip(*np.unique(a,return_counts=True))}
        flags=r['cmr_envelope']['umm'].get('MeasuredParameters',[])
        l1t=[]
        for item in r['l1t_metadata_candidates']:
            umm=item['umm']
            measured=[a for a in umm.get('AdditionalAttributes',[]) if any(t in a['Name'].lower() for t in ('rmse','geolocationaccuracy','geometricaccuracy'))]
            l1t.append({'granule_title':umm['GranuleUR'],'cmr_meta':item['meta'],
                         'measured_registration_accuracy_fields':measured,'qa':umm.get('MeasuredParameters',[])})
        records.append({'pilot_id':r['pilot_id'],'title':r['granule_title'],'utc_time':r['time_start'],
            'phase':r['actual_phase'],'native_geometry':native,'nonfill_skt_pixels':int(valid.sum()),
            'first_qa_plane_nonfill_histogram':histogram(q[valid]),
            'cloud_status_nonfill_histogram':histogram((q[valid]>>2)&3),
            'lower_four_bits_nonfill_histogram':histogram(q[valid]&15),
            'qa2_dtype':qa2_dtype,'qa2_nonfill_histogram':histogram(q2[valid]),
            'scene_qa':flags,'asset_integrity':verified,'matched_l1t_metadata':l1t,
            'derived_qa100m_cells':result.get('qa_pass_cells',0),'source_screen_pass':False,
            'blockers':['No active cloud/adjacency bits observed in this sample; cloud absence not proven',
                        'No measured AST_08 registration accuracy; L1T execution success does not certify L2 alignment'],
            'next_check':'Contemporaneous Terra native1km cloud/LST comparison, separately from causal prediction context; independent optical registration required before100m label admission.'})
    audit={'version':'aster-native-audit-20260910-v1','source_plan_sha256':eco.digest(args.source/'plan.json'),
        'source_manifest_sha256':eco.digest(args.source/'manifest.json'),'audit_source_sha256':eco.digest(__file__),
        'official_sources':aster.SOURCES,'network_requests':0,'new_assets':0,'records':records,
        'all_published_checksums_match':all(a['published_checksum_match'] and a['published_bytes_match'] for r in records for a in r['asset_integrity']),
        'all_lower_four_bits_zero_in_nonfill_skt':all(r['lower_four_bits_nonfill_histogram']=={'0':r['nonfill_skt_pixels']} for r in records)}
    eco.save_json(args.output,audit)
    print(json.dumps({k:audit[k] for k in ('all_published_checksums_match','all_lower_four_bits_zero_in_nonfill_skt')}),flush=True)


if __name__=='__main__':main()
