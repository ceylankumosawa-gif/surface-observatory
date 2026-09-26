"""Separate bounded ECOSTRESS nighttime fitting extension; no evaluation labels.

The daytime/evaluation campaign remains immutable. Native QA and row provenance
reuse the reviewed decoder; this file owns its own frozen plan and hard limits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import resource
import time

from . import ecostress_campaign as reviewed
from . import ecostress as eco, ecostress_geo as geo, highres_inventory as h
from . import highres_collect as engineering, option_b_acquire as base

VERSION = 'ecostress-night-fit-extension-20260910-v1'
MAX_BYTES = 1024**3
MAX_CANDIDATES = 100
MAX_PHASE_SECONDS = 1200
queue_key = reviewed.queue_key
qualifying_dates = reviewed.qualifying_dates
next_candidate = reviewed.next_candidate
acquire = reviewed.acquire


def balanced_queues(records):
    """One candidate per year/season turn before a second in that stratum."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        if (row['temporal_split']!='fit' or row['utc_date'][:4] not in ('2021','2022')
                or row['actual_phase']!='night' or row['solar_elevation_max_centre_corners']>-6):
            raise ValueError('Only actual 2021/22 fitting nights may enter the extension.')
        if row['selection']['rank_within_stratum']>4:
            continue
        month = int(row['utc_date'][5:7])
        quarter = (month-1)//3+1
        # Quarter then year alternates the two years within each season.
        groups[row['pilot_id']][(quarter,row['utc_date'][:4])].append(dict(row, fresh_2023=False))
    queues = {}
    for pilot in h.PILOTS:
        seasons = groups[pilot]
        for rows in seasons.values():
            rows.sort(key=lambda r:(r['selection']['rank_within_stratum'],r['utc_date'][:7],
                                    r['local_solar_3hour_bin'],r['selection']['rank']))
        result=[]
        for depth in range(max((len(r) for r in seasons.values()), default=0)):
            for season in sorted(seasons):
                if depth<len(seasons[season]):
                    row=seasons[season][depth]
                    row['selection']={**row['selection'],'year_season_turn':depth+1,'season':list(season)}
                    result.append(row)
        queues[f'{pilot}:fit:night']=result
    return queues


def prepare(root, inventory_path, output):
    if output.exists():
        raise ValueError('A new extension output directory is required.')
    inventory=json.loads(inventory_path.read_text())
    if (not inventory['catalog_complete'] or inventory['thermal_arrays_opened']!=0
            or inventory['public_requests']!=0 or inventory['protected_requests']!=0):
        raise ValueError('Frozen offline inventory required.')
    for path,sha in inventory['source_hashes'].items():
        if eco.digest(path)!=sha:
            raise ValueError('Night inventory source changed.')
    queues=balanced_queues(inventory['candidates'])
    prior=root/'runs/option_b_metadata_merged_2021_2023_20260909'
    paths=[Path(__file__),Path(reviewed.__file__),Path(eco.__file__),Path(geo.__file__),Path(h.__file__),
           Path(engineering.__file__),Path(base.__file__),inventory_path,prior/'spatial_blocks.json',
           root/'pilot/areas_resolved.json',root/'reports/multisensor/NIGHT_FIT_EXTENSION_ADDENDUM.md',
           root/'reports/multisensor/PROTOCOL_2026-09-10.md']
    plan={'version':VERSION,'queues':queues,'queue_order':[f'{p}:fit:night' for p in h.PILOTS],
          'targets':{k:6 for k in queues},
          'source_hashes':{**inventory['source_hashes'],**{str(p):eco.digest(p) for p in paths}},
          'limits':{'candidate_attempts':MAX_CANDIDATES,'additional_network_bytes':MAX_BYTES,
                    'soft_seconds_per_pilot_phase':MAX_PHASE_SECONDS,'acquisitions_per_calendar_hour_stratum':4},
          'source_counts':{'queue_lengths':{k:len(v) for k,v in queues.items()},
                           'queue_dates':{k:len({r['utc_date'] for r in v}) for k,v in queues.items()}},
          'selection_rule':'Alternate pilots. Within each, round robin eight calendar-quarter/year strata in quarter then year order, taking one candidate from each nonempty stratum per turn. Within a stratum, fixed rank-within-month/solar-hour then month, solar hour, SHA rank. At most four acquisition ranks per month/3-hour stratum. No LST/error selection; continue fixed queue after source-QA failure.',
          'date_rules':'2021/22 NIGHT fitting only. Exclude every root registry pilot-date and all prior200 attempted orbits/dates, including alternate native tiles. No2023/24/25 thermal labels.',
          'exclusions':inventory['exclusions'],'metadata_qa_limitation':inventory['limits'],
          'qa_rules':eco.QA_RULES,'engineering_only':True,'training_eligible':False,
          'qualifying_rule':'At least200 native-fit-safe100m cells across two fixed nonreserved10km blocks with50 cells each. Final features/admission remain separate.',
          'pilot_areas':{r['id']:r for r in json.loads((root/'pilot/areas_resolved.json').read_text())['areas'] if r['id'] in h.PILOTS},
          'spatial_blocks':json.loads((prior/'spatial_blocks.json').read_text())}
    output.mkdir(parents=True)
    eco.save_json(output/'plan.json',plan)
    return plan


class Ledger(reviewed.Ledger):
    def reserve(self, amount):
        if self.state['network_bytes_charged']+amount>MAX_BYTES:
            raise base.CampaignStop('Night extension cumulative network byte budget exhausted.')
        self.state['network_bytes_charged']+=amount
        self.save()


class Downloader(eco.NasaDownloader):
    def __init__(self,ledger,token=None):
        super().__init__(max_bytes=MAX_BYTES,max_requests=4000,token=token)
        self.ledger=ledger

    def download(self,url,destination,kind='cog'):
        path=Path(destination)
        cached=path.is_file() and path.with_suffix('.download.json').is_file()
        reserve=0 if cached else (32 if kind=='cog' else 4)*1024**2+65536
        self.ledger.reserve(reserve)
        before_bytes,before_requests=self.bytes,self.requests
        self.max_bytes=self.bytes+MAX_BYTES-self.ledger.state['network_bytes_charged']+reserve
        try:
            return super().download(url,destination,kind)
        finally:
            self.ledger.settle(reserve,self.bytes-before_bytes,self.requests-before_requests,protected=True)


def run(root,output,batch_size):
    plan=json.loads((output/'plan.json').read_text());plan_sha=eco.digest(output/'plan.json')
    for path,sha in plan['source_hashes'].items():
        if eco.digest(path)!=sha:
            raise ValueError('Frozen extension source/code changed.')
    if plan['limits']['candidate_attempts']!=MAX_CANDIDATES or plan['limits']['additional_network_bytes']!=MAX_BYTES:
        raise ValueError('Plan and runner hard limits differ.')
    state_path,manifest_path=output/'state.json',output/'manifest.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'plan_sha256':plan_sha,
        'network_bytes_charged':0,'protected_http_requests':0,'public_http_requests':0,'candidate_attempts':0,
        'rotation':0,'cursors':{},'inflight':None,'seconds_by_pilot_phase':{}}
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'version':VERSION,
        'plan_sha256':plan_sha,'records':[],'source_counts':plan['source_counts'],'engineering_only':True,'training_eligible':False}
    if state['plan_sha256']!=plan_sha or manifest['plan_sha256']!=plan_sha:
        raise ValueError('Extension plan changed during resume.')
    ledger=Ledger(output,state)
    if state['inflight'] and any(r['granule_id']==state['inflight']['granule_concept_id'] for r in manifest['records']):
        state['inflight']=None;ledger.save()
    client=Downloader(ledger)
    for _ in range(batch_size):
        if (state['candidate_attempts']>=MAX_CANDIDATES and state['inflight'] is None) or state['network_bytes_charged']>=MAX_BYTES:
            break
        row=state['inflight'] or next_candidate(plan,state,manifest['records'])
        if row is None:
            break
        if state['inflight'] is None:
            state['inflight']=row;state['candidate_attempts']+=1;ledger.save()
        started=time.monotonic()
        try:
            result=acquire(row,plan,root,output,client,ledger)
        except Exception as error:
            result={k:row[k] for k in ('pilot_id','time_start','utc_date','temporal_split','actual_phase')}
            result.update(granule_id=row['granule_concept_id'],title=row['granule_title'],product='ecostress_v2',
                status='bounded_acquisition_error',error_type=type(error).__name__,source_screen_pass=False,
                qualifying_independent_date=False,training_eligible=False)
        key=row['pilot_id']+':'+row['actual_phase']
        state['seconds_by_pilot_phase'][key]=state['seconds_by_pilot_phase'].get(key,0)+time.monotonic()-started
        manifest['records']=[r for r in manifest['records'] if r['granule_id']!=result['granule_id']]+[result]
        manifest['qualifying_dates']={k:sorted(v) for k,v in qualifying_dates(manifest['records']).items()}
        manifest['counters']={k:v for k,v in state.items() if k not in ('inflight','cursors')}
        eco.save_json(manifest_path,manifest)
        state['inflight']=None;ledger.save()
        print(json.dumps({'attempt':state['candidate_attempts'],'queue':queue_key(row),'date':row['utc_date'],
                          'status':result['status'],'source_screen_pass':result.get('source_screen_pass'),
                          'qualifying_dates':{k:len(v) for k,v in manifest['qualifying_dates'].items()},
                          'charged_mib':round(state['network_bytes_charged']/1024**2,2)}),flush=True)
    manifest['batch_complete']=True
    manifest['candidate_cap_reached']=state['candidate_attempts']>=MAX_CANDIDATES
    eco.save_json(manifest_path,manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inventory',type=Path);parser.add_argument('--batch-size',type=int,default=20)
    args=parser.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if args.action=='prepare':
        plan=prepare(args.root,args.inventory,args.output)
        print(json.dumps({'plan_sha256':eco.digest(args.output/'plan.json'),'source_counts':plan['source_counts']}))
    else:
        if not 1<=args.batch_size<=100:parser.error('Batch size1..100 required.')
        with (args.output/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.root,args.output,args.batch_size)


if __name__=='__main__':main()
