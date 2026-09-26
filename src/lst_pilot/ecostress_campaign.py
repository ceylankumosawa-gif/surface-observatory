"""Metadata-frozen ECOSTRESS DAY fit / fresh2023 DAY-NIGHT research campaign.

No fitting or deployment. A separate immutable plan fixes all temporal sampling
before any new2023 thermal read. Existing source decoders are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import rasterio

from . import ecostress as eco, ecostress_geo as geo, highres_inventory as h
from . import highres_collect as engineering, option_b_acquire as base

VERSION = 'ecostress-multisensor-campaign-20260910-v1'
MAX_BYTES = 2 * 1024**3
MAX_CANDIDATES = 200
MAX_PHASE_SECONDS = 1200
REGISTRY_SHA = '2be509c11960eb23ddc0dd4d9c1e1e63c4b2c77bbfd417270847cab133125453'


def queue_key(record):
    return ':'.join((record['pilot_id'], record['temporal_split'], record['actual_phase']))


def make_queues(records, registry, initial):
    seen_dates = {(r['region_id'], r['utc_date']) for r in registry['dates']}
    attempted = {r['granule_id'] for r in initial['records']}
    attempted_orbits, initial_qualifying_dates = set(), set()
    for row in initial['records']:
        title = row.get('title', '')
        if title.startswith('ECOv002_L2T_LSTE_'):
            attempted_orbits.add((row['pilot_id'], geo.parse_identity(title)['orbit']))
            if row.get('source_screen_pass'):
                initial_qualifying_dates.add((row['pilot_id'], pd.Timestamp(row['time_start']).date().isoformat()))
    eligible = []
    for r in records:
        if r['product'] != 'ecostress_v2' or r['pilot_id'] not in h.PILOTS or r['granule_concept_id'] in attempted:
            continue
        if ((r['pilot_id'], geo.parse_identity(r['granule_title'])['orbit']) in attempted_orbits
                or (r['pilot_id'], r['utc_date']) in initial_qualifying_dates):
            continue
        stamp = pd.Timestamp(r['time_start'])
        if stamp.year not in (2021, 2022, 2023):
            raise ValueError('Reserved observation year in campaign input.')
        phase = ('day' if r['solar_elevation_min_centre_corners'] >= 10 else
                 'night' if r['solar_elevation_max_centre_corners'] <= -6 else 'twilight')
        if phase == 'twilight' or (stamp.year in (2021, 2022) and phase != 'day'):
            continue
        if stamp.year == 2023 and (r['pilot_id'], r['utc_date']) in seen_dates:
            continue
        eligible.append({**r, 'actual_phase': phase, 'fresh_2023': stamp.year == 2023})
    candidates = h.candidate_queue(eligible)
    lookup = {r['granule_concept_id']: r for r in eligible}
    queues = defaultdict(list)
    for acq in candidates:
        if acq['rank_within_stratum'] > 4:
            continue
        latest = {}
        for gid in acq['granule_concept_ids']:
            r = lookup[gid]
            ident = geo.parse_identity(r['granule_title'])
            key = (ident['scene'], ident['tile'])
            processing = (int(ident['build']), int(ident['counter']))
            if key not in latest or processing > latest[key][0]:
                latest[key] = (processing, r)
        row = dict(min((r for _, r in latest.values()), key=lambda r: (-r['metadata_overlap_m2'], r['granule_title'])))
        row['selection'] = {'rank': acq['rank'], 'rank_within_stratum': acq['rank_within_stratum'],
                            'stratum': acq['stratum'], 'greatest_metadata_overlap_m2': row['metadata_overlap_m2']}
        queues[queue_key(row)].append(row)
    # Equal calendar coverage before extra observations within a month. Within a
    # calendar turn alternate solar bins; no thermal quality or error sorting.
    for key, rows in queues.items():
        rows.sort(key=lambda r: (r['selection']['rank_within_stratum'], r['utc_date'][:7],
                                 r['local_solar_3hour_bin'], r['selection']['rank']))
    return dict(queues)


def prepare(root, inventory_path, registry_path, output):
    if output.exists():
        raise ValueError('New campaign output required.')
    if eco.digest(registry_path) != REGISTRY_SHA:
        raise ValueError('Previously inspected date registry differs from root freeze.')
    inventory = json.loads((inventory_path/'inventory.json').read_text())
    if not inventory['complete']:
        raise ValueError('Completed DAY metadata inventory required.')
    prior = root/'runs/option_b_metadata_merged_2021_2023_20260909'
    if json.loads((prior/'merge_registry.json').read_text())['status'] != 'complete_metadata_only':
        raise ValueError('Completed NIGHT metadata inventory required.')
    catalog_path = root/'web/catalog.json'
    pilots_geo = {r['properties']['id']: r for r in json.loads(catalog_path.read_text())['pilots']['features']}
    rows = json.loads((inventory_path/'granules.json').read_text())
    night = [r for r in json.loads((prior/'granules.json').read_text())
             if r['product']=='ecostress_v2' and r['day_night_flag']=='NIGHT' and pd.Timestamp(r['time_start']).year==2023]
    enriched = h.enrich(night, pilots_geo)
    by_id = {(r['pilot_id'], r['granule_concept_id']): r for r in rows+enriched}
    initial_path = root/'runs/multisensor_20260910_v1/highres_engineering_v1/manifest.json'
    initial = json.loads(initial_path.read_text())
    registry = json.loads(registry_path.read_text())
    queues = make_queues(list(by_id.values()), registry, initial)
    obstruction_path = root/'cache/ecostress_geo/public_obstruction/obstruction_dfceb418ba111fae4db89816442ffd540b7c57b0e1e9cc51b6163a1efee5b26a.txt'
    obstruction = geo.parse_obstruction_list(obstruction_path.read_bytes(), complete=True)
    if obstruction['status'] != 'Parsed' or obstruction['errors']:
        raise ValueError('Complete historical obstruction index required.')
    rejected = []
    for key in list(queues):
        candidates = []
        for row in queues[key]:
            status = geo.obstruction_status(obstruction, row['granule_title'])
            if status.get('obstructed') is True or status['status']=='Unknown':
                rejected.append({'granule_id': row['granule_concept_id'], 'queue': key,
                                 'reason': 'confirmed_or_ambiguous_obstruction'})
            else:
                candidates.append({**row, 'obstruction': status})
        queues[key] = candidates
    keys = [f'{pilot}:{split}:{phase}' for split, phase in
            (('fit','day'),('development','day'),('development','night'),('calibration','day'),('calibration','night'))
            for pilot in h.PILOTS]
    for key in keys:
        queues.setdefault(key, [])
    paths = [Path(__file__), Path(h.__file__), Path(eco.__file__), Path(geo.__file__), Path(base.__file__),
             Path(engineering.__file__), inventory_path/'inventory.json', inventory_path/'granules.json',
             inventory_path/'spatial_blocks.json', prior/'granules.json', prior/'merge_registry.json',
             registry_path, initial_path, catalog_path, root/'pilot/areas_resolved.json', obstruction_path,
             root/'reports/multisensor/PROTOCOL_2026-09-10.md',
             root/'reports/multisensor/HIGHRES_COLLECTION_ADDENDUM.md']
    output.mkdir(parents=True)
    plan = {'version': VERSION, 'queues': queues, 'queue_order': keys,
        'targets': {key: 12 if ':fit:' in key else 3 for key in keys},
        'source_hashes': {str(p): eco.digest(p) for p in paths}, 'registry_sha256': REGISTRY_SHA,
        'limits': {'candidate_attempts': MAX_CANDIDATES, 'additional_network_bytes': MAX_BYTES,
                   'soft_seconds_per_pilot_phase': MAX_PHASE_SECONDS, 'acquisitions_per_calendar_hour_stratum': 4},
        'source_counts': {'queue_lengths': {k:len(v) for k,v in queues.items()},
                          'queue_dates': {k:len({r['utc_date'] for r in v}) for k,v in queues.items()},
                          'metadata_obstruction_rejections': len(rejected)},
        'metadata_rejections': rejected, 'engineering_only': True, 'training_eligible': False,
        'selection_rule': 'Round-robin10 pilot/split/phase queues. Within each: rank-within-month/3hour stratum, then month, solar bin, stable metadata hash. At most4 acquisition ranks per stratum. Latest build/iteration per scene/tile, then greatest CMR overlap per orbit. Initial attempted ECOSTRESS orbits and initial qualifying pilot-dates excluded, including alternate tiles. After source QA failure continue frozen queue; stop a queue at12 qualifying new fit dates or3 fresh evaluation dates. Same qualifying pilot/date is not counted twice in a phase. No residual or thermal-magnitude ranking.',
        'date_rules': '2021/22 DAY fit; fresh2023H1 development andH2 calibration DAY/NIGHT; all previously attempted/inspected2023pilot-dates excluded across sensors. Whole pilot-date split is fixed. No2024/25 labels.',
        'solar_rules': {'day_minimum_centre_and_corners_deg':10,'night_maximum_centre_and_corners_deg':-6},
        'qa_rules': eco.QA_RULES, 'qualifying_rule': 'At least200 native-fit-safe100m cells across2 fixed nonreserved10km blocks with50 cells each; at least200QA cells for evaluation dates. Final source/feature/fit admission remains separate.',
        'pilot_areas': {r['id']:r for r in json.loads((root/'pilot/areas_resolved.json').read_text())['areas'] if r['id'] in h.PILOTS},
        'spatial_blocks': json.loads((inventory_path/'spatial_blocks.json').read_text())}
    eco.save_json(output/'plan.json', plan)
    return plan


class Ledger(base.Ledger):
    def public_json(self, params, path):
        # The frozen older ledger permits one final streamed chunk. Reserve it
        # separately without modifying its historical implementation.
        extra = 0 if Path(path).is_file() else 65536
        self.reserve(extra)
        try:
            return super().public_json(params, path)
        finally:
            self.settle(extra, 0, 0)


class Downloader(eco.NasaDownloader):
    def __init__(self, ledger, token=None):
        super().__init__(max_bytes=MAX_BYTES, max_requests=4000, token=token)
        self.ledger = ledger

    def download(self, url, destination, kind='cog'):
        path = Path(destination)
        cached = path.is_file() and path.with_suffix('.download.json').is_file()
        reserve = 0 if cached else (32 if kind=='cog' else 4)*1024**2+65536
        self.ledger.reserve(reserve)
        before_bytes, before_requests = self.bytes, self.requests
        self.max_bytes = self.bytes + MAX_BYTES - self.ledger.state['network_bytes_charged'] + reserve
        try:
            return super().download(url, destination, kind)
        finally:
            self.ledger.settle(reserve, self.bytes-before_bytes, self.requests-before_requests, protected=True)


def qualifying_dates(records):
    dates = defaultdict(set)
    for r in records:
        if r.get('qualifying_independent_date'):
            dates[queue_key(r)].add(r['utc_date'])
    return dates


def next_candidate(plan, state, records):
    counts = qualifying_dates(records)
    for offset in range(len(plan['queue_order'])):
        index = (state['rotation']+offset) % len(plan['queue_order'])
        key = plan['queue_order'][index]
        pilot, _, phase = key.split(':')
        if (len(counts[key]) >= plan['targets'][key]
                or state['seconds_by_pilot_phase'].get(pilot+':'+phase,0) >= MAX_PHASE_SECONDS):
            continue
        cursor = state['cursors'].get(key,0)
        while cursor < len(plan['queues'][key]):
            row = plan['queues'][key][cursor]
            cursor += 1
            state['cursors'][key] = cursor
            if row['utc_date'] in counts[key]:
                continue
            state['rotation'] = (index+1) % len(plan['queue_order'])
            return row
    return None


def acquire(record, plan, root, output, client, ledger):
    result = base.acquire_one(record, plan, root, output, client, ledger)
    result.update(actual_phase=record['actual_phase'], product='ecostress_v2', fresh_2023=record['fresh_2023'])
    if 'thermal_summary' not in result:
        return result
    summary = result['thermal_summary']
    result.update(qa_pass_cells=summary['qa_pass_cells'], downloads=summary['downloads'], qa_rules=summary['qa_rules'],
                  cmr_meta=summary['cmr_meta'], native_geometry=summary['native_geometry'])
    native = root/'cache/ecostress_v2'/record['granule_title']/'LST.tif'
    pilot = plan['pilot_areas'][record['pilot_id']]
    support = engineering.write_fit_support(result, pilot, plan['spatial_blocks'], native)
    result['native_fit_support'] = support
    with rasterio.open(support['path']) as src:
        safe = src.read(1)==1
    blocks = []
    for block in result['spatial_coverage']['blocks']:
        selected = next(b for b in plan['spatial_blocks'] if b['id']==block['block_id'])
        l,b,r,t = selected['bounds_m']
        r0 = round((pilot['extent_m'][3]-t)/100); r1 = round((pilot['extent_m'][3]-b)/100)
        c0 = round((l-pilot['extent_m'][0])/100); c1 = round((r-pilot['extent_m'][0])/100)
        blocks.append(int(safe[r0:r1,c0:c1].sum()))
    result['qualifying_independent_date'] = (support['safe_cells']>=200 and sum(n>=50 for n in blocks)>=2
                                              if record['temporal_split']=='fit' else result['qa_pass_cells']>=200)
    result['label_source_sha256'] = summary['downloads']['LST']['sha256']
    result['label_valid_fraction_band'] = 2
    result['label_source_geolocation_proof'] = {'matched_geo':result['geolocation'], 'obstruction':result['obstruction']}
    result['label_source_cloud_proof'] = {'native_qa_rules':eco.QA_RULES, 'independent_cloud_absence_proven':False}
    return result


def run(root, output, batch_size):
    plan = json.loads((output/'plan.json').read_text()); plan_sha = eco.digest(output/'plan.json')
    for path, sha in plan['source_hashes'].items():
        if eco.digest(path)!=sha:
            raise ValueError('Frozen campaign input/code changed.')
    state_path, manifest_path = output/'state.json', output/'manifest.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'plan_sha256':plan_sha,
        'network_bytes_charged':0,'protected_http_requests':0,'public_http_requests':0,'candidate_attempts':0,
        'rotation':0,'cursors':{},'inflight':None,'seconds_by_pilot_phase':{}}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'version':VERSION,
        'plan_sha256':plan_sha,'records':[],'source_counts':plan['source_counts'],'engineering_only':True,'training_eligible':False}
    if state['plan_sha256']!=plan_sha or manifest['plan_sha256']!=plan_sha:
        raise ValueError('Campaign plan changed during resume.')
    ledger = Ledger(output,state)
    if state['inflight'] and any(r['granule_id']==state['inflight']['granule_concept_id'] for r in manifest['records']):
        state['inflight']=None; ledger.save()
    client = Downloader(ledger)
    for _ in range(batch_size):
        if (state['candidate_attempts']>=MAX_CANDIDATES and state['inflight'] is None) or state['network_bytes_charged']>=MAX_BYTES:
            break
        row = state['inflight'] or next_candidate(plan,state,manifest['records'])
        if row is None:
            break
        if state['inflight'] is None:
            state['inflight']=row; state['candidate_attempts']+=1; ledger.save()
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
        eco.save_json(manifest_path,manifest)  # Commit result while inflight remains recoverable.
        state['inflight']=None; ledger.save()
        print(json.dumps({'attempt':state['candidate_attempts'],'queue':queue_key(row),'date':row['utc_date'],
            'status':result['status'],'source_screen_pass':result.get('source_screen_pass'),
            'qualifying_dates':{k:len(v) for k,v in manifest['qualifying_dates'].items()},
            'charged_mib':round(state['network_bytes_charged']/1024**2,2)}),flush=True)
    manifest['batch_complete']=True
    manifest['candidate_cap_reached']=state['candidate_attempts']>=MAX_CANDIDATES
    eco.save_json(manifest_path,manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path); p.add_argument('--registry',type=Path)
    p.add_argument('--batch-size',type=int,default=20)
    args=p.parse_args(); resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if args.action=='prepare': prepare(args.root,args.inventory,args.registry,args.output)
    else:
        if not 1<=args.batch_size<=200: p.error('Batch size1..200 required.')
        with (args.output/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.root,args.output,args.batch_size)


if __name__=='__main__': main()
