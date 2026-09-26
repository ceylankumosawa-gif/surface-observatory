"""Bounded, resumable Option B nighttime acquisition; no fitting or serving changes.

Freeze the metadata order first. Positive matched GEO precedes thermal access.
Source screening does not certify complete training or 100 m alignment eligibility.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import signal
import time

import numpy as np
import pandas as pd
import pvlib
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
import requests
from shapely.geometry import box, shape
from shapely.ops import transform

from . import ecostress as eco
from . import ecostress_geo as geo

VERSION = 'option-b-acquire-night-20260909-v2'
PILOTS = ('greater_london', 'sioux_falls')
SPLITS = ('fit', 'development', 'calibration')
TARGETS = {'fit': 12, 'development': 4, 'calibration': 4}
MAX_CANDIDATES = 200
MAX_BYTES = 2 * 1024**3
MAX_SECONDS = 45 * 60
STREAM_CHUNK_BYTES = 65536
OBSTRUCTION_SHA = 'dfceb418ba111fae4db89816442ffd540b7c57b0e1e9cc51b6163a1efee5b26a'


class CampaignStop(ValueError):
    pass


def write(path, obj):
    eco.save_json(path, obj)


def frozen_split(stamp):
    stamp = pd.Timestamp(stamp)
    if stamp.tzinfo is None:
        raise ValueError('Timezone is required.')
    stamp = stamp.tz_convert('UTC')
    if stamp.year in (2021, 2022):
        return 'fit'
    if stamp.year == 2023:
        return 'development' if stamp.month <= 6 else 'calibration'
    raise ValueError('Reserved or unsupported observation year.')


def ordered_acquisitions(acquisitions, pilot, split):
    """Round across frozen calendar strata, retaining each stratum's rank order."""
    rows = [a for a in acquisitions if a['pilot_id'] == pilot and a['temporal_split'] == split
            and 'ecostress_v2' in a['products_present']]
    def key(a):
        year, month = map(int, a['stratum'][2].split('-'))
        if year not in (2021, 2022, 2023):
            raise ValueError('Reserved year in acquisition queue.')
        year_priority = (0 if year == 2022 else 1) if pilot == 'greater_london' and split == 'fit' else year
        return (a['rank_within_stratum'], year_priority, month, a['stratum'][3], a['rank'], a['acquisition_group'])
    return sorted(rows, key=key)


def build_plan(root, output):
    source = root / 'runs/option_b_metadata_merged_2021_2023_20260909'
    obstruction = root / f'cache/ecostress_geo/public_obstruction/obstruction_{OBSTRUCTION_SHA}.txt'
    if eco.digest(obstruction) != OBSTRUCTION_SHA:
        raise ValueError('Complete cached obstruction source hash changed.')
    index = geo.parse_obstruction_list(obstruction.read_bytes(), complete=True)
    if index['status'] != 'Parsed' or index['errors']:
        raise ValueError('Complete obstruction source does not parse cleanly.')
    candidates = json.loads((source / 'calendar_candidates.json').read_text())
    if not candidates['inventory_complete']:
        raise ValueError('Full source inventory is not complete.')
    areas = {p['id']: p for p in json.loads((root / 'pilot/areas_resolved.json').read_text())['areas']}
    rows = [r for r in json.loads((source / 'granules.json').read_text())
            if r['pilot_id'] in PILOTS and r['product'] == 'ecostress_v2'
            and r['footprint_qa']['status'] == 'metadata_consistent_pixels_unverified']
    grouped = defaultdict(list)
    for record in rows:
        if frozen_split(record['time_start']) != record['temporal_split']:
            raise ValueError('Source split mismatch.')
        grouped[(record['pilot_id'], record['identity']['acquisition_group'])].append(record)
    obs_by_scene = defaultdict(list)
    for record in index['records']:
        obs_by_scene[(record['orbit'], record['scene'])].append(record)
    queues, rejected = {}, []
    for pilot in PILOTS:
        area = areas[pilot]
        stamps = pd.DatetimeIndex(sorted({r['time_start'] for r in rows if r['pilot_id'] == pilot}))
        lon, lat = area.get('resolved_center_lonlat', area['center_lonlat'])
        left, bottom, right, top = area['extent_m']
        to_geo = Transformer.from_crs(area['epsg'], 4326, always_xy=True)
        points = [(lon, lat)] + [to_geo.transform(x, y) for x, y in ((left, bottom), (left, top), (right, bottom), (right, top))]
        elevations = np.array([pvlib.solarposition.get_solarposition(stamps, y, x)['elevation'].to_numpy() for x, y in points])
        solar = {stamp.isoformat(): (float(elevations[0, n]), float(elevations[:, n].max())) for n, stamp in enumerate(stamps)}
        project = Transformer.from_crs(4326, area['epsg'], always_xy=True).transform
        extent = box(*area['extent_m'])
        for split in SPLITS:
            queue = []
            for acquisition in ordered_acquisitions(candidates['acquisitions'], pilot, split):
                latest = {}
                for record in grouped[(pilot, acquisition['acquisition_group'])]:
                    identity = geo.parse_identity(record['granule_title'])
                    key = (identity['scene'], identity['tile'])
                    version = (int(identity['build']), int(identity['counter']), record['granule_title'], record['granule_concept_id'])
                    if key not in latest or version > latest[key][0]:
                        latest[key] = (version, record)
                options, reasons = [], Counter()
                for version, record in latest.values():
                    centre, maximum = solar[pd.Timestamp(record['time_start']).isoformat()]
                    if maximum > -6:
                        reasons['not_night_at_centre_and_four_pilot_corners'] += 1
                        continue
                    identity = record['identity']
                    subset = {**index, 'records': obs_by_scene[(int(identity['orbit']), int(identity['scene']))]}
                    status = geo.obstruction_status(subset, record['granule_title'])
                    if status.get('obstructed') is True or status['status'] == 'Unknown':
                        reasons['confirmed_obstruction_or_ambiguous_obstruction_identity'] += 1
                        continue
                    overlap = float(transform(project, shape(record['footprint_qa']['cmr_geometry'])).intersection(extent).area)
                    if overlap > 0:
                        options.append((-overlap, record['granule_title'], record, centre, maximum, status, version[:2]))
                if not options:
                    rejected.append({'pilot_id': pilot, 'acquisition_group': acquisition['acquisition_group'],
                                     'temporal_split': split, 'reasons': dict(reasons)})
                    continue
                _, _, selected, centre, maximum, status, processing = min(options, key=lambda row: row[:2])
                selected = dict(selected)
                selected['selection'] = {'existing_rank': acquisition['rank'], 'rank_within_stratum': acquisition['rank_within_stratum'],
                                         'source_stratum': acquisition['stratum'], 'queue_position': len(queue) + 1,
                                         'solar_elevation_centre': centre, 'solar_elevation_max_centre_corners': maximum,
                                         'metadata_overlap_m2': -min(options, key=lambda row: row[:2])[0],
                                         'processing_build': processing[0], 'processing_iteration': processing[1]}
                selected['obstruction'] = status
                queue.append(selected)
            queues[f'{pilot}:{split}'] = queue
    paths = [source / p for p in ('calendar_candidates.json', 'granules.json', 'merge_registry.json', 'spatial_blocks.json')]
    paths += [root / 'pilot/areas_resolved.json', obstruction, Path(__file__), Path(eco.__file__), Path(geo.__file__),
              root / 'reports/night_replacement/OPTION_B_PROTOCOL.md', root / 'reports/night_replacement/OPTION_B_FITTING_ADDENDUM.md']
    plan = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(), 'phase': 'night',
            'training_eligible': False, 'source_hashes': {str(path): eco.digest(path) for path in paths},
            'selection_rule': 'Round-robin the six pilot/split queues, stopping a queue after its distinct qualifying-date target. Within each queue round across calendar strata by rank_within_stratum, then year (London fitting prioritizes 2022), month, original solar bin, frozen original rank and acquisition ID. Each acquisition contributes one latest-build/iteration-per-scene/tile candidate with maximum CMR overlap among true-night and obstruction-screened options. No rank depends on thermal values or prediction errors. Extend within these frozen queues after quality failure; never reorder to favor labels.',
            'targets_exploratory_distinct_dates': TARGETS, 'limits': {'candidate_granules': MAX_CANDIDATES, 'network_bytes': MAX_BYTES, 'wall_seconds': MAX_SECONDS},
            'qualifying_date_rule': 'Positive Good/Best matched GEO; no confirmed/ambiguous obstruction; native thermal QA; at least200 buffered-nonreserved supported100m cells across at least2 nonreserved10km blocks with at least50 cells each. These acquisition-planning thresholds are not a completed training-eligibility test.',
            'obstruction_caveat': 'NotListed in the complete confirmed-obstruction list is only an exclusion-list screen; it does not prove unobstructed pixels. Preserve unresolved obstruction and independent registration as pending training gates.',
            'source_counts': {'granules': len(rows), 'prescreened_acquisitions': len(rejected), 'queue_lengths': {key: len(value) for key, value in queues.items()}},
            'metadata_rejections': rejected, 'queues': queues, 'pilot_areas': {pilot: areas[pilot] for pilot in PILOTS},
            'spatial_blocks': json.loads((source / 'spatial_blocks.json').read_text())}
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'plan.json'
    if path.exists():
        raise ValueError('Existing frozen plan must not be overwritten.')
    write(path, plan)
    return plan


def spatial_coverage(raster_path, pilot, blocks):
    """Whole-cell block/buffer screening; raster row/column IDs remain stable."""
    with rasterio.open(raster_path) as src:
        values = src.read(1)
        good = np.isfinite(values)
        left, bottom, right, top = pilot['extent_m']
        expected = from_origin(left, top, 100, 100)
        if src.transform != expected or src.shape != tuple(pilot['grid_shape']) or src.crs.to_epsg() != pilot['epsg']:
            raise ValueError('Label raster differs from the fixed pilot grid.')
        rows, cols = np.indices(src.shape)
        x, y = left + (cols + .5) * 100, top - (rows + .5) * 100
        reserved, buffered = np.zeros(src.shape, bool), np.zeros(src.shape, bool)
        selected_blocks = [b for b in blocks if b['id'].startswith(pilot['id'] + '_')]
        for block in selected_blocks:
            if not block['spatial_holdout']:
                continue
            l, b, r, t = block['bounds_m']
            inside = (x >= l) & (x < r) & (y >= b) & (y < t)
            reserved |= inside
            # Distance from the entire 100m cell, not just its centre, to reserve.
            dx = np.maximum.reduce([l - (x + 50), (x - 50) - r, np.zeros(src.shape)])
            dy = np.maximum.reduce([b - (y + 50), (y - 50) - t, np.zeros(src.shape)])
            buffered |= np.hypot(dx, dy) <= block['holdout_buffer_m']
        safe = good & ~buffered
        coverage = []
        for block in selected_blocks:
            l, b, r, t = block['bounds_m']
            inside = (x >= l) & (x < r) & (y >= b) & (y < t)
            coverage.append({'block_id': block['id'], 'spatial_holdout': block['spatial_holdout'],
                             'qa_pass_cells': int((good & inside).sum()), 'nonreserved_buffer_safe_cells': int((safe & inside).sum())})
        return {'qa_pass_cells': int(good.sum()), 'reserved_cells': int((good & reserved).sum()),
                'holdout_and_buffer_excluded_cells': int((good & buffered).sum()), 'nonreserved_buffer_safe_cells': int(safe.sum()),
                'blocks_at_least50_safe_cells': sum(b['nonreserved_buffer_safe_cells'] >= 50 for b in coverage), 'blocks': coverage,
                'row_address': 'zero-based raster row/column on fixed pilot100m grid', 'raster_shape': list(src.shape),
                'raster_transform': list(src.transform), 'epsg': pilot['epsg'], 'cell_area_m2': 10000}


class Ledger:
    """Persist conservative reservations so interruptions cannot reset the cap."""
    def __init__(self, output, state):
        self.output, self.state = output, state

    def save(self):
        write(self.output / 'state.json', self.state)

    def reserve(self, amount):
        if self.state['network_bytes_charged'] + amount > MAX_BYTES:
            raise CampaignStop('Cumulative network byte budget exhausted.')
        self.state['network_bytes_charged'] += amount
        self.save()

    def settle(self, reservation, actual, request_count, protected=False):
        self.state['network_bytes_charged'] += actual - reservation
        key = 'protected_http_requests' if protected else 'public_http_requests'
        self.state[key] += request_count
        self.save()

    def public_json(self, params, path):
        if path.exists():
            source = json.loads(path.with_suffix('.source.json').read_text())
            if source['sha256'] != eco.digest(path) or source['request'] != params:
                raise ValueError('Cached public metadata identity changed.')
            return json.loads(path.read_text())
        response_cap = 2 * 1024**2
        reserve = response_cap + STREAM_CHUNK_BYTES
        self.reserve(reserve)
        actual = 0
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.get(geo.CMR_URL, params=params, stream=True, allow_redirects=False,
                                 headers={'Accept-Encoding': 'identity'}, timeout=(10, 35)) as response:
                    if response.status_code != 200:
                        raise eco.AcquisitionError(f'Anonymous CMR returned HTTP {response.status_code}.')
                    payload = bytearray()
                    for chunk in response.iter_content(65536):
                        actual += len(chunk)
                        if actual > response_cap:
                            raise eco.AcquisitionError('CMR response exceeded the2MiB metadata cap.')
                        payload.extend(chunk)
                    body = json.loads(payload)
            write(path, body)
            write(path.with_suffix('.source.json'), {'sha256': eco.digest(path), 'request': params,
                                                   'source_url': geo.CMR_URL, 'network_bytes': actual})
            return body
        finally:
            self.settle(reserve, actual, 1)


class CampaignDownloader(eco.NasaDownloader):
    def __init__(self, ledger, token=None):
        super().__init__(max_bytes=MAX_BYTES, max_requests=4000, token=token)
        self.ledger = ledger

    def download(self, url, destination, kind='cog'):
        path = Path(destination)
        cached = path.is_file() and path.with_suffix('.download.json').is_file()
        # The parent downloader checks caps after each received chunk. Reserve
        # that final possible chunk too, so the global transfer cap stays hard.
        reserve = 0 if cached else (32 if kind == 'cog' else 4) * 1024**2 + STREAM_CHUNK_BYTES
        self.ledger.reserve(reserve)
        old_bytes, old_requests = self.bytes, self.requests
        self.max_bytes = self.bytes + MAX_BYTES - self.ledger.state['network_bytes_charged'] + reserve
        try:
            return super().download(url, destination, kind)
        finally:
            self.ledger.settle(reserve, self.bytes - old_bytes, self.requests - old_requests, protected=True)


def campaign_counts(records):
    dates = defaultdict(set)
    for record in records:
        if record.get('qualifying_independent_date'):
            dates[f"{record['pilot_id']}:{record['temporal_split']}"].add(record['utc_date'])
    return {f'{pilot}:{split}': sorted(dates[f'{pilot}:{split}']) for pilot in PILOTS for split in SPLITS}


def next_candidate(plan, state, records):
    dates = campaign_counts(records)
    keys = [f'{pilot}:{split}' for split in SPLITS for pilot in PILOTS]
    for offset in range(len(keys)):
        position = (state['rotation'] + offset) % len(keys)
        key = keys[position]
        if len(dates[key]) >= TARGETS[key.split(':')[1]]:
            continue
        cursor = state['queue_cursors'].get(key, 0)
        while cursor < len(plan['queues'][key]):
            record = plan['queues'][key][cursor]
            cursor += 1
            state['queue_cursors'][key] = cursor
            if record['utc_date'] in dates[key]:
                state['same_qualifying_date_skips'] += 1
                continue
            state['rotation'] = (position + 1) % len(keys)
            return record
    return None


def acquire_one(record, plan, root, output, downloader, ledger):
    pilot = plan['pilot_areas'][record['pilot_id']]
    result = {key: record[key] for key in ('pilot_id', 'granule_concept_id', 'granule_title', 'time_start', 'utc_date', 'temporal_split')}
    result.update(granule_id=record['granule_concept_id'], title=record['granule_title'],
                  acquisition_group=record['identity']['acquisition_group'], selection=record['selection'],
                  obstruction=record['obstruction'], source_screen_pass=False, training_eligible=False,
                  qualifying_independent_date=False, status='started')
    if frozen_split(record['time_start']) != record['temporal_split']:
        raise ValueError('Reserved label or split mismatch.')
    metadata = output / 'public_metadata'
    target = record['granule_title']
    body = ledger.public_json(geo.discovery_params(target), metadata / ('geo_' + record['granule_concept_id'] + '.json'))
    items = body.get('items', [])
    if int(body.get('hits', -1)) != len(items):
        result['status'] = 'geo_discovery_incomplete'
        return result
    discovery = geo.select_geo_candidate(items, target)
    result['geo_discovery'] = discovery
    if discovery['status'] != 'Matched':
        result['status'] = 'geo_identity_unresolved'
        return result
    if discovery['identity']['build'] != geo.parse_identity(target)['build']:
        result['status'] = 'geo_thermal_processing_build_mismatch'
        return result
    dmrpp = root / 'cache/ecostress_v2/geometry' / (discovery['identity']['granule_name'] + '.h5.dmrpp')
    result['geo_download'] = downloader.download(discovery['dmrpp_url'], dmrpp, kind='geolocation_metadata')
    result['geolocation'] = geo.parse_geolocation_dmrpp(dmrpp.read_bytes(), discovery['identity']['granule_name'])
    if not result['geolocation']['accepted']:
        result['status'] = 'geo_' + result['geolocation']['qa_label'].lower()
        return result
    # Precache bounded anonymous UMM so the existing inspector makes no unmetered request.
    path = root / 'cache/ecostress_v2/metadata' / (record['granule_concept_id'] + '.json')
    if not path.exists():
        thermal_body = ledger.public_json({'concept_id': record['granule_concept_id']}, metadata / ('thermal_' + record['granule_concept_id'] + '.json'))
        if len(thermal_body.get('items', [])) != 1 or thermal_body['items'][0]['meta']['concept-id'] != record['granule_concept_id']:
            result['status'] = 'thermal_cmr_identity_unresolved'
            return result
        write(path, thermal_body['items'][0])
    summary = eco.inspect_granule(record, pilot, downloader, root / 'cache/ecostress_v2', output / 'labels')
    raster_path = output / 'labels' / record['pilot_id'] / target / 'engineering_labels.tif'
    coverage = spatial_coverage(raster_path, pilot, plan['spatial_blocks'])
    result.update(thermal_summary=summary, raster_path=str(raster_path), raster_sha256=eco.digest(raster_path),
                  spatial_coverage=coverage, source_screen_pass=coverage['qa_pass_cells'] > 0,
                  status='source_screen_pass' if coverage['qa_pass_cells'] > 0 else 'zero_thermal_qa_cells',
                  qualifying_independent_date=coverage['nonreserved_buffer_safe_cells'] >= 200 and coverage['blocks_at_least50_safe_cells'] >= 2,
                  pending_training_gates=['Independent spatial registration evidence', 'Unresolved obstruction evidence where NotListed',
                                          'Exact per-row solar/surface/optical/weather/station/thermal-memory joins',
                                          'Independent land fraction and water fraction', 'Frozen final row sampling and split verification'])
    return result


def run(root, output, batch_size):
    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        _run_batch(root, output, batch_size)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_batch(root, output, batch_size):
    plan = json.loads((output / 'plan.json').read_text())
    plan_hash = eco.digest(output / 'plan.json')
    for path, digest in plan['source_hashes'].items():
        if eco.digest(path) != digest:
            raise ValueError('Frozen source or acquisition code changed; review before resuming.')
    state_path = output / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'version': VERSION, 'plan_sha256': plan_hash, 'network_bytes_charged': 0, 'protected_http_requests': 0, 'public_http_requests': 0,
        'candidate_attempts': 0, 'active_seconds': 0, 'queue_cursors': {}, 'rotation': 0,
        'same_qualifying_date_skips': 0, 'inflight': None, 'campaign_started_epoch': time.time()}
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'version': VERSION, 'phase': 'night', 'plan_sha256': plan_hash, 'records': [],
        'training_eligible': False, 'selection_rule': plan['selection_rule'], 'source_counts': plan['source_counts']}
    if state.get('plan_sha256') != plan_hash or manifest.get('plan_sha256') != plan_hash:
        raise ValueError('Frozen plan changed or legacy state lacks its binding; preserve the original runner and audit.')
    ledger = Ledger(output, state)
    # A result is committed before inflight clears. Recover that narrow window
    # by acknowledging the already committed identity without reacquisition.
    inflight = state.get('inflight')
    if inflight and any((r['pilot_id'], r['granule_concept_id']) ==
                        (inflight['pilot_id'], inflight['granule_concept_id']) for r in manifest['records']):
        state['inflight'] = None
        ledger.save()
    remaining_seconds = MAX_SECONDS - (time.time() - state['campaign_started_epoch'])
    if remaining_seconds <= 0:
        manifest['status'] = 'campaign_wall_time_exhausted'
        write(manifest_path, manifest)
        return
    def deadline_handler(*_):
        raise CampaignStop('Cumulative campaign wall-time budget exhausted.')
    signal.signal(signal.SIGALRM, deadline_handler)
    signal.alarm(max(1, int(remaining_seconds)))
    downloader = None
    started = time.monotonic()
    spent_initial = state['active_seconds']
    stop = 'batch_complete'
    for _ in range(batch_size):
        if ((state['candidate_attempts'] >= MAX_CANDIDATES and not state.get('inflight'))
                or state['network_bytes_charged'] >= MAX_BYTES
                or spent_initial + time.monotonic() - started >= MAX_SECONDS):
            stop = 'campaign_budget_exhausted'
            break
        record = state.get('inflight') or next_candidate(plan, state, manifest['records'])
        if record is None:
            stop = 'targets_reached_or_frozen_queues_exhausted'
            break
        if state.get('inflight') is None:
            state['candidate_attempts'] += 1
            state['inflight'] = record
            ledger.save()
        try:
            if downloader is None:
                downloader = CampaignDownloader(ledger)
            result = acquire_one(record, plan, root, output, downloader, ledger)
        except Exception as error:
            result = {key: record[key] for key in ('pilot_id', 'granule_concept_id', 'granule_title', 'time_start', 'utc_date', 'temporal_split')}
            result.update(granule_id=record['granule_concept_id'], acquisition_group=record['identity']['acquisition_group'],
                          status='bounded_acquisition_error', error_type=type(error).__name__,
                          source_screen_pass=False, training_eligible=False, qualifying_independent_date=False)
            if isinstance(error, CampaignStop):
                stop = 'campaign_budget_exhausted'
        manifest['records'] = [r for r in manifest['records'] if r['granule_concept_id'] != result['granule_concept_id'] or r['pilot_id'] != result['pilot_id']]
        manifest['records'].append(result)
        state['active_seconds'] = spent_initial + time.monotonic() - started
        manifest.update(qualifying_dates=campaign_counts(manifest['records']), status=stop,
                        counters={key: value for key, value in state.items() if key not in ('inflight', 'queue_cursors')},
                        result_status_counts=dict(Counter(r['status'] for r in manifest['records'])),
                        updated_utc=datetime.now(timezone.utc).isoformat())
        write(manifest_path, manifest)
        state['inflight'] = None
        ledger.save()
        print(json.dumps({'attempt': state['candidate_attempts'], 'pilot': record['pilot_id'], 'split': record['temporal_split'],
                          'date': record['utc_date'], 'status': result['status'],
                          'safe_cells': result.get('spatial_coverage', {}).get('nonreserved_buffer_safe_cells', 0),
                          'charged_mib': round(state['network_bytes_charged'] / 1024**2, 2),
                          'qualifying_dates': {key: len(value) for key, value in manifest['qualifying_dates'].items()}}), flush=True)
        if stop == 'campaign_budget_exhausted':
            break
    state['active_seconds'] = spent_initial + time.monotonic() - started
    ledger.save()
    if state['candidate_attempts'] >= MAX_CANDIDATES and not state.get('inflight'):
        stop = 'campaign_candidate_cap_reached'
    elif state['network_bytes_charged'] >= MAX_BYTES:
        stop = 'campaign_byte_cap_reached'
    manifest.update(status=stop, qualifying_dates=campaign_counts(manifest['records']),
                    counters={key: value for key, value in state.items() if key not in ('inflight', 'queue_cursors')})
    write(manifest_path, manifest)
    signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run'))
    parser.add_argument('--root', type=Path, default=Path('/opt/lst-pilot'))
    parser.add_argument('--output', type=Path, default=Path('/opt/lst-pilot/runs/option_b_retrain_20260909/acquisition_night'))
    parser.add_argument('--batch-size', type=int, default=20)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if not 1 <= args.batch_size <= 200:
        parser.error('Batch size must be1..200.')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'plan':
            plan = build_plan(args.root, args.output)
            print(json.dumps({'plan': str(args.output / 'plan.json'), 'queues': plan['source_counts']['queue_lengths'], 'protected_downloads': 0}), flush=True)
        else:
            run(args.root, args.output, args.batch_size)


if __name__ == '__main__':
    main()
