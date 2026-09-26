"""Freeze and run a16-granule /512MiB 2021 high-resolution engineering batch.

Credentials remain in the dedicated worker process. No fit, calibration or
production action is implemented. ASTER native-QA success is not registration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import re
import resource
from urllib.parse import urljoin, urlsplit

import pandas as pd
import requests
import numpy as np
from pyproj import Transformer
import rasterio

from . import aster, ecostress as eco, ecostress_geo as geo, highres_inventory as inv
from .option_b_acquire import spatial_coverage

VERSION = 'multisensor-highres-engineering-20260910-v1'
CAP = 512 * 1024**2
MAX_GRANULES = 16


def allowed(url, kind, redirect=False):
    collection = {'cog': 'ECO_L2T_LSTE.002', 'geolocation_metadata': 'ECO_L1B_GEO.002',
                  'aster_cog': 'AST_08.004'}.get(kind)
    if collection is None:
        return False
    p = urlsplit(url)
    if p.scheme != 'https' or p.port not in (None, 443) or p.username or p.password or p.fragment:
        return False
    prefix = '/lp-prod-protected/' + collection + '/'
    extension = '.h5.dmrpp' if kind == 'geolocation_metadata' else '.tif'
    if not p.path.endswith(extension):
        return False
    if p.hostname == eco.HOST:
        return p.path.startswith(prefix) and (redirect or not p.query)
    if redirect and p.hostname == eco.CDN_HOST:
        return (p.path.startswith(prefix) or bool(re.match(
            r'^/s3-[0-9a-f]{32}/lp-prod-protected\.s3\.us-west-2\.amazonaws\.com/'
            + re.escape(collection) + '/', p.path)))
    return False


class Downloader:
    """Durable conservative reservations cannot be reset by restarting the batch."""
    def __init__(self, output, token=None):
        self.output = Path(output)
        self.path = self.output / 'download_ledger.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'charged_bytes': 0, 'requests': 0, 'completed': [], 'interrupted_reservations_are_retained': True}
        self._token = token if token is not None else json.loads(eco.CREDENTIAL.read_text())['access_token']
        self.session = requests.Session()
        self.session.trust_env = False

    def save(self):
        eco.save_json(self.path, self.state)

    def download(self, url, destination, kind='cog'):
        if not allowed(url, kind):
            raise eco.AcquisitionError('Protected asset is outside the frozen collections.')
        destination = Path(destination)
        manifest = destination.with_suffix('.download.json')
        if destination.is_file() and manifest.is_file():
            record = json.loads(manifest.read_text())
            if record['url'] != url or record['sha256'] != eco.digest(destination):
                raise eco.AcquisitionError('Existing source identity/checksum differs.')
            return {**record, 'cache_reused': True}
        file_cap = (4 if kind == 'geolocation_metadata' else 8 if kind == 'aster_cog' else 32) * 1024**2
        reservation = file_cap + 65536
        if self.state['charged_bytes'] + reservation > CAP:
            raise eco.AcquisitionError('Engineering512MiB download ceiling reached.')
        self.state['charged_bytes'] += reservation
        self.save()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.partial')
        current, total, h = url, 0, hashlib.sha256()
        try:
            for _ in range(5):
                if not allowed(current, kind, redirect=True) or self.state['requests'] >= 300:
                    raise eco.AcquisitionError('Redirect or protected request limit rejected.')
                self.state['requests'] += 1
                self.save()
                headers = {'Accept-Encoding': 'identity'}
                if urlsplit(current).hostname == eco.HOST:
                    headers['Authorization'] = 'Bearer ' + self._token
                self.session.cookies.clear()
                with self.session.get(current, headers=headers, timeout=(10, 45), stream=True,
                                      allow_redirects=False) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        current = urljoin(current, response.headers.get('Location', ''))
                        continue
                    if response.status_code != 200:
                        raise eco.AcquisitionError(f'NASA asset HTTP{response.status_code}; details suppressed.')
                    with temporary.open('wb') as handle:
                        for chunk in response.iter_content(65536):
                            total += len(chunk)
                            if total > file_cap:
                                raise eco.AcquisitionError('Protected asset file ceiling reached.')
                            h.update(chunk)
                            handle.write(chunk)
                    with temporary.open('rb') as handle:
                        header = handle.read(128)
                    if ((kind == 'geolocation_metadata' and not header.lstrip().startswith(b'<'))
                            or (kind != 'geolocation_metadata' and header[:4] not in
                                (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'))):
                        raise eco.AcquisitionError('Unexpected protected payload format; content suppressed.')
                    temporary.replace(destination)
                    record = {'url': url, 'bytes': total, 'sha256': h.hexdigest(), 'kind': kind}
                    eco.save_json(manifest, record)
                    self.state['completed'].append(record)
                    self.state['charged_bytes'] += total - reservation
                    self.save()
                    return record
            raise eco.AcquisitionError('NASA redirect ceiling reached.')
        except requests.RequestException:
            raise eco.AcquisitionError('NASA network failure; signed URLs and credentials suppressed.') from None
        finally:
            temporary.unlink(missing_ok=True)


def public_metadata(params, output, name):
    """A bounded anonymous UMM request; callers prepare at most16 candidates."""
    with requests.Session() as session:
        session.trust_env = False
        with session.get(geo.CMR_URL, params=params, timeout=(10, 45), stream=True, allow_redirects=False) as response:
            if response.status_code != 200:
                raise eco.AcquisitionError(f'Public CMR HTTP{response.status_code}.')
            body = bytearray()
            for chunk in response.iter_content(65536):
                body.extend(chunk)
                if len(body) > 4 * 1024**2:
                    raise eco.AcquisitionError('Public CMR response ceiling reached.')
    value = json.loads(body)
    path = Path(output) / 'metadata' / (name + '.json')
    eco.save_json(path, value)
    eco.save_json(path.with_suffix('.source.json'), {'params': params, 'bytes': len(body),
                                                   'body_sha256': hashlib.sha256(body).hexdigest()})
    return value


def select_records(records, acquisitions):
    """Round across6 sensor/phase/pilot queues using metadata ranks only."""
    lookup = {r['granule_concept_id']: r for r in records}
    queues = defaultdict(list)
    for acquisition in acquisitions:
        if acquisition['temporal_split'] != 'fit':
            continue
        product, phase = acquisition['product'], acquisition['phase']
        if (product, phase) not in (('ecostress_v2', 'day'), ('aster_v4', 'day'), ('aster_v4', 'night')):
            continue
        options = [lookup[g] for g in acquisition['granule_concept_ids']
                   if g in lookup and pd.Timestamp(lookup[g]['time_start']).year == 2021]
        latest = {}
        for record in options:
            if product == 'ecostress_v2':
                ident = geo.parse_identity(record['granule_title'])
                key = (ident['scene'], ident['tile'])
                processing = (int(ident['build']), int(ident['counter']))
            else:
                key = record['time_start']
                processing = (aster.identity(record['granule_title'])['processing_utc'],)
            if key not in latest or processing > latest[key][0]:
                latest[key] = (processing, record)
        if not latest:
            continue
        selected = min((r for _, r in latest.values()), key=lambda r: (-r['metadata_overlap_m2'], r['granule_title']))
        queues[(selected['pilot_id'], product, phase)].append({**selected, 'selection_rank': acquisition['rank'],
                                                             'selection_stratum': acquisition['stratum']})
    keys = [(p, source, phase) for source, phase in (('ecostress_v2', 'day'), ('aster_v4', 'night'), ('aster_v4', 'day'))
            for p in inv.PILOTS]
    result, seen_dates = [], defaultdict(set)
    while len(result) < MAX_GRANULES and any(queues.values()):
        for key in keys:
            while queues[key] and queues[key][0]['utc_date'] in seen_dates[key]:
                queues[key].pop(0)
            if queues[key] and len(result) < MAX_GRANULES:
                row = queues[key].pop(0)
                seen_dates[key].add(row['utc_date'])
                result.append(row)
    return result


def prepare(root, inventory_path, output):
    if output.exists():
        raise ValueError('Use a new engineering directory.')
    summary = json.loads((inventory_path / 'inventory.json').read_text())
    required = [q for q in summary['queries'] if q['params']['temporal'].startswith('2021-')]
    if len(required) != 16 or not all(q['complete'] for q in required):
        raise ValueError('All16 London/Sioux2021 DAY quarter/product queries must be complete.')
    records = json.loads((inventory_path / 'granules.json').read_text())
    acquisitions = json.loads((inventory_path / 'calendar_candidates.json').read_text())['acquisitions']
    selected = select_records(records, acquisitions)
    output.mkdir(parents=True)
    obstruction_path = root / 'cache/ecostress_geo/public_obstruction/obstruction_dfceb418ba111fae4db89816442ffd540b7c57b0e1e9cc51b6163a1efee5b26a.txt'
    obstruction = geo.parse_obstruction_list(obstruction_path.read_bytes(), complete=True)
    if obstruction['status'] != 'Parsed' or obstruction['errors']:
        raise ValueError('Complete obstruction index required.')
    for index, record in enumerate(selected):
        body = public_metadata({'concept_id': record['granule_concept_id']}, output, f'{index:02d}_l2')
        if len(body.get('items', [])) != 1:
            raise ValueError('Expected one L2 CMR record.')
        envelope = body['items'][0]
        record['cmr_revision_id'] = envelope['meta']['revision-id']
        record['cmr_envelope'] = envelope
        if record['product'] == 'aster_v4':
            aster.asset_links(envelope, record)
            token = aster.NAME.fullmatch(record['granule_title'])['time']
            companion = public_metadata({'short_name': 'AST_L1T', 'version': '004', 'provider': 'LPCLOUD',
                'readable_granule_name': f'AST_L1T_004{token}_*',
                'options[readable_granule_name][pattern]': 'true', 'page_size': 10}, output, f'{index:02d}_l1t')
            record['l1t_metadata_candidates'] = companion.get('items', [])
            record['l1t_does_not_certify_l2_registration'] = True
        else:
            if envelope['umm']['GranuleUR'] != record['granule_title'] or envelope['meta']['concept-id'] != record['granule_concept_id']:
                raise ValueError('ECOSTRESS CMR identity mismatch.')
            eco.asset_links(envelope['umm'])
            record['obstruction'] = geo.obstruction_status(obstruction, record['granule_title'])
            body = public_metadata(geo.discovery_params(record['granule_title']), output, f'{index:02d}_geo')
            record['geo_match'] = geo.select_geo_candidate(body.get('items', []), record['granule_title'])
    from . import option_b_acquire
    sources = [Path(__file__), Path(aster.__file__), Path(eco.__file__), Path(geo.__file__), Path(inv.__file__),
               Path(option_b_acquire.__file__), inventory_path / 'inventory.json', inventory_path / 'granules.json',
               inventory_path / 'calendar_candidates.json', inventory_path / 'spatial_blocks.json',
               root / 'pilot/areas_resolved.json', obstruction_path]
    plan = {'version': VERSION, 'period': '2021 only; complete source shards inside a partial wider inventory',
            'records': selected, 'limits': {'thermal_granules': 16, 'protected_bytes_including_companions': CAP},
            'source_hashes': {str(p): eco.digest(p) for p in sources},
            'selection_rule': 'Round-robin ECOSTRESSday/ASTERnight/ASTERday × London/Sioux queues, each retains frozen calendar/hour rank; one date per queue, latest processing per scene/tile then greatest metadata overlap. All sixteen identities fixed before GEO/thermal reads. No replacements based on thermal values.',
            'pilot_areas': {p['id']: p for p in json.loads((root / 'pilot/areas_resolved.json').read_text())['areas'] if p['id'] in inv.PILOTS},
            'spatial_blocks': json.loads((inventory_path / 'spatial_blocks.json').read_text()),
            'engineering_only': True, 'training_eligible': False, 'no2023_2024_2025_labels': True}
    eco.save_json(output / 'plan.json', plan)
    return plan


def write_fit_support(result, pilot, blocks, native_path):
    """Exclude an extra2 native-pixel diagonals beyond whole-cell1km buffers.

    The extra margin covers a source pixel intersecting a destination cell but
    extending beyond it. This does not correct a geolocation error; that remains
    a separate source gate. Native and target CRS are metres in this experiment.
    """
    with rasterio.open(native_path) as source:
        project = Transformer.from_crs(source.crs, pilot['epsg'], always_xy=True)
        diameters = []
        for column in (0, source.width / 2, source.width):
            for row in (0, source.height / 2, source.height):
                points = [project.transform(*(source.transform * (column + dx, row + dy)))
                          for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))]
                diameters.extend(np.hypot(a[0]-b[0], a[1]-b[1]) for a in points for b in points)
        margin = float(np.ceil(2 * max(diameters) + 10))
        native_transform = list(source.transform)
    with rasterio.open(result['raster_path']) as labels:
        profile = labels.profile.copy()
        good = np.isfinite(labels.read(1))
        rr, cc = np.indices(labels.shape)
        x, y = labels.transform * (cc + .5, rr + .5)
    excluded = np.zeros(good.shape, dtype=bool)
    for block in blocks:
        if block['id'].startswith(pilot['id'] + '_') and block['spatial_holdout']:
            left, bottom, right, top = block['bounds_m']
            dx = np.maximum.reduce([left - (x + 50), (x - 50) - right, np.zeros(good.shape)])
            dy = np.maximum.reduce([bottom - (y + 50), (y - 50) - top, np.zeros(good.shape)])
            excluded |= np.hypot(dx, dy) <= block['holdout_buffer_m'] + margin
    safe = good & ~excluded
    output = Path(result['raster_path']).with_name('native_fit_support.tif')
    profile.update(count=1, dtype='uint8', nodata=None)
    with rasterio.open(output, 'w', **profile) as dest:
        dest.write(safe.astype('uint8'), 1)
        dest.set_band_description(1, 'native_fit_support_pass')
    return {'path': str(output), 'sha256': eco.digest(output), 'safe_cells': int(safe.sum()),
            'extra_native_support_margin_m': margin, 'native_transform': native_transform,
            'method': 'Whole100m cell outside fixed reserved blocks+1000m Euclidean buffer+2 transformed native-pixel diagonals+10m; source pixel diameters evaluated at9 scene positions.',
            'separate_from_source_quality_and_training_eligibility': True}


def run(root, output):
    plan_path = output / 'plan.json'
    plan = json.loads(plan_path.read_text())
    if plan['version'] != VERSION or not 1 <= len(plan['records']) <= MAX_GRANULES:
        raise ValueError('Invalid frozen engineering plan.')
    for path, sha in plan['source_hashes'].items():
        if eco.digest(path) != sha:
            raise ValueError('Frozen source checksum changed; do not resume.')
    path = output / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {'version': VERSION,
        'plan_sha256': eco.digest(plan_path), 'engineering_only': True, 'training_eligible': False, 'records': []}
    if manifest['plan_sha256'] != eco.digest(plan_path):
        raise ValueError('Plan changed after collection began.')
    client = Downloader(output)
    finished = {r['granule_id'] for r in manifest['records']}
    for record in plan['records']:
        if record['granule_concept_id'] in finished:
            continue
        pilot = plan['pilot_areas'][record['pilot_id']]
        result = {'granule_id': record['granule_concept_id'], 'pilot_id': record['pilot_id'],
                  'title': record['granule_title'], 'product': record['product'], 'actual_phase': record['actual_phase'],
                  'time_start': record['time_start'], 'thermal_read_attempted': False, 'training_eligible': False}
        try:
            if record['product'] == 'ecostress_v2':
                obstruction, match = record['obstruction'], record['geo_match']
                if obstruction.get('obstructed') is True or obstruction['status'] == 'Unknown' or match['status'] != 'Matched':
                    raise eco.AcquisitionError('ECOSTRESS obstruction/GEO matching gate failed before pixels.')
                geo_path = root / 'cache/ecostress_v2/geometry' / (match['identity']['granule_name'] + '.h5.dmrpp')
                client.download(match['dmrpp_url'], geo_path, kind='geolocation_metadata')
                qa = geo.parse_geolocation_dmrpp(geo_path.read_bytes(), match['identity']['granule_name'])
                result['geolocation'] = qa
                if not qa['accepted']:
                    raise eco.AcquisitionError('ECOSTRESS GEO is not positivelyGood/Best; thermal pixels not fetched.')
                cache = root / 'cache/multisensor_20260910/ecostress_v2'
                eco.save_json(cache / 'metadata' / (record['granule_concept_id'] + '.json'), record['cmr_envelope'])
                result['thermal_read_attempted'] = True
                summary = eco.inspect_granule(record, pilot, client, cache, output)
                raster_path = output / record['pilot_id'] / record['granule_title'] / 'engineering_labels.tif'
                result.update(summary, raster_path=str(raster_path), raster_sha256=eco.digest(raster_path),
                              source_screen_pass=summary['qa_pass_cells'] > 0)
                native_path = cache / record['granule_title'] / 'LST.tif'
                native_layer = 'LST'
            else:
                result['thermal_read_attempted'] = True
                result.update(aster.inspect(record, pilot, record['cmr_envelope'], client, root / 'cache/aster_v4', output))
                result['native_qa_pass'] = result['qa_pass_cells'] > 0
                result['source_screen_pass'] = False
                native_path = root / 'cache/aster_v4' / record['granule_title'] / 'SKT.tif'
                native_layer = 'SKT'
            result['spatial_coverage'] = spatial_coverage(result['raster_path'], pilot, plan['spatial_blocks'])
            result['native_fit_support'] = write_fit_support(result, pilot, plan['spatial_blocks'], native_path)
            result['label_source_sha256'] = result['downloads'][native_layer]['sha256']
            result['label_valid_fraction_band'] = 2
            result['label_source_geolocation_proof'] = result.get('geolocation', {'status': 'Unverified; independent AST_08 registration required'})
            result['label_source_cloud_proof'] = {'native_qa_rules': result['qa_rules'],
                'independent_cloud_absence_proven': False,
                'note': 'Quality screening is not a proof that every residual cloud is absent.'}
            result['status'] = 'engineering_qa_complete'
        except Exception as error:
            result.update(status='rejected_or_incomplete', error_type=type(error).__name__,
                          reason=str(error) if isinstance(error, eco.AcquisitionError) else 'Engineering adapter needs schema review; external details suppressed.')
        manifest['records'].append(result)
        manifest['download_ledger'] = client.state
        eco.save_json(path, manifest)
        print(json.dumps({k: result.get(k) for k in ('pilot_id', 'product', 'actual_phase', 'status', 'qa_pass_cells', 'reason')}), flush=True)
    manifest['complete'] = len(manifest['records']) == len(plan['records'])
    manifest['all_results_qa_successful'] = all(r['status'] == 'engineering_qa_complete' for r in manifest['records'])
    eco.save_json(path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.action == 'prepare':
        prepare(args.root, args.inventory, args.output)
    else:
        with (args.output / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args.root, args.output)


if __name__ == '__main__':
    main()
