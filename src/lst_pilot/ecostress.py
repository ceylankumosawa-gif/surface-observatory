"""Bounded ECOSTRESS V002 engineering acquisition; never changes serving models.

Tiled V002 COGs already store floating Kelvin, unlike scaled swath HDF SDSs.
Mask the native grid before 100 m aggregation. QC alone is not a cloud mask.
NASA sources and frozen QA settings are recorded in every engineering run.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import time
from urllib.parse import urljoin, urlsplit

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer
from shapely.geometry import box

from .night_inventory import name_identity, PRODUCTS

VERSION = 'ecostress-v2-engineering-20260909-v2'
HOST = 'data.lpdaac.earthdatacloud.nasa.gov'
CDN_HOST = 'd1nklfio7vscoe.cloudfront.net'  # Observed HTTPS redirect from the NASA host, 2026-09-09.
PREFIX = '/lp-prod-protected/ECO_L2T_LSTE.002/'
GEO_PREFIX = '/lp-prod-protected/ECO_L1B_GEO.002/'
CREDENTIAL = Path('/var/lib/lst-data/earthdata/credential.json')
LAYERS = ('cloud', 'water', 'QC', 'view_zenith', 'LST_err', 'LST')
QA_RULES = {'cloud_clear_value': 0, 'cloud_buffer_m': 350,
            'water_land_value': 0, 'qc_low_four_bits': 0,
            'qc_lst_accuracy_codes': [1, 2, 3],
            'lst_uncertainty_max_k': 2.0, 'view_zenith_max_degrees': 30.0,
            'minimum_valid_fraction': 0.9, 'plausibility_kelvin': [150.0, 400.0],
            'nearby_clear_pixels_do_not_prove_cloud_absence': True}


class AcquisitionError(ValueError):
    """Fixed messages only: request errors/headers/URLs may contain credentials."""


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def asset_url_ok(url, redirect=False, kind='cog'):
    p = urlsplit(url)
    if p.scheme != 'https' or p.port not in (None, 443) or p.username or p.password:
        return False
    if p.hostname == HOST:
        path_ok = (p.path.startswith(PREFIX) if kind == 'cog' else
                   kind == 'geolocation_metadata' and p.path.startswith(GEO_PREFIX)
                   and p.path.endswith('.h5.dmrpp'))
        return bool(path_ok) and (redirect or not p.query)
    collection = 'ECO_L2T_LSTE.002' if kind == 'cog' else 'ECO_L1B_GEO.002'
    cdn_path_ok = (p.path.startswith(PREFIX if kind == 'cog' else GEO_PREFIX)
                   or re.match(r'^/s3-[0-9a-f]{32}/lp-prod-protected\.s3\.us-west-2\.amazonaws\.com/'
                               + re.escape(collection) + '/', p.path))
    return redirect and ((p.hostname == CDN_HOST and cdn_path_ok)
                         or (p.hostname or '').endswith('.s3.us-west-2.amazonaws.com')
                         or (p.hostname or '').endswith('.s3.amazonaws.com'))


class NasaDownloader:
    """Token only in private process memory; exact-host auth, bounded streaming."""
    def __init__(self, max_bytes=256 * 1024 * 1024, max_requests=100, token=None):
        self._token = token if token is not None else json.loads(CREDENTIAL.read_text())['access_token']
        self.bytes = self.requests = 0
        self.max_bytes, self.max_requests = max_bytes, max_requests
        self.session = requests.Session()
        self.session.trust_env = False

    def download(self, url, destination, kind='cog'):
        if kind not in ('cog', 'geolocation_metadata') or not asset_url_ok(url, kind=kind):
            raise AcquisitionError('Asset is outside the approved NASA V002 collection.')
        destination = Path(destination)
        manifest = destination.with_suffix('.download.json')
        if destination.is_file() and manifest.is_file():
            prior = json.loads(manifest.read_text())
            if prior['url'] == url and prior['sha256'] == digest(destination):
                return prior
            raise AcquisitionError('Cached asset checksum/identity changed; preserve and review it.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix('.partial')
        current = url
        try:
            for _ in range(5):
                if self.requests >= self.max_requests or self.bytes >= self.max_bytes:
                    raise AcquisitionError('Protected-download request or byte budget reached.')
                if not asset_url_ok(current, redirect=True, kind=kind):
                    raise AcquisitionError('NASA access redirected outside approved download hosts.')
                self.requests += 1
                headers = {'Accept-Encoding': 'identity'}
                if urlsplit(current).hostname == HOST:
                    headers['Authorization'] = 'Bearer ' + self._token
                self.session.cookies.clear()
                with self.session.get(current, headers=headers, stream=True,
                                      allow_redirects=False, timeout=(10, 45)) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        current = urljoin(current, response.headers.get('Location', ''))
                        continue
                    if response.status_code != 200:
                        raise AcquisitionError(f'NASA asset returned HTTP {response.status_code}; no response body is logged.')
                    total = 0
                    h = hashlib.sha256()
                    with temp.open('wb') as handle:
                        for chunk in response.iter_content(64 * 1024):
                            self.bytes += len(chunk)
                            total += len(chunk)
                            file_cap = (32 if kind == 'cog' else 4) * 1024 * 1024
                            if total > file_cap or self.bytes > self.max_bytes:
                                raise AcquisitionError('NASA asset exceeded its file or total download cap.')
                            h.update(chunk)
                            handle.write(chunk)
                    with temp.open('rb') as handle:
                        header = handle.read(128)
                        if kind == 'cog' and header[:4] not in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
                            raise AcquisitionError('Download was not a TIFF; no response content is logged.')
                        if kind == 'geolocation_metadata' and not header.lstrip().startswith(b'<'):
                            raise AcquisitionError('Geolocation metadata was not XML; content is suppressed.')
                    temp.replace(destination)
                    record = {'url': url, 'bytes': total, 'sha256': h.hexdigest()}
                    save_json(manifest, record)
                    return record
            raise AcquisitionError('Too many NASA download redirects.')
        except requests.RequestException:
            raise AcquisitionError('NASA network request failed; credentials and signed URL are suppressed.') from None
        finally:
            temp.unlink(missing_ok=True)


def fetch_umm(concept_id, cache):
    if not re.fullmatch(r'G[0-9]+-LPCLOUD', concept_id):
        raise AcquisitionError('Invalid CMR granule ID.')
    path = Path(cache) / (concept_id + '.json')
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('meta', {}).get('concept-id') != concept_id or 'umm' not in cached:
            raise AcquisitionError('Cached CMR metadata lacks verified identity and revision.')
        return cached
    with requests.Session() as session:
        session.trust_env = False
        response = session.get('https://cmr.earthdata.nasa.gov/search/granules.umm_json',
                               params={'concept_id': concept_id}, timeout=(10, 45), allow_redirects=False)
        response.raise_for_status()
        records = response.json()['items']
    if len(records) != 1 or records[0]['meta']['concept-id'] != concept_id:
        raise AcquisitionError('CMR identity mismatch.')
    save_json(path, records[0])
    return records[0]


def asset_links(umm):
    if umm['CollectionReference']['ShortName'] != 'ECO_L2T_LSTE' or umm['CollectionReference']['Version'] != '002':
        raise AcquisitionError('Only ECOSTRESS tiled V002 is implemented.')
    title = umm['GranuleUR']
    links = {}
    for entry in umm['RelatedUrls']:
        if entry.get('Type') != 'GET DATA' or not asset_url_ok(entry['URL']):
            continue
        for layer in LAYERS:
            if urlsplit(entry['URL']).path.endswith('/' + title + '_' + layer + '.tif'):
                if layer in links:
                    raise AcquisitionError('Ambiguous CMR asset link.')
                links[layer] = entry['URL']
    if set(links) != set(LAYERS):
        raise AcquisitionError('Required cloud/water/QC/view/uncertainty/LST layers are missing.')
    return links


def native_valid(raw, rules=QA_RULES, spacing_m=(70, 70)):
    finite_qc = np.isfinite(raw['QC'])
    qc = np.where(finite_qc, raw['QC'], 65535).astype(np.uint16)
    cloudy_or_missing = ((raw['cloud'] != rules['cloud_clear_value'])
                         | ~np.isfinite(raw['LST']) | (raw['LST'] <= 0))
    cloud_distance = (distance_transform_edt(~cloudy_or_missing, sampling=spacing_m)
                      if cloudy_or_missing.any() else np.full(raw['LST'].shape, np.inf))
    cloud_buffer = cloud_distance <= rules['cloud_buffer_m']
    checks = {
        'clear_after_cloud_buffer': ~cloud_buffer,
        'land': raw['water'] == rules['water_land_value'],
        'qc': finite_qc & ((qc & 15) == rules['qc_low_four_bits']),
        'qc_lst_accuracy': finite_qc & np.isin((qc >> 14) & 3, rules['qc_lst_accuracy_codes']),
        'view': np.isfinite(raw['view_zenith']) & (np.abs(raw['view_zenith']) <= rules['view_zenith_max_degrees']),
        'uncertainty': np.isfinite(raw['LST_err']) & (raw['LST_err'] > 0) & (raw['LST_err'] <= rules['lst_uncertainty_max_k']),
        'temperature_finite_plausible': np.isfinite(raw['LST']) & (raw['LST'] >= rules['plausibility_kelvin'][0]) & (raw['LST'] <= rules['plausibility_kelvin'][1]),
    }
    valid = np.logical_and.reduce(list(checks.values()))
    return valid, checks


def aggregate(raw, source_transform, source_crs, bounds, epsg):
    valid, checks = native_valid(raw, spacing_m=(abs(source_transform.e), abs(source_transform.a)))
    left, bottom, right, top = bounds
    dst_transform = from_origin(left, top, 100, 100)
    dst_shape = (round((top - bottom) / 100), round((right - left) / 100))
    def warp(value, nodata=None, resampling=Resampling.average):
        dest = np.full(dst_shape, np.nan, dtype='float32')
        reproject(np.asarray(value, dtype='float32'), dest, src_transform=source_transform,
                  src_crs=source_crs, dst_transform=dst_transform, dst_crs=f'EPSG:{epsg}',
                  src_nodata=nodata, dst_nodata=np.nan, resampling=resampling,
                  num_threads=1, tolerance=0)
        return dest
    fraction = warp(valid.astype('float32'))
    # Require a whole destination cell inside source support; average-resampling
    # alone would call a partly covered edge cell 100% valid.
    to_src = Transformer.from_crs(epsg, source_crs, always_xy=True)
    rows, cols = np.indices(dst_shape)
    edge_ok = np.ones(dst_shape, dtype=bool)
    height, width = raw['LST'].shape
    inv = ~source_transform
    for dx, dy in ((0, 0), (0, 100), (100, 0), (100, 100)):
        sx, sy = to_src.transform(left + cols * 100 + dx, top - rows * 100 - dy)
        cc, rr = inv * (sx, sy)
        edge_ok &= (cc >= 0) & (cc <= width) & (rr >= 0) & (rr <= height)
    accepted = edge_ok & np.isfinite(fraction) & (fraction >= QA_RULES['minimum_valid_fraction'])
    lst = warp(np.where(valid, raw['LST'] - 273.15, np.nan), np.nan)
    error = warp(np.where(valid, raw['LST_err'], np.nan), np.nan, Resampling.max)
    return {'lst_c': np.where(accepted, lst, np.nan), 'valid_fraction': fraction,
            'max_source_lst_error_k': np.where(accepted, error, np.nan)}, dst_transform, checks


def inspect_granule(record, pilot, client, cache, output):
    timestamp = pd.Timestamp(record['time_start'])
    if timestamp.year not in (2021, 2022, 2023):
        raise AcquisitionError('Engineering acquisition refuses reserved 2024/2025 or other years.')
    if record['product'] != 'ecostress_v2' or record['footprint_qa']['status'] != 'metadata_consistent_pixels_unverified':
        raise AcquisitionError('Granule is not a metadata-consistent V002 candidate.')
    metadata_path = Path(cache) / 'metadata' / (record['granule_concept_id'] + '.json')
    envelope = fetch_umm(record['granule_concept_id'], metadata_path.parent)
    umm = envelope['umm']
    if umm['GranuleUR'] != record['granule_title']:
        raise AcquisitionError('Granule title changed after planning.')
    name_identity(umm['GranuleUR'], PRODUCTS['ecostress_v2'], timestamp.to_pydatetime())
    links = asset_links(umm)
    title = umm['GranuleUR']
    files = {layer: Path(cache) / title / (layer + '.tif') for layer in LAYERS}
    downloads = {layer: client.download(links[layer], files[layer]) for layer in LAYERS}
    with ExitStack() as stack:
        sources = {layer: stack.enter_context(rasterio.open(files[layer])) for layer in LAYERS}
        base = sources['LST']
        if base.width * base.height > 3_000_000 or base.count != 1 or base.crs is None:
            raise AcquisitionError('Unexpected COG dimensions or CRS.')
        if any(s.crs != base.crs or s.transform != base.transform or s.shape != base.shape for s in sources.values()):
            raise AcquisitionError('Quality/temperature rasters do not share one native grid.')
        for key in ('LST', 'LST_err', 'view_zenith'):
            src = sources[key]
            if not np.issubdtype(np.dtype(src.dtypes[0]), np.floating) or src.scales != (1.0,) or src.offsets != (0.0,):
                raise AcquisitionError('Unsupported tiled scaling; never apply swath scale factors by guess.')
        if not (65 <= abs(base.res[0]) <= 75 and 65 <= abs(base.res[1]) <= 75):
            raise AcquisitionError('Unexpected native thermal grid spacing.')
        actual = transform_bounds(base.crs, f"EPSG:{pilot['epsg']}", *base.bounds, densify_pts=21)
        if not box(*actual).intersects(box(*pilot['extent_m'])):
            raise AcquisitionError('Actual raster does not intersect the pilot.')
        raw = {layer: src.read(1) for layer, src in sources.items()}
        # QC=0 can mean good. Do not apply QC nodata=0 in isolation.
        for layer in ('LST', 'LST_err', 'view_zenith'):
            raw[layer] = raw[layer].astype('float32')
            if sources[layer].nodata is not None:
                raw[layer][raw[layer] == sources[layer].nodata] = np.nan
        values, transform, checks = aggregate(raw, base.transform, base.crs, pilot['extent_m'], pilot['epsg'])
        native_geometry = {'crs': str(base.crs), 'bounds': list(base.bounds), 'shape': list(base.shape),
                           'resolution_m': list(base.res), 'dtypes': {k:s.dtypes[0] for k,s in sources.items()}}
    directory = Path(output) / record['pilot_id'] / title
    directory.mkdir(parents=True, exist_ok=True)
    with rasterio.open(directory / 'engineering_labels.tif', 'w', driver='GTiff', width=values['lst_c'].shape[1],
                       height=values['lst_c'].shape[0], count=3, dtype='float32', crs=f"EPSG:{pilot['epsg']}",
                       transform=transform, nodata=np.nan, compress='deflate') as dest:
        for band, (key, value) in enumerate(values.items(), 1):
            dest.write(value.astype('float32'), band)
            dest.set_band_description(band, key)
    from pvlib.solarposition import get_solarposition
    lon, lat = pilot.get('resolved_center_lonlat', pilot['center_lonlat'])
    elevation = float(get_solarposition(pd.DatetimeIndex([timestamp]), lat, lon)['elevation'].iloc[0])
    good = np.isfinite(values['lst_c'])
    accepted_values = values['lst_c'][good]
    summary = {'granule_id': record['granule_concept_id'], 'title': title, 'pilot_id': record['pilot_id'],
               'cmr_meta': envelope['meta'], 'cmr_metadata_sha256': digest(metadata_path),
               'time_start': timestamp.isoformat(), 'solar_elevation_centre': elevation,
               'native_geometry': native_geometry, 'downloads': downloads, 'qa_rules': QA_RULES,
               'native_checks_pass_count': {k:int(v.sum()) for k,v in checks.items()},
               'grid_cells': int(good.size), 'qa_pass_cells': int(good.sum()),
               'lst_c_percentiles': np.percentile(accepted_values, [0, 5, 50, 95, 100]).tolist() if len(accepted_values) else None,
               'engineering_only': True, 'training_eligible': False,
               'pending': ['Independent geolocation and historical obstruction checks',
                           'Spatial holdout and buffer exclusion', 'Optical/weather/station joins',
                           'Frozen fitting addendum and minimum independent sample counts'],
               'thermal_values_are_labels_not_predictors': True}
    save_json(directory / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-mib', type=int, default=256)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if not 1 <= args.max_mib <= 1024:
        parser.error('max-mib must be1..1024')
    plan = json.loads(args.plan.read_text())
    if not plan.get('engineering_only') or not 1 <= len(plan['records']) <= 16:
        parser.error('A frozen engineering-only plan with1..16 granules is required.')
    if args.output.exists():
        parser.error('Use a new output directory; engineering results are immutable.')
    args.output.mkdir(parents=True)
    (args.output / 'source_ecostress.py').write_bytes(Path(__file__).read_bytes())
    save_json(args.output / 'plan.json', plan)
    areas = {p['id']:p for p in json.loads((args.root/'pilot/areas_resolved.json').read_text())['areas']}
    client = NasaDownloader(max_bytes=args.max_mib * 1024 * 1024, max_requests=120)
    started = time.monotonic()
    audit = {'version': VERSION, 'plan_sha256': digest(args.plan), 'module_sha256': digest(__file__),
             'runtime_versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'rasterio': rasterio.__version__},
             'engineering_only': True, 'results': [], 'failures': []}
    for record in plan['records']:
        try:
            result = inspect_granule(record, areas[record['pilot_id']], client,
                                     args.root/'cache/ecostress_v2', args.output)
            audit['results'].append(result)
            print(json.dumps({k:result[k] for k in ('pilot_id','title','qa_pass_cells','solar_elevation_centre')}), flush=True)
        except Exception as error:
            # Never expose external exception text or request details.
            audit['failures'].append({'granule_id':record.get('granule_concept_id'),
                                      'error_type':type(error).__name__,
                                      'reason':str(error) if isinstance(error, AcquisitionError) else 'Unexpected acquisition failure; details withheld to protect credentials.'})
            print(json.dumps(audit['failures'][-1]), flush=True)
        audit.update(bytes_downloaded=client.bytes, protected_http_requests=client.requests, seconds=time.monotonic()-started)
        save_json(args.output/'audit.json', audit)
        if client.bytes >= client.max_bytes or client.requests >= client.max_requests:
            break
    print(json.dumps({'output':str(args.output),'completed':len(audit['results']), 'failures':len(audit['failures']),
                      'bytes_downloaded':client.bytes, 'model_fitted':False}), flush=True)
    processed = {r['granule_id'] for r in audit['results'] + audit['failures']}
    audit['skipped'] = [{'granule_id': r['granule_concept_id'], 'reason': 'Run budget exhausted.'}
                        for r in plan['records'] if r['granule_concept_id'] not in processed]
    audit['complete'] = not audit['skipped'] and not audit['failures']
    save_json(args.output/'audit.json', audit)


if __name__ == '__main__':
    main()
