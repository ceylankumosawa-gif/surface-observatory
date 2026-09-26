"""Bounded official ERA5-Land ARCO extraction, with no training or label reads.

Plan reads identity/location/time columns only. Extraction selects the nearest
native 0.1-degree cell and the exact floor UTC hour; it never substitutes a
nearby land cell or interpolates time. Reanalysis is retrospectively available.
Actual store schemas must pass metadata checks; synthetic tests are not proof
that an authenticated live store supplies every requested variable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from numcodecs import get_codec


DOC = 'https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data'
PUG = 'https://confluence.ecmwf.int/spaces/CKB/pages/536218894'
NOTEBOOK = 'https://raw.githubusercontent.com/ecmwf-training/dss-notebooks/main/datasets/reanalysis-era5-land/arco-access.ipynb'
HOST = 'https://arco.datastores.ecmwf.int'
GROUPS = {'skin': ('043', 'skin-temperature'), 'air': ('007', '2m-temperature'),
          'snow': ('030', 'snow'), 'soil_temperature': ('006', 'soil-temperature'),
          'soil_water': ('005', 'soil-water')}
VARIABLES = {
    'skin': {'names': ['skt', 'skin_temperature'], 'param': 235, 'unit': 'K',
             'output': 'era5_land_skin_temperature_c', 'offset': -273.15},
    'air': {'names': ['t2m', '2t', '2m_temperature'], 'param': 167, 'unit': 'K',
            'output': 'era5_land_air_temperature_c', 'offset': -273.15},
    'snow': {'names': ['sd', 'snow_depth_water_equivalent'], 'param': 141,
             'unit': 'm of water equivalent', 'output': 'era5_land_snow_water_equivalent_m', 'offset': 0},
    'soil_temperature': {'names': ['stl1', 'soil_temperature_level_1'], 'param': 139,
                         'unit': 'K', 'output': 'era5_land_soil_temperature_0_7cm_c', 'offset': -273.15},
    'soil_water': {'names': ['swvl1', 'volumetric_soil_water_layer_1'], 'param': 39,
                   'unit': 'm**3 m**-3', 'output': 'era5_land_soil_moisture_0_7cm_m3_m3', 'offset': 0},
}
COLUMNS = ['sample_id', 'region_id', 'datetime_utc', 'latitude', 'longitude']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def store_url(group):
    number, name = GROUPS[group]
    return f'{HOST}/cadl-arco-geo-{number}/arco/reanalysis_era5_land/sfc-{name}/geoChunked.zarr'


def prepare_points(frame, period):
    data = frame[COLUMNS].copy().reset_index(drop=True)
    if len(data) == 0 or len(data) > 100000 or data.sample_id.isna().any() or data.sample_id.duplicated().any():
        raise ValueError('Require 1–100000 uniquely identified input rows.')
    if data.region_id.isna().any() or not isinstance(data.datetime_utc.dtype, pd.DatetimeTZDtype):
        raise ValueError('Require explicit timezone-aware source timestamps and region IDs.')
    data['datetime_utc'] = pd.to_datetime(data.datetime_utc, utc=True)
    years = [2021, 2022] if period == 'fit' else [2021, 2022, 2023, 2024]
    if data.datetime_utc.isna().any() or not data.datetime_utc.dt.year.isin(years).all():
        raise ValueError('Input dates exceed the authorized period; never request 2025.')
    coords = data[['latitude', 'longitude']].to_numpy(float)
    if not np.isfinite(coords).all() or np.any(np.abs(coords[:, 0]) > 90) or np.any(np.abs(coords[:, 1]) > 180):
        raise ValueError('Invalid requested geographic coordinates.')
    data['era5_land_valid_time_utc'] = data.datetime_utc.dt.floor('h')
    data['era5_land_time_age_minutes'] = (data.datetime_utc - data.era5_land_valid_time_utc).dt.total_seconds() / 60
    return data


def make_plan(input_path, output, period, *, max_bytes=1024**3, max_requests=2000):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new plan directory; existing plans are immutable.')
    before = sha(input_path)
    points = prepare_points(pd.read_parquet(input_path, columns=COLUMNS), period)
    if sha(input_path) != before:
        raise ValueError('Input changed while preparing the request.')
    output.mkdir(parents=True)
    points.to_parquet(output / 'points.parquet', index=False)
    manifest = {'dataset': 'ERA5-Land', 'official_documentation': [DOC, PUG, NOTEBOOK],
                'input_path': str(Path(input_path).resolve()), 'input_sha256': before,
                'columns_decoded': COLUMNS, 'period': period, 'rows': len(points),
                'points_sha256': sha(output / 'points.parquet'), 'source_sha256': sha(__file__),
                'years': sorted(points.datetime_utc.dt.year.unique().tolist()),
                'requested_utc_hours': int(points.era5_land_valid_time_utc.nunique()),
                'pilot_count': int(points.region_id.nunique()), 'variables': VARIABLES,
                'stores': {k: store_url(k) for k in GROUPS},
                'selection': 'nearest native cell; exact UTC floor hour; no interpolation or land substitution',
                'retrospective_reanalysis_not_realtime_availability': True,
                'skin_is_modeled_coarse_support_not_observed_100m_label': True,
                'server_chunks_may_include_other_weather_times': True,
                'snow_rule': 'Require sd / GRIB141 water equivalent; never relabel sde physical depth.',
                'limits': {'transfer_bytes': int(max_bytes), 'requests': int(max_requests),
                           'object_bytes': 64*1024**2, 'wall_seconds': 1800,
                           'decoded_chunk_bytes': 64*1024**2},
                'no_new_thermal_labels': True, 'no_2025_requested_rows': True}
    save(output / 'plan.json', manifest)
    return manifest


def load_token(path=None):
    token = os.environ.get('CDSAPI_KEY')
    if path:
        path = Path(path)
        if path.stat().st_mode & 0o077:
            raise ValueError('CDS credential file must be private (mode0600).')
        raw = path.read_text()
        if path.suffix == '.json':
            token = json.loads(raw).get('token')
        else:
            token = next((line.split(':', 1)[1].strip() for line in raw.splitlines()
                          if line.strip().startswith('key:')), None)
    if not isinstance(token, str) or not token or len(token) > 8192 or any(c.isspace() for c in token):
        raise ValueError('A valid private CDS API token is required; Earthdata credentials are separate.')
    return token


class BoundedHTTP:
    """Sequential, hash-checked cache; request/object reservations survive interruption."""
    def __init__(self, output, token, plan_sha, limits):
        self.root = Path(output)
        self.cache = self.root / 'http_cache'
        self.cache.mkdir(exist_ok=True)
        self.path = self.root / 'transfer_ledger.json'
        self.limits = limits
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({'Authorization': 'Bearer ' + token,
                                     'Accept-Encoding': 'identity'})
        self.state = (json.loads(self.path.read_text()) if self.path.exists() else
                      {'plan_sha256': plan_sha, 'started_epoch': time.time(),
                       'requests': 0, 'charged_bytes': 0, 'objects': {}})
        if self.state['plan_sha256'] != plan_sha:
            raise ValueError('Resume plan hash changed.')
        save(self.path, self.state)

    def get(self, url):
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.netloc != 'arco.datastores.ecmwf.int' or parsed.query or parsed.fragment:
            raise ValueError('Only the fixed official ECMWF source is permitted.')
        if not any(url.startswith(store_url(k) + '/') for k in GROUPS):
            raise ValueError('Unplanned source store.')
        key = hashlib.sha256(url.encode()).hexdigest()
        file = self.cache / key
        previous = self.state['objects'].get(url)
        if previous and previous.get('status') == 'complete':
            if not file.is_file() or sha(file) != previous['sha256']:
                raise ValueError('Cached source checksum mismatch.')
            return file.read_bytes()
        if self.state['requests'] >= self.limits['requests'] or time.time()-self.state['started_epoch'] > self.limits['wall_seconds']:
            raise RuntimeError('Metadata/data request or wall-time budget reached.')
        self.state['requests'] += 1
        save(self.path, self.state)
        try:
            response = self.session.get(url, stream=True, allow_redirects=False, timeout=(15, 60))
        except requests.RequestException:
            raise RuntimeError('Official source connection failed; no credentials were logged.') from None
        with response:
            if response.status_code in (401, 403):
                raise PermissionError('Official ERA5-Land source rejected CDS authorization.')
            if response.status_code != 200:
                raise RuntimeError(f'Official source HTTP{response.status_code}; no redirect or substitute source used.')
            length = response.headers.get('Content-Length')
            reserved = int(length) if length is not None else self.limits['object_bytes']
            if reserved < 0 or reserved > self.limits['object_bytes'] or self.state['charged_bytes'] + reserved > self.limits['transfer_bytes']:
                raise RuntimeError('Protected-transfer or object-size bound would be exceeded.')
            self.state['charged_bytes'] += reserved
            self.state['objects'][url] = {'status': 'reserved', 'reserved_bytes': reserved}
            save(self.path, self.state)
            body = bytearray()
            for block in response.iter_content(65536):
                if len(body) + len(block) > reserved:
                    raise RuntimeError('Source body exceeded its reserved length.')
                body.extend(block)
            file.write_bytes(body)
            self.state['objects'][url] = {'status': 'complete', 'bytes': len(body),
                                          'charged_bytes': reserved, 'sha256': sha(file),
                                          'etag': response.headers.get('ETag'),
                                          'last_modified': response.headers.get('Last-Modified')}
            save(self.path, self.state)
            return bytes(body)


def decode_chunk(raw, meta, index):
    dtype = np.dtype(meta['dtype'])
    if dtype.kind not in 'iuf' or meta.get('order', 'C') not in ('C', 'F'):
        raise ValueError('Unsupported nonnumeric Zarr encoding.')
    chunk_shape = tuple(meta['chunks'])
    expected = int(np.prod(chunk_shape)) * dtype.itemsize
    if expected > 64*1024**2:
        raise ValueError('Decoded source chunk exceeds memory bound.')
    value = get_codec(meta['compressor']).decode(raw) if meta.get('compressor') else raw
    for codec in reversed(meta.get('filters') or []):
        value = get_codec(codec).decode(value)
    array = np.frombuffer(value, dtype=dtype)
    edge = tuple(min(c, n-i*c) for i, c, n in zip(index, chunk_shape, meta['shape']))
    shape = chunk_shape if array.size == int(np.prod(chunk_shape)) else edge
    if array.size != int(np.prod(shape)):
        raise ValueError('Decoded Zarr chunk length does not match metadata.')
    return array.reshape(shape, order=meta.get('order', 'C'))


def physical_values(raw, attrs, meta):
    values = np.asarray(raw, dtype=float).copy()
    # A Zarr storage fill of zero is not an independent CF missing-data flag.
    # Existing decoded zero-valued snow/moisture and coordinates remain valid.
    fill = attrs.get('_FillValue')
    if fill is not None and not isinstance(fill, str):
        values[values == float(fill)] = np.nan
    return values*float(attrs.get('scale_factor', 1)) + float(attrs.get('add_offset', 0))


class ZarrReader:
    def __init__(self, client, group):
        self.client, self.url = client, store_url(group)
        blob = client.get(self.url + '/.zmetadata')
        root = json.loads(blob)
        if root.get('zarr_consolidated_format') != 1:
            raise ValueError('Require verified consolidated Zarr v2 metadata.')
        self.meta = root['metadata']
        self.metadata_sha = hashlib.sha256(blob).hexdigest()

    def array(self, name):
        meta, attrs = self.meta[name+'/.zarray'], self.meta[name+'/.zattrs']
        if meta.get('zarr_format') != 2:
            raise ValueError('Unsupported live array format; preflight review required.')
        return meta, attrs

    def read(self, name, positions):
        meta, attrs = self.array(name)
        pos = np.asarray(positions, dtype=np.int64)
        if pos.ndim != 2 or pos.shape[1] != len(meta['shape']) or np.any(pos < 0) or np.any(pos >= np.asarray(meta['shape'])):
            raise ValueError('Array selection outside source dimensions.')
        buckets = defaultdict(list)
        for i, row in enumerate(pos):
            buckets[tuple(row // np.asarray(meta['chunks']))].append(i)
        result = np.full(len(pos), np.nan)
        for index, ids in sorted(buckets.items()):
            suffix = meta.get('dimension_separator', '.').join(map(str, index))
            raw = self.client.get(self.url + '/' + name + '/' + suffix)
            chunk = decode_chunk(raw, meta, index)
            local = pos[ids] - np.asarray(index)*np.asarray(meta['chunks'])
            result[ids] = physical_values(chunk[tuple(local.T)], attrs, meta)
        return result

    def coordinate(self, name):
        meta, attrs = self.array(name)
        if len(meta['shape']) != 1 or meta['shape'][0] > 1000000:
            raise ValueError('Coordinate size/shape exceeds fixed preflight bound.')
        values = np.full(meta['shape'][0], np.nan)
        for first in range(0, len(values), meta['chunks'][0]):
            index = first//meta['chunks'][0]
            raw = self.client.get(self.url+'/'+name+'/'+str(index))
            chunk = decode_chunk(raw,meta,(index,))
            stop = min(first+meta['chunks'][0],len(values))
            values[first:stop] = physical_values(chunk[:stop-first],attrs,meta)
        if not np.isfinite(values).all():
            raise ValueError('Missing source coordinates.')
        return values, attrs


def decode_times(values, attrs):
    if attrs.get('calendar', 'standard') not in ('standard', 'gregorian', 'proleptic_gregorian'):
        raise ValueError('Unverified time calendar.')
    match = re.fullmatch(r'(hours|seconds|days|minutes) since (.+)', attrs.get('units', ''))
    if not match:
        raise ValueError('Unverified CF time units.')
    origin = pd.Timestamp(match[2])
    origin = origin.tz_localize('UTC') if origin.tz is None else origin.tz_convert('UTC')
    return origin + pd.to_timedelta(values, unit={'hours':'h','seconds':'s','days':'D','minutes':'m'}[match[1]])


def nearest_indexes(axis, requested, *, longitude=False):
    axis, requested = np.asarray(axis, float), np.asarray(requested, float)
    if axis.ndim != 1 or not np.isfinite(axis).all() or not np.allclose(np.diff(axis), .1, rtol=0, atol=1e-6):
        raise ValueError('Require the documented increasing native 0.1-degree grid.')
    insertion = np.searchsorted(axis,requested)
    candidates = np.column_stack([np.clip(insertion-1,0,len(axis)-1),
                                  np.clip(insertion,0,len(axis)-1),
                                  np.zeros(len(requested),dtype=int),
                                  np.full(len(requested),len(axis)-1,dtype=int)])
    candidates.sort(axis=1)
    delta = axis[candidates]-requested[:,None]
    distance = np.abs((delta+180)%360-180) if longitude else np.abs(delta)
    tied = np.isclose(distance,distance.min(axis=1)[:,None],atol=1e-10,rtol=0)
    chosen = candidates[np.arange(len(requested)),tied.argmax(axis=1)]
    if np.any(distance.min(axis=1)>.050001):
        raise ValueError('Requested point is outside the native grid support.')
    return chosen


def variable_name(reader, group):
    spec = VARIABLES[group]
    found = [n for n in spec['names'] if n+'/.zarray' in reader.meta]
    if len(found) != 1:
        raise ValueError(f'{group}: requested variable absent/ambiguous; no substitute variable permitted.')
    name = found[0]
    _, attrs = reader.array(name)
    if attrs.get('_ARRAY_DIMENSIONS') != ['time', 'latitude', 'longitude']:
        raise ValueError('Variable dimension order differs from verified source contract.')
    param = attrs.get('GRIB_paramId')
    if param is not None and int(param) != spec['param']:
        raise ValueError('Unexpected ECMWF parameter identity.')
    units = attrs.get('units')
    if group == 'snow':
        # GRIB141 proves water-equivalent identity even if units are simply m.
        valid_unit = units == 'm of water equivalent' or (units == 'm' and param is not None and int(param) == 141)
    elif group == 'soil_water':
        valid_unit = units in ('m**3 m**-3', 'm3 m-3', 'm^3 m^-3', 'm3/m3')
    else:
        valid_unit = units == spec['unit']
    if not valid_unit:
        raise ValueError(f'{group}: source units are not verified.')
    return name, attrs


def extract(output, credential_file=None, *, metadata_only=False):
    output = Path(output)
    plan_path = output / 'plan.json'
    plan = json.loads(plan_path.read_text())
    if sha(__file__) != plan['source_sha256'] or sha(output/'points.parquet') != plan['points_sha256'] or sha(plan['input_path']) != plan['input_sha256']:
        raise ValueError('Frozen source, input or request points changed.')
    if (output/'completion.json').exists():
        raise FileExistsError('Completed extraction is immutable.')
    points = pd.read_parquet(output/'points.parquet')
    client = BoundedHTTP(output, load_token(credential_file), sha(plan_path), plan['limits'])
    reference_grid = None
    records, readers = {}, {}
    for group in GROUPS:
        reader = ZarrReader(client, group)
        name, attrs = variable_name(reader, group)
        lat, latattrs = reader.coordinate('latitude')
        lon, lonattrs = reader.coordinate('longitude')
        tv, ta = reader.coordinate('time')
        times = decode_times(tv, ta)
        if latattrs.get('units') not in ('degrees_north', 'degree_north') or lonattrs.get('units') not in ('degrees_east', 'degree_east'):
            raise ValueError('Unverified spatial coordinate units.')
        grid = (lat, lon, times.asi8)
        if reference_grid is None:
            reference_grid = grid
        elif not all(np.array_equal(a,b) for a,b in zip(grid, reference_grid)):
            raise ValueError('Variables do not share identical native grid and valid-time axes.')
        ilat = nearest_indexes(lat, points.latitude)
        ilon = nearest_indexes(lon, points.longitude, longitude=True)
        wanted = pd.DatetimeIndex(points.era5_land_valid_time_utc)
        itime = times.get_indexer(wanted)
        if np.any(itime < 0):
            raise ValueError('Requested exact floor hour missing; no temporal interpolation used.')
        positions = np.column_stack([itime, ilat, ilon])
        meta, _ = reader.array(name)
        chunk_ids = sorted(set(map(tuple, positions//np.asarray(meta['chunks']))))
        records[group] = {'store': reader.url, 'variable': name, 'metadata_sha256': reader.metadata_sha,
                          'attributes': attrs, 'array': meta, 'coordinate_attributes': {'latitude':latattrs,'longitude':lonattrs,'time':ta},
                          'requested_chunk_count': len(chunk_ids), 'chunk_indexes': [list(c) for c in chunk_ids],
                          'chunk_time_coverage': [{'first': times[c[0]*meta['chunks'][0]].isoformat(),
                            'last': times[min((c[0]+1)*meta['chunks'][0],len(times))-1].isoformat()}
                            for c in sorted(set((c[0],) for c in chunk_ids))]}
        readers[group] = (reader, name, positions)
    preflight = {'plan_sha256': sha(plan_path), 'stores': records, 'metadata_only': True,
                 'all_variable_native_grids_identical': True, 'retrospective_reanalysis': True}
    if (output/'preflight.json').exists():
        if json.loads((output/'preflight.json').read_text()) != preflight:
            raise ValueError('Source metadata changed since frozen preflight.')
    else:
        save(output/'preflight.json', preflight)
    if metadata_only:
        return {'status':'preflight_complete','requested_variable_chunks':sum(r['requested_chunk_count'] for r in records.values())}
    values = points.copy()
    values['era5_land_grid_latitude'] = reference_grid[0][ilat]
    values['era5_land_grid_longitude'] = reference_grid[1][ilon]
    values['era5_land_latitude_delta_degrees'] = values.era5_land_grid_latitude-values.latitude
    values['era5_land_longitude_delta_degrees'] = (values.era5_land_grid_longitude-values.longitude+180)%360-180
    for group, (reader, name, positions) in readers.items():
        native = reader.read(name, positions)
        values[VARIABLES[group]['output']] = native + VARIABLES[group]['offset']
    columns = [v['output'] for v in VARIABLES.values()]
    values['era5_land_complete'] = np.isfinite(values[columns].to_numpy()).all(axis=1)
    values['era5_land_status'] = np.where(values.era5_land_complete, 'native_cell_available', 'native_cell_missing_no_land_substitution')
    values['era5_land_preflight_sha256'] = sha(output/'preflight.json')
    values.to_parquet(output/'values.parquet', index=False)
    if sha(__file__) != plan['source_sha256'] or sha(plan_path) != client.state['plan_sha256']:
        raise ValueError('Source/plan changed during extraction.')
    summary = {'status':'complete','rows':len(values),'available_rows':int(values.era5_land_complete.sum()),
               'plan_sha256':sha(plan_path),'preflight_sha256':sha(output/'preflight.json'),
               'output_sha256':sha(output/'values.parquet'),'ledger_sha256':sha(client.path),
               'requests':client.state['requests'],'charged_transfer_bytes':client.state['charged_bytes'],
               'valid_time_only_not_operational_publication_causality':True,
               'no_new_thermal_labels':True,'no_training':True,'no_2025_requested_rows':True}
    save(output/'completion.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--period', choices=['fit','evaluation'], default='fit')
    for command in ['preflight','extract']:
        p = sub.add_parser(command)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--credential-file', type=Path)
    args = ap.parse_args()
    if args.command == 'plan':
        result = make_plan(args.input, args.output, args.period)
        print(json.dumps({k:result[k] for k in ['rows','years','pilot_count','requested_utc_hours']}))
    else:
        print(json.dumps(extract(args.output,args.credential_file,metadata_only=args.command=='preflight')))


if __name__ == '__main__':
    main()
