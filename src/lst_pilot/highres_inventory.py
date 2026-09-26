"""Anonymous daytime ECOSTRESS / ASTER preflight. Never opens protected assets.

The old nighttime inventories and permanent spatial split remain immutable.
This inventory adds actual solar geometry and independently counted UTC dates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pvlib.solarposition import get_solarposition
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform

from . import night_inventory as inventory

VERSION = 'highres-metadata-20260910-v1'
PILOTS = ('greater_london', 'sioux_falls')
SEED = 'lst-multisensor-20260910-v1'


class TimedClient(inventory.MetadataClient):
    def __init__(self, requests=120, mib=64, seconds=1200):
        super().__init__(requests, mib * 1024**2)
        self.deadline = time.monotonic() + seconds

    def get(self, endpoint, params):
        if time.monotonic() >= self.deadline:
            raise inventory.PreflightError('Metadata wall-time budget reached; inventory is partial')
        return super().get(endpoint, params)


def enrich(records, pilots):
    """Solar geometry is calculated before labels, including all pilot corners."""
    output = []
    for pilot_id in sorted({r['pilot_id'] for r in records}):
        area = pilots[pilot_id]
        props = area['properties']
        left, bottom, right, top = props['extent_m']
        to_geo = Transformer.from_crs(props['epsg'], 4326, always_xy=True)
        centre = props['center']
        points = [centre] + [to_geo.transform(x, y) for x, y in
                             ((left, bottom), (left, top), (right, bottom), (right, top))]
        subset = [r for r in records if r['pilot_id'] == pilot_id]
        stamps = pd.DatetimeIndex(sorted({r['time_start'] for r in subset}))
        solar = np.array([get_solarposition(stamps, lat, lon)['elevation'].to_numpy()
                          for lon, lat in points])
        by_time = {t.isoformat(): solar[:, i] for i, t in enumerate(stamps)}
        project = Transformer.from_crs(4326, props['epsg'], always_xy=True).transform
        for row in subset:
            stamp = pd.Timestamp(row['time_start'])
            if stamp.year not in (2021, 2022, 2023):
                raise ValueError('Reserved years cannot enter this inventory.')
            values = by_time[stamp.isoformat()]
            phase = ('day' if values.min() >= 6 else 'night' if values.max() <= -6 else 'twilight_or_mixed')
            hour = (stamp.hour + stamp.minute / 60 + stamp.second / 3600 + centre[0] / 15) % 24
            geometry = row['footprint_qa'].get('cmr_geometry')
            overlap = (transform(project, shape(geometry)).intersection(box(left, bottom, right, top)).area
                       if geometry else 0)
            output.append({**row, 'actual_phase': phase, 'solar_elevation_centre': float(values[0]),
                           'solar_elevation_min_centre_corners': float(values.min()),
                           'solar_elevation_max_centre_corners': float(values.max()),
                           'local_solar_hour_approx': hour, 'local_solar_3hour_bin': int(hour // 3),
                           'metadata_overlap_m2': overlap, 'training_eligible': False})
    return sorted(output, key=lambda r: (r['pilot_id'], r['time_start'], r['granule_title']))


def candidate_queue(records):
    """Whole acquisitions, calendar/hour stratification; never sort by LST or errors."""
    groups = defaultdict(list)
    for row in records:
        if (row['footprint_qa']['status'] == 'metadata_consistent_pixels_unverified'
                and row['metadata_overlap_m2'] > 0 and row['actual_phase'] in ('day', 'night')):
            groups[(row['pilot_id'], row['product'], row['identity']['acquisition_group'])].append(row)
    strata = defaultdict(list)
    for (pilot, product, acquisition), rows in groups.items():
        phases = {r['actual_phase'] for r in rows}
        splits = {r['temporal_split'] for r in rows}
        if len(phases) != 1 or len(splits) != 1:
            continue
        first = min(rows, key=lambda r: r['time_start'])
        stratum = (pilot, product, first['actual_phase'], first['temporal_split'],
                   first['utc_date'][:7], first['local_solar_3hour_bin'])
        rank = hashlib.sha256(f'{SEED}:{pilot}:{product}:{acquisition}'.encode()).hexdigest()
        strata[stratum].append({'pilot_id': pilot, 'product': product, 'acquisition_group': acquisition,
                               'utc_dates': sorted({r['utc_date'] for r in rows}),
                               'temporal_split': first['temporal_split'], 'phase': first['actual_phase'],
                               'stratum': list(stratum), 'rank': rank,
                               'granule_concept_ids': sorted({r['granule_concept_id'] for r in rows})})
    result = []
    for stratum, rows in sorted(strata.items()):
        for index, row in enumerate(sorted(rows, key=lambda r: r['rank']), 1):
            result.append({**row, 'rank_within_stratum': index})
    return sorted(result, key=lambda r: (r['rank_within_stratum'], r['stratum'], r['rank']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-requests', type=int, default=120)
    parser.add_argument('--max-mib', type=int, default=64)
    parser.add_argument('--max-seconds', type=int, default=1200)
    args = parser.parse_args(argv)
    if not (1 <= args.max_requests <= 120 and 1 <= args.max_mib <= 64 and 1 <= args.max_seconds <= 1200):
        parser.error('Maximum metadata bounds are 120 requests / 64 MiB / 1200 seconds.')
    if args.output.exists():
        parser.error('Use a new immutable output directory.')
    args.output.mkdir(parents=True)
    root = args.root.resolve()
    catalog_path = root / 'web/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    pilots = {p['properties']['id']: p for p in catalog['pilots']['features'] if p['properties']['id'] in PILOTS}
    prior = root / 'runs/option_b_metadata_merged_2021_2023_20260909'
    registry = json.loads((prior / 'merge_registry.json').read_text())
    if registry['status'] != 'complete_metadata_only' or registry['period'] != ['2021-01-01', '2023-12-31']:
        raise ValueError('Prior complete nighttime inventory is required.')
    source_paths = [catalog_path, prior / 'granules.json', prior / 'merge_registry.json',
                    prior / 'spatial_blocks.json', Path(__file__), Path(inventory.__file__)]
    source_hashes = {str(p): inventory.sha256_file(p) for p in source_paths}
    records = [r for r in json.loads((prior / 'granules.json').read_text())
               if r['pilot_id'] in PILOTS and r['product'] == 'aster_v4' and r['day_night_flag'] == 'NIGHT']
    client = TimedClient(args.max_requests, args.max_mib, args.max_seconds)
    queries, verified, failures = [], {}, []
    for product in ('ecostress_v2', 'aster_v4'):
        try:
            verified[product] = client.verify_product(product)
        except Exception as error:
            failures.append({'product': product, 'error_type': type(error).__name__})
    # One quarter per query makes completion auditable and bounds each shard.
    for year in (2021, 2022, 2023):
        for pilot_id in PILOTS:
            for product in verified:
                for start_month in (1, 4, 7, 10):
                    start = pd.Timestamp(year=year, month=start_month, day=1)
                    end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
                    rows, audit = inventory.inventory_query(client, pilots[pilot_id], product, 'DAY',
                                                            str(start.date()), str(end.date()), 100, 12)
                    records.extend(rows)
                    queries.append(audit)
                    inventory.write_json(args.output / 'checkpoint.json', {'queries': queries, 'http_audit': client.audit})
                    print(json.dumps({k: audit[k] for k in ('pilot_id', 'product', 'cmr_hits', 'complete', 'unique_utc_dates')}), flush=True)
    unique = {}
    for row in records:
        key = (row['pilot_id'], row['product'], row['granule_concept_id'])
        if key in unique and unique[key] != row:
            failures.append({'reason': 'Conflicting metadata for repeated granule', 'identity': list(key)})
        unique[key] = row
    enriched = enrich(list(unique.values()), pilots)
    complete = not failures and len(queries) == 48 and all(q['complete'] for q in queries)
    inventory.write_json(args.output / 'granules.json', enriched)
    inventory.write_json(args.output / 'calendar_candidates.json', {
        'version': VERSION, 'inventory_complete': complete, 'frozen_for_label_acquisition': False,
        'selection_rule': 'rank_within_stratum then pilot/product/phase/split/month/3-hour-bin, stable SHA rank; whole ECOSTRESS orbit or ASTER timestamp group. Freeze latest revision and candidate tiles before any QA reads.',
        'acquisitions': candidate_queue(enriched)})
    inventory.write_json(args.output / 'spatial_blocks.json', json.loads((prior / 'spatial_blocks.json').read_text()))
    counts = []
    for pilot in PILOTS:
        for product in ('ecostress_v2', 'aster_v4'):
            for phase in ('day', 'night', 'twilight_or_mixed'):
                rows = [r for r in enriched if (r['pilot_id'], r['product'], r['actual_phase']) == (pilot, product, phase)]
                counts.append({'pilot_id': pilot, 'product': product, 'phase': phase, 'granules': len(rows),
                               'utc_dates': len({r['utc_date'] for r in rows}),
                               'acquisitions': len({r['identity']['acquisition_group'] for r in rows}),
                               'dates_by_split': {s: len({r['utc_date'] for r in rows if r['temporal_split'] == s})
                                                  for s in ('fit', 'development', 'calibration')}})
    climate_counts = []
    # Broader climate visibility is count-only, subordinate to completing the two-pilot inventory.
    for area in catalog['pilots']['features']:
        if area['properties']['id'] in PILOTS:
            continue
        for product, flag in (('ecostress_v2', 'DAY'), ('aster_v4', 'DAY'), ('aster_v4', 'NIGHT')):
            try:
                _, hits = client.get('granules.json', {
                    'collection_concept_id': inventory.PRODUCTS[product]['concept_id'],
                    'bounding_box': ','.join(map(str, shape(area['geometry']).bounds)),
                    'day_night_flag': flag, 'temporal': '2021-01-01T00:00:00Z,2023-12-31T23:59:59Z',
                    'page_size': 1})
                climate_counts.append({'pilot_id': area['properties']['id'], 'product': product,
                                       'phase_catalog': flag, 'granule_hits': hits,
                                       'unique_dates': None, 'count_only': True})
            except Exception as error:
                climate_counts.append({'pilot_id': area['properties']['id'], 'product': product,
                                       'phase_catalog': flag, 'granule_hits': None,
                                       'count_only': True, 'error_type': type(error).__name__})
    summary = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'complete': complete, 'source_hashes': source_hashes, 'verified_products': verified,
               'queries': queries, 'failures': failures, 'counts': counts,
               'public_http_requests': client.requests, 'metadata_bytes': client.bytes,
               'http_audit': client.audit, 'protected_downloads': 0, 'other_pilots_count_only': climate_counts,
               'limitations': ['Metadata candidate counts are not clear-sky usable dates.',
                              'Fresh validation pilot-dates must be disjoint from previously inspected dates before collection.',
                              'ASTER uses timestamp groups; whole UTC-date grouping is required downstream.',
                              'No 2024/2025 observations; Cabauw remains excluded; no training eligibility conferred.']}
    inventory.write_json(args.output / 'inventory.json', summary)
    print(json.dumps({'complete': complete, 'counts': counts, 'requests': client.requests, 'bytes': client.bytes}), flush=True)
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
