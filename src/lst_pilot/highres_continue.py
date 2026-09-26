"""Continue incomplete anonymous high-resolution metadata shards; no pixels."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from shapely.geometry import shape

from . import highres_inventory as h
from . import night_inventory as n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('New continuation output required.')
    args.output.mkdir(parents=True)
    original = json.loads((args.prior / 'inventory.json').read_text())
    old = json.loads((args.prior / 'granules.json').read_text())
    catalog_path = args.root / 'web/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    pilots = {p['properties']['id']: p for p in catalog['pilots']['features']}
    client = h.TimedClient(160, 256, 1200)
    queries, records = [], list(old)
    for previous in original['queries']:
        if previous['complete']:
            queries.append(previous)
            continue
        params = previous['params']
        start, end = params['temporal'].split(',')
        rows, audit = n.inventory_query(client, pilots[previous['pilot_id']], previous['product'],
                                       previous['day_night_flag'], start[:10], end[:10], 100, 12)
        keys = {(r['pilot_id'], r['granule_concept_id']) for r in rows}
        if audit['complete']:
            records = [r for r in records if not (r['pilot_id'] == previous['pilot_id']
                       and r['product'] == previous['product'] and r['day_night_flag'] == previous['day_night_flag']
                       and start[:10] <= r['utc_date'] <= end[:10])] + rows
        else:
            records = [r for r in records if (r['pilot_id'], r['granule_concept_id']) not in keys] + rows
        queries.append(audit)
        n.write_json(args.output / 'checkpoint.json', {'queries': queries, 'http_audit': client.audit})
        print(json.dumps({k: audit[k] for k in ('pilot_id', 'product', 'complete', 'cmr_hits', 'unique_utc_dates')}), flush=True)
    enriched = h.enrich(records, pilots)
    complete = len(queries) == 48 and all(q['complete'] for q in queries)
    n.write_json(args.output / 'granules.json', enriched)
    n.write_json(args.output / 'calendar_candidates.json', {'inventory_complete': complete,
        'frozen_for_label_acquisition': False, 'acquisitions': h.candidate_queue(enriched)})
    (args.output / 'spatial_blocks.json').write_bytes((args.prior / 'spatial_blocks.json').read_bytes())
    counts = []
    for pilot in h.PILOTS:
        for product in ('ecostress_v2', 'aster_v4'):
            for phase in ('day', 'night', 'twilight_or_mixed'):
                subset = [r for r in enriched if (r['pilot_id'], r['product'], r['actual_phase']) == (pilot, product, phase)]
                counts.append({'pilot_id': pilot, 'product': product, 'phase': phase,
                    'granules': len(subset), 'utc_dates': len({r['utc_date'] for r in subset}),
                    'acquisitions': len({r['identity']['acquisition_group'] for r in subset}),
                    'dates_by_split': {s: len({r['utc_date'] for r in subset if r['temporal_split'] == s})
                                       for s in ('fit', 'development', 'calibration')}})
    climate_counts = []
    for pilot_id, area in pilots.items():
        if pilot_id in h.PILOTS:
            continue
        for product, flag in (('ecostress_v2', 'DAY'), ('aster_v4', 'DAY'), ('aster_v4', 'NIGHT')):
            try:
                _, hits = client.get('granules.json', {'collection_concept_id': n.PRODUCTS[product]['concept_id'],
                    'bounding_box': ','.join(map(str, shape(area['geometry']).bounds)),
                    'day_night_flag': flag, 'temporal': '2021-01-01T00:00:00Z,2023-12-31T23:59:59Z', 'page_size': 1})
                climate_counts.append({'pilot_id': pilot_id, 'product': product, 'phase_catalog': flag,
                                       'granule_hits': hits, 'unique_dates': None, 'count_only': True})
            except Exception as error:
                climate_counts.append({'pilot_id': pilot_id, 'product': product, 'phase_catalog': flag,
                                       'granule_hits': None, 'count_only': True, 'error_type': type(error).__name__})
    source_paths = [args.prior / name for name in ('inventory.json', 'granules.json', 'calendar_candidates.json', 'spatial_blocks.json')]
    source_paths += [Path(__file__), Path(h.__file__), Path(n.__file__), catalog_path]
    report = {'version': 'highres-metadata-continuation-20260910-v1', 'complete': complete,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'queries': queries, 'counts': counts,
              'source_hashes': {str(p): n.sha256_file(p) for p in source_paths},
              'other_pilots_count_only': climate_counts, 'protected_downloads': 0,
              'public_http_requests': client.requests, 'metadata_bytes': client.bytes,
              'http_audit': client.audit, 'prior_requests': original['public_http_requests'],
              'prior_metadata_bytes': original['metadata_bytes'],
              'limits_this_continuation': {'requests': 160, 'bytes': 256*1024**2, 'wall_seconds': 1200}}
    n.write_json(args.output / 'inventory.json', report)
    print(json.dumps({'complete': complete, 'counts': counts, 'requests': client.requests, 'bytes': client.bytes}), flush=True)
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
