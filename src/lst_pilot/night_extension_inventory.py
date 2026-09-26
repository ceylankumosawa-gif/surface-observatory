"""Offline inventory of unattempted fitting-year ECOSTRESS nights.

Reads frozen catalog and cached geolocation metadata only. This does not
authorize acquisition, query NASA, open thermal arrays or confer label QA.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import pandas as pd

from . import ecostress as eco, ecostress_geo as geo, highres_inventory as h


def select(records, attempted, prior_dates):
    eligible = []
    for r in records:
        if (r['product'] != 'ecostress_v2' or r['temporal_split'] != 'fit'
                or pd.Timestamp(r['time_start']).year not in (2021, 2022)
                or r['solar_elevation_max_centre_corners'] > -6):
            continue
        identity = geo.parse_identity(r['granule_title'])
        if ((r['pilot_id'], identity['orbit']) in attempted
                or (r['pilot_id'], r['utc_date']) in prior_dates):
            continue
        eligible.append(r)
    lookup = {r['granule_concept_id']: r for r in eligible}
    output = []
    for acquisition in h.candidate_queue(eligible):
        latest = {}
        for gid in acquisition['granule_concept_ids']:
            r = lookup[gid]
            identity = geo.parse_identity(r['granule_title'])
            key = (identity['scene'], identity['tile'])
            revision = (int(identity['build']), int(identity['counter']))
            if key not in latest or revision > latest[key][0]:
                latest[key] = (revision, r)
        row = dict(min((r for _, r in latest.values()),
                       key=lambda r: (-r['metadata_overlap_m2'], r['granule_title'])))
        row['selection'] = {k: acquisition[k] for k in ('rank', 'rank_within_stratum', 'stratum')}
        output.append(row)
    return sorted(output, key=lambda r: (r['selection']['rank_within_stratum'], r['utc_date'][:7],
                                        r['local_solar_3hour_bin'], r['selection']['rank']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root, out = args.root, args.output
    if out.exists():
        raise ValueError('A new output directory is required.')
    prior = root/'runs/option_b_metadata_merged_2021_2023_20260909'
    old = root/'runs/option_b_retrain_20260909/acquisition_night/manifest.json'
    registry_path = root/'runs/multisensor_20260910_v1/registry_v1/prior_inspected_dates.json'
    obstruction_path = root/'cache/ecostress_geo/public_obstruction/obstruction_dfceb418ba111fae4db89816442ffd540b7c57b0e1e9cc51b6163a1efee5b26a.txt'
    catalog_path = root/'web/catalog.json'
    if json.loads((prior/'merge_registry.json').read_text())['status'] != 'complete_metadata_only':
        raise ValueError('Complete frozen archive inventory required.')
    paths = [Path(__file__), Path(h.__file__), Path(geo.__file__), prior/'merge_registry.json',
             prior/'granules.json', old, registry_path, obstruction_path, catalog_path]
    sources = {str(p): eco.digest(p) for p in paths}
    catalog = json.loads(catalog_path.read_text())
    pilots = {p['properties']['id']: p for p in catalog['pilots']['features'] if p['properties']['id'] in h.PILOTS}
    records = h.enrich([r for r in json.loads((prior/'granules.json').read_text())
                        if r['product']=='ecostress_v2' and r['temporal_split']=='fit'], pilots)
    old_records = json.loads(old.read_text())['records']
    attempted = {(r['pilot_id'], geo.parse_identity(r['title'])['orbit']) for r in old_records}
    old_dates = {(r['pilot_id'], r['utc_date']) for r in old_records}
    registered = {(r['region_id'], r['utc_date']) for r in json.loads(registry_path.read_text())['dates']}
    rows = select(records, attempted, registered | old_dates)
    obstruction = geo.parse_obstruction_list(obstruction_path.read_bytes(), complete=True)
    if obstruction['status'] != 'Parsed' or obstruction['errors']:
        raise ValueError('Complete obstruction list required.')
    # Public CMR snapshots are read only. Combine snapshots by exact identity;
    # only a complete individual snapshot establishes its own latest revision.
    snapshots = defaultdict(list)
    directories = [root/'runs/option_b_retrain_20260909/acquisition_night/public_metadata',
                   root/'runs/earthdata_engineering', root/'runs/multisensor_20260910_v1/ecostress_expanded_v2/public_metadata']
    for directory in directories:
        for path in sorted(directory.rglob('geo_*.json')):
            if path.name.endswith('.source.json'):
                continue
            body = json.loads(path.read_text())
            items = body.get('items', [])
            if int(body.get('hits', len(items)+1)) > len(items):
                continue
            identities = set()
            for item in items:
                try:
                    identity = geo.parse_identity(item['umm']['GranuleUR'])
                    identities.add(tuple(identity[k] for k in ('orbit','scene','time')))
                except (ValueError, KeyError):
                    pass
            for identity in identities:
                snapshots[identity].append((path, items))
    rejected, candidates = [], []
    for row in rows:
        obstruction_result = geo.obstruction_status(obstruction, row['granule_title'])
        if obstruction_result['status']=='Unknown' or obstruction_result.get('obstructed') is True:
            rejected.append({'pilot_id':row['pilot_id'], 'granule_id':row['granule_concept_id'], 'reason':obstruction_result['status']})
            continue
        identity = geo.parse_identity(row['granule_title'])
        cached = []
        for path, items in snapshots.get(tuple(identity[k] for k in ('orbit','scene','time')), []):
            matched = geo.select_geo_candidate(items, row['granule_title'])
            if matched['status'] != 'Matched':
                continue
            item = {'cmr_snapshot_path':str(path), 'cmr_snapshot_sha256':eco.digest(path), 'matched':matched}
            sources[str(path)] = item['cmr_snapshot_sha256']
            name = matched['identity']['granule_name']
            dmrpp = root/'cache/ecostress_v2/geometry'/f'{name}.h5.dmrpp'
            if dmrpp.is_file():
                item['geolocation'] = geo.parse_geolocation_dmrpp(dmrpp.read_bytes(), name)
                sources[str(dmrpp)] = item['geolocation']['source_sha256']
            item['same_processing_build'] = matched['identity']['build']==identity['build']
            cached.append(item)
        listed = sorted({r['listed_geolocation_qa'] for r in obstruction_result.get('matching_variants', [])})
        # Conflicting snapshots cannot become affirmative proof.
        labels = {c.get('geolocation',{}).get('qa_label','Unknown') for c in cached}
        positive = bool(cached) and all(c['same_processing_build'] and c.get('geolocation',{}).get('accepted') for c in cached)
        candidates.append({**row, 'obstruction':obstruction_result, 'historical_listed_geolocation_labels':listed,
                           'cached_geo_snapshots':cached, 'cached_positive_geo':positive,
                           'cached_geo_labels':sorted(labels), 'source_screen_pass':False,
                           'thermal_qa_not_read':True, 'training_eligible':False})
    summary = []
    for pilot in h.PILOTS:
        subset = [r for r in candidates if r['pilot_id']==pilot]
        summary.append({'pilot_id':pilot, 'remaining_orbits':len(subset), 'remaining_dates':len({r['utc_date'] for r in subset}),
                        'dates_by_year':{year:len({r['utc_date'] for r in subset if r['utc_date'].startswith(year)}) for year in ('2021','2022')},
                        'orbits_by_month':dict(sorted(Counter(r['utc_date'][:7] for r in subset).items())),
                        'cached_positive_geo_orbits':sum(r['cached_positive_geo'] for r in subset),
                        'historical_list_positive_orbits':sum(bool(set(r['historical_listed_geolocation_labels']) & {'Good','Best'}) for r in subset),
                        'historical_list_negative_orbits':sum(bool(set(r['historical_listed_geolocation_labels']) & {'Poor','Suspect'}) for r in subset)})
    out.mkdir(parents=True)
    result = {'version':'night-extension-offline-inventory-20260910-v1', 'source_hashes':sources,
              'catalog_complete':True, 'public_requests':0, 'protected_requests':0, 'thermal_arrays_opened':0,
              'selection_frozen_for_acquisition':False, 'summary':summary, 'candidates':candidates,
              'metadata_rejected':rejected,
              'exclusions':'All prior200 attempted pilot-orbits, their whole pilot-dates, and every root registered pilot-date. Entire orbits excluded, including alternate tiles and native physical-cell duplicate identities. Reobserving a geographic cell on a different new date is allowed; it is not a duplicate thermal observation.',
              'limits':'Candidate counts are metadata opportunities only. Historical list Good/Best is not matched latest GEO QA. Cached GEO is latest only within its recorded complete CMR snapshot. Uncached, missing and negative flags do not qualify; all candidates still require acquisition authorization, current frozen matching metadata, thermal/nativeQA, permanent spatial exclusions and complete causal features.'}
    eco.save_json(out/'inventory.json', result)
    print(json.dumps({'output':str(out/'inventory.json'), 'sha256':eco.digest(out/'inventory.json'), 'summary':summary}), flush=True)


if __name__=='__main__':
    main()
