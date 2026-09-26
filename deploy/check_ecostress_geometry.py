"""Bounded geometry metadata check for an existing engineering-only plan."""
import argparse
import json
from pathlib import Path
import resource

from lst_pilot.ecostress import NasaDownloader, AcquisitionError, save_json, digest
from lst_pilot.ecostress_geo import (discover_geo, parse_geolocation_dmrpp,
                                    fetch_obstruction_list, obstruction_status,
                                    GeoMetadataError)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    plan = json.loads(args.plan.read_text())
    if not plan.get('engineering_only') or not 1 <= len(plan['records']) <= 16:
        parser.error('Expected a frozen engineering plan with at most sixteen records.')
    if any(int(r['time_start'][:4]) not in (2021, 2022, 2023) for r in plan['records']):
        parser.error('Only fitting/development years are allowed.')
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.root / 'cache/ecostress_v2/geometry'
    client = NasaDownloader(max_bytes=32 * 1024**2, max_requests=96)
    audit = {'engineering_only': True, 'training_eligible': False,
             'plan_sha256': digest(args.plan), 'results': []}
    try:
        index = fetch_obstruction_list(cache)
        save_json(args.output / 'obstruction_index.json', index)
    except GeoMetadataError:
        index = {'status': 'Unknown', 'records': [], 'complete_download': False}
    for record in plan['records']:
        result = {'title': record['granule_title'], 'pilot_id': record['pilot_id'],
                  'training_eligible': False}
        try:
            match = record.get('geo_discovery') or discover_geo(record['granule_title'], cache)
            if match['status'] == 'Matched' and any(match['identity'][key] != record['identity'][key]
                                                     for key in ('orbit', 'scene', 'time')):
                raise GeoMetadataError('Frozen GEO discovery differs from the planned acquisition.')
            result['discovery'] = match
            if match['status'] == 'Matched':
                geo_name = match['identity']['granule_name']
                path = cache / (geo_name + '.h5.dmrpp')
                result['download'] = client.download(match['dmrpp_url'], path, kind='geolocation_metadata')
                result['geolocation'] = parse_geolocation_dmrpp(path.read_bytes(), geo_name)
                result['same_processing_build'] = match['identity']['build'] == record['granule_title'].split('_')[-2]
            result['obstruction'] = obstruction_status(index, record['granule_title'])
        except (AcquisitionError, GeoMetadataError) as error:
            result['error'] = str(error)  # These types contain fixed, secret-free messages only.
        except Exception:
            result['error'] = 'Metadata check failed; external details are suppressed.'
        audit['results'].append(result)
        audit.update(protected_bytes=client.bytes, protected_requests=client.requests)
        save_json(args.output / 'audit.json', audit)
        print(json.dumps({'title': result['title'],
                          'geolocation': result.get('geolocation', {}).get('qa_label', 'Unknown'),
                          'obstruction': result.get('obstruction', {}).get('status', 'Unknown'),
                          'error': result.get('error')}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        raise SystemExit('Geometry metadata check stopped; credentials and external details are suppressed.')
