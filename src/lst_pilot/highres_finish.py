"""Complete the single London2023Q2 metadata shard after its page ceiling."""
import argparse
import json
from pathlib import Path

from . import highres_inventory as h, night_inventory as n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('New immutable output required.')
    old = json.loads((args.prior/'inventory.json').read_text())
    missing = [q for q in old['queries'] if not q['complete']]
    if len(missing) != 1 or missing[0]['pilot_id'] != 'greater_london' or missing[0]['params']['temporal'] != '2023-04-01T00:00:00Z,2023-06-30T23:59:59Z':
        raise ValueError('This bounded finish covers only the audited incomplete shard.')
    pilots = {a['properties']['id']: a for a in json.loads((args.root/'web/catalog.json').read_text())['pilots']['features']}
    client = h.TimedClient(20, 64, 600)
    rows, query = n.inventory_query(client, pilots['greater_london'], 'ecostress_v2', 'DAY', '2023-04-01', '2023-06-30', 100, 20)
    if not query['complete']:
        raise ValueError('The bounded finish is still incomplete; no merged freeze produced.')
    prior_records = json.loads((args.prior/'granules.json').read_text())
    records = [r for r in prior_records if not (r['pilot_id']=='greater_london' and r['product']=='ecostress_v2'
                  and r['day_night_flag']=='DAY' and '2023-04-01' <= r['utc_date'] <= '2023-06-30')]
    records = h.enrich(records + rows, pilots)
    args.output.mkdir(parents=True)
    n.write_json(args.output/'granules.json', records)
    n.write_json(args.output/'calendar_candidates.json', {'inventory_complete': True, 'frozen_for_label_acquisition': False,
        'acquisitions': h.candidate_queue(records)})
    (args.output/'spatial_blocks.json').write_bytes((args.prior/'spatial_blocks.json').read_bytes())
    summary = {**old, 'version': 'highres-completed-metadata-20260910-v1', 'complete': True,
        'queries': [q if q['complete'] else query for q in old['queries']],
        'continuation_public_requests_total': old['public_http_requests'] + client.requests,
        'continuation_metadata_bytes_total': old['metadata_bytes'] + client.bytes,
        'this_finish_requests': client.requests, 'this_finish_bytes': client.bytes,
        'this_finish_http_audit': client.audit,
        'finish_source_hashes': {str(p): n.sha256_file(p) for p in (args.prior/'inventory.json', args.prior/'granules.json', Path(__file__), Path(h.__file__), Path(n.__file__))}}
    for count in summary['counts']:
        subset = [r for r in records if (r['pilot_id'],r['product'],r['actual_phase']) == (count['pilot_id'],count['product'],count['phase'])]
        count.update(granules=len(subset), utc_dates=len({r['utc_date'] for r in subset}),
                     acquisitions=len({r['identity']['acquisition_group'] for r in subset}),
                     dates_by_split={s:len({r['utc_date'] for r in subset if r['temporal_split']==s}) for s in ('fit','development','calibration')})
    n.write_json(args.output/'inventory.json', summary)
    print(json.dumps({'complete': True, 'counts': summary['counts'], 'finish_requests': client.requests,
                      'finish_bytes': client.bytes}), flush=True)


if __name__ == '__main__':
    main()
