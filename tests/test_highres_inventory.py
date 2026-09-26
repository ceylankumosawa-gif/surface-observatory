from datetime import datetime, timezone

import pytest

from lst_pilot import highres_inventory as h


def row(group='1', stamp='2021-07-01T12:00:00+00:00', phase='day', split='fit'):
    return {'pilot_id': 'greater_london', 'product': 'ecostress_v2',
            'granule_concept_id': 'G'+group, 'granule_title': 'name'+group,
            'identity': {'acquisition_group': group}, 'time_start': stamp,
            'utc_date': stamp[:10], 'temporal_split': split, 'actual_phase': phase,
            'local_solar_3hour_bin': 4, 'metadata_overlap_m2': 10000,
            'footprint_qa': {'status': 'metadata_consistent_pixels_unverified'}}


def test_queue_groups_tiles_and_is_order_independent():
    rows = [row('1'), row('2'), {**row('1'), 'granule_concept_id': 'G3'}]
    queue = h.candidate_queue(rows)
    assert queue == h.candidate_queue(list(reversed(rows)))
    assert len(queue) == 2
    assert sorted(len(r['granule_concept_ids']) for r in queue) == [1, 2]
    assert [r['rank_within_stratum'] for r in queue] == [1, 2]


def test_queue_rejects_cross_split_and_twilight():
    rows = [row('1'), row('1', split='development'), row('2', phase='twilight_or_mixed')]
    assert h.candidate_queue(rows) == []


def test_queue_does_not_look_at_thermal_magnitude():
    rows = [row('1'), row('2')]
    assert h.candidate_queue(rows) == h.candidate_queue([{**r, 'lst_c': 1000} for r in rows])


def test_solar_enrichment_and_reserved_year_rejection():
    pilot = {'properties': {'id': 'greater_london', 'center': [-.1, 51.5],
                             'epsg': 32630, 'extent_m': [695000, 5710000, 696000, 5711000]}}
    rows = [row('1'), row('2', '2021-07-01T00:00:00+00:00')]
    for r in rows:
        r['footprint_qa']['cmr_geometry'] = None
    result = h.enrich(rows, {'greater_london': pilot})
    assert {r['actual_phase'] for r in result} == {'day', 'night'}
    assert all(r['training_eligible'] is False for r in result)
    with pytest.raises(ValueError, match='Reserved'):
        h.enrich([row('3', '2025-07-01T00:00:00+00:00')], {'greater_london': pilot})


def test_deadline_stops_without_network(monkeypatch):
    client = h.TimedClient(seconds=-1)
    with pytest.raises(h.inventory.PreflightError, match='wall-time'):
        client.get('granules.json', {})
    assert client.requests == 0
