import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lst_pilot.option_b_acquire import (
    CampaignDownloader, CampaignStop, Ledger, MAX_BYTES, campaign_counts,
    frozen_split, next_candidate, ordered_acquisitions, spatial_coverage,
)


def minimal_campaign(tmp_path, monkeypatch):
    from lst_pilot import option_b_acquire as campaign
    record = {'pilot_id': 'greater_london', 'temporal_split': 'fit', 'utc_date': '2022-01-01',
              'time_start': '2022-01-01T00:00:00Z', 'granule_concept_id': 'G1-LPCLOUD', 'granule_title': 'fixture',
              'identity': {'acquisition_group': 'ECOSTRESS:orbit:1'}}
    queues = {f'{pilot}:{split}': [] for pilot in campaign.PILOTS for split in campaign.SPLITS}
    queues['greater_london:fit'] = [record]
    campaign.write(tmp_path / 'plan.json', {'source_hashes': {}, 'selection_rule': 'frozen fixture',
                                           'source_counts': {}, 'queues': queues})
    calls = []
    monkeypatch.setattr(campaign, 'CampaignDownloader', lambda ledger: object())
    def acquire(*_):
        calls.append(record['granule_concept_id'])
        return {**record, 'status': 'fixture_screened', 'source_screen_pass': False,
                'training_eligible': False, 'qualifying_independent_date': False}
    monkeypatch.setattr(campaign, 'acquire_one', acquire)
    return campaign, record, calls


@pytest.mark.parametrize('stamp,expected', [('2022-12-31T23:59:59Z', 'fit'),
                                          ('2023-06-30T23:59:59Z', 'development'),
                                          ('2023-07-01T00:00:00Z', 'calibration')])
def test_temporal_boundary_is_utc_and_reserved_years_refused(stamp, expected):
    assert frozen_split(stamp) == expected
    for forbidden in ('2024-01-01T00:00:00Z', '2025-08-01T12:00:00Z', '2021-01-01'):
        with pytest.raises(ValueError):
            frozen_split(forbidden)


def test_round_robin_calendar_strata_preserves_frozen_rank_and_london2022_priority():
    def item(month, rank, ordinal):
        return {'pilot_id': 'greater_london', 'temporal_split': 'fit', 'products_present': ['ecostress_v2'],
                'stratum': ['greater_london', 'ECOSTRESS', month, 0, 'NIGHT'], 'rank': rank,
                'rank_within_stratum': ordinal, 'acquisition_group': month + rank}
    rows = [item('2021-01', 'b', 1), item('2022-01', 'z', 2), item('2022-02', 'c', 1), item('2022-01', 'a', 1)]
    ordered = ordered_acquisitions(rows, 'greater_london', 'fit')
    assert [r['acquisition_group'] for r in ordered] == ['2022-01a', '2022-02c', '2021-01b', '2022-01z']
    assert ordered_acquisitions(list(reversed(rows)), 'greater_london', 'fit') == ordered


def test_whole_cell_buffer_and_fixed_raster_addresses(tmp_path):
    path = tmp_path / 'labels.tif'
    transform = from_origin(0, 10000, 100, 100)
    with rasterio.open(path, 'w', driver='GTiff', count=1, dtype='float32', width=200, height=100,
                       crs='EPSG:32614', transform=transform) as ds:
        ds.write(np.ones((100, 200), dtype='float32'), 1)
    pilot = {'id': 'p', 'epsg': 32614, 'extent_m': [0, 0, 20000, 10000], 'grid_shape': [100, 200]}
    blocks = [{'id': 'p_r00_c00', 'bounds_m': [0, 0, 10000, 10000], 'spatial_holdout': True, 'holdout_buffer_m': 1000},
              {'id': 'p_r00_c01', 'bounds_m': [10000, 0, 20000, 10000], 'spatial_holdout': False, 'holdout_buffer_m': 1000}]
    result = spatial_coverage(path, pilot, blocks)
    assert result['qa_pass_cells'] == 20000
    assert result['reserved_cells'] == 10000
    assert result['nonreserved_buffer_safe_cells'] == 8900
    assert result['blocks'][0]['nonreserved_buffer_safe_cells'] == 0
    assert result['blocks_at_least50_safe_cells'] == 1
    pilot['extent_m'] = [100, 0, 20100, 10000]
    with pytest.raises(ValueError, match='fixed pilot grid'):
        spatial_coverage(path, pilot, blocks)


def test_date_counts_deduplicate_orbits_and_do_not_count_failed_sources():
    def row(date, qualifies=True):
        return {'pilot_id': 'greater_london', 'temporal_split': 'fit', 'utc_date': date,
                'qualifying_independent_date': qualifies}
    result = campaign_counts([row('2021-01-01'), row('2021-01-01'), row('2021-01-02', False)])
    assert result['greater_london:fit'] == ['2021-01-01']


def test_next_candidate_skips_already_qualified_date_without_reordering():
    records = [{'pilot_id': 'greater_london', 'temporal_split': 'fit', 'utc_date': '2022-01-01',
                'qualifying_independent_date': True}]
    keys = [f'{pilot}:{split}' for pilot in ('greater_london', 'sioux_falls') for split in ('fit', 'development', 'calibration')]
    plan = {'queues': {key: [] for key in keys}}
    plan['queues']['greater_london:fit'] = [{'utc_date': '2022-01-01'}, {'utc_date': '2022-01-02'}]
    state = {'rotation': 0, 'queue_cursors': {}, 'same_qualifying_date_skips': 0}
    assert next_candidate(plan, state, records)['utc_date'] == '2022-01-02'
    assert state['same_qualifying_date_skips'] == 1


def test_budget_reservation_is_durable_and_never_reset_by_new_ledger(tmp_path):
    state = {'network_bytes_charged': MAX_BYTES - 3, 'public_http_requests': 0, 'protected_http_requests': 0}
    ledger = Ledger(tmp_path, state)
    ledger.reserve(3)
    recovered = Ledger(tmp_path, json.loads((tmp_path / 'state.json').read_text()))
    with pytest.raises(CampaignStop):
        recovered.reserve(1)


def test_cached_protected_bytes_are_verified_without_recharging(tmp_path):
    state = {'network_bytes_charged': 0, 'public_http_requests': 0, 'protected_http_requests': 0}
    ledger = Ledger(tmp_path, state)
    downloader = CampaignDownloader(ledger, token='fake-unused-secret')
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def iter_content(self, *_): yield b'II*\x00abcd'
    downloader.session.get = lambda *args, **kwargs: Response()
    url = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/ECO_L2T_LSTE.002/scene/scene_LST.tif'
    path = tmp_path / 'file.tif'
    first = downloader.download(url, path)
    assert state['network_bytes_charged'] == 8 and state['protected_http_requests'] == 1
    assert downloader.download(url, path) == first
    assert state['network_bytes_charged'] == 8 and state['protected_http_requests'] == 1
    assert 'fake-unused-secret' not in (tmp_path / 'state.json').read_text()


def test_interrupted_manifest_commit_recovers_inflight_at_candidate_cap(tmp_path, monkeypatch):
    campaign, record, calls = minimal_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, 'MAX_CANDIDATES', 1)
    original_write = campaign.write
    def interrupted(path, value):
        if path.name == 'manifest.json':
            raise OSError('simulated interruption before manifest commit')
        original_write(path, value)
    monkeypatch.setattr(campaign, 'write', interrupted)
    with pytest.raises(OSError):
        campaign.run(tmp_path, tmp_path, 1)
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['candidate_attempts'] == 1
    assert state['inflight']['granule_concept_id'] == record['granule_concept_id']
    monkeypatch.setattr(campaign, 'write', original_write)
    campaign.run(tmp_path, tmp_path, 1)
    state = json.loads((tmp_path / 'state.json').read_text())
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert calls == ['G1-LPCLOUD', 'G1-LPCLOUD']
    assert state['candidate_attempts'] == 1 and state['inflight'] is None
    assert len(manifest['records']) == 1
    assert manifest['status'] == 'campaign_candidate_cap_reached'


def test_committed_inflight_recovery_does_not_reacquire_or_charge_attempt(tmp_path, monkeypatch):
    campaign, record, calls = minimal_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, 'MAX_CANDIDATES', 1)
    campaign.run(tmp_path, tmp_path, 1)
    state = json.loads((tmp_path / 'state.json').read_text())
    state['inflight'] = record  # Simulated interruption after manifest commit, before clearing state.
    campaign.write(tmp_path / 'state.json', state)
    campaign.run(tmp_path, tmp_path, 1)
    assert calls == ['G1-LPCLOUD']
    assert json.loads((tmp_path / 'state.json').read_text())['inflight'] is None


def test_changed_frozen_plan_is_rejected_before_downloader_creation(tmp_path, monkeypatch):
    campaign, _, _ = minimal_campaign(tmp_path, monkeypatch)
    campaign.run(tmp_path, tmp_path, 1)
    plan = json.loads((tmp_path / 'plan.json').read_text())
    plan['selection_rule'] = 'changed after the first batch'
    campaign.write(tmp_path / 'plan.json', plan)
    def forbidden(*_):
        raise AssertionError('No credential-bearing downloader should be constructed.')
    monkeypatch.setattr(campaign, 'CampaignDownloader', forbidden)
    with pytest.raises(ValueError, match='Frozen plan changed'):
        campaign.run(tmp_path, tmp_path, 1)
