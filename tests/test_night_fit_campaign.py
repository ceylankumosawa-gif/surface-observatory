import json

import pytest

from lst_pilot import night_fit_campaign as n


def row(date, rank=1):
    return {'pilot_id':'greater_london','temporal_split':'fit','actual_phase':'night',
            'utc_date':date,'solar_elevation_max_centre_corners':-7,'local_solar_3hour_bin':0,
            'selection':{'rank_within_stratum':rank,'rank':date}}


def test_alternates_year_and_season_before_second_candidate():
    rows=[row('2021-01-02'),row('2021-01-01'),row('2022-01-01'),row('2021-04-01'),row('2022-04-01')]
    queue=n.balanced_queues(rows)['greater_london:fit:night']
    assert [r['utc_date'] for r in queue]==['2021-01-01','2022-01-01','2021-04-01','2022-04-01','2021-01-02']
    assert all(not r['fresh_2023'] for r in queue)


def test_four_ranks_per_calendar_hour_and_no_reserved_labels():
    assert not n.balanced_queues([row('2021-01-01',5)])['greater_london:fit:night']
    with pytest.raises(ValueError,match='2021/22'):
        n.balanced_queues([row('2023-01-01')])


def test_smaller_hard_limit_and_durable_reservation(tmp_path):
    state={'network_bytes_charged':n.MAX_BYTES-10}
    ledger=n.Ledger(tmp_path,state)
    with pytest.raises(n.base.CampaignStop):ledger.reserve(11)
    ledger.reserve(10)
    assert json.loads((tmp_path/'state.json').read_text())['network_bytes_charged']==1024**3


def test_changed_plan_fails_before_authentication(tmp_path,monkeypatch):
    plan={'source_hashes':{},'limits':{'candidate_attempts':100,'additional_network_bytes':1024**3}}
    n.eco.save_json(tmp_path/'plan.json',plan)
    n.eco.save_json(tmp_path/'state.json',{'plan_sha256':'wrong'})
    n.eco.save_json(tmp_path/'manifest.json',{'plan_sha256':'wrong'})
    monkeypatch.setattr(n,'Downloader',lambda *a:pytest.fail('Authentication must not run'))
    with pytest.raises(ValueError,match='plan changed'):
        n.run(tmp_path,tmp_path,1)


def test_source_drift_fails_before_authentication(tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_text('changed')
    n.eco.save_json(tmp_path/'plan.json',{'source_hashes':{str(source):'wrong'}})
    monkeypatch.setattr(n,'Downloader',lambda *a:pytest.fail('Authentication must not run'))
    with pytest.raises(ValueError,match='source/code changed'):
        n.run(tmp_path,tmp_path,1)
