import json

import pytest

from lst_pilot import ecostress_campaign as c


def row(orbit=1, day='2021-05-01', phase='day', build='0712', overlap=10000):
    token=day.replace('-','')+'T120000'
    stamp=day+'T12:00:00+00:00'
    split='fit' if day[:4] in ('2021','2022') else 'development' if day[5:7]<='06' else 'calibration'
    return {'pilot_id':'greater_london','product':'ecostress_v2','granule_concept_id':f'G{orbit}{build}-LPCLOUD',
        'granule_title':f'ECOv002_L2T_LSTE_{orbit:05d}_001_30UXC_{token}_{build}_01',
        'identity':{'acquisition_group':f'ECOSTRESS:orbit:{orbit:05d}'},
        'time_start':stamp,'utc_date':day,'temporal_split':split,'actual_phase':phase,
        'solar_elevation_min_centre_corners':20 if phase=='day' else -20,
        'solar_elevation_max_centre_corners':21 if phase=='day' else -19,
        'metadata_overlap_m2':overlap,'local_solar_3hour_bin':4,
        'footprint_qa':{'status':'metadata_consistent_pixels_unverified'}}


def test_fresh_registry_and_initial_attempts_excluded():
    a,b,d=row(1,'2023-05-01'),row(2,'2023-05-02'),row(3)
    queues=c.make_queues([a,b,d],{'dates':[{'region_id':'greater_london','utc_date':'2023-05-01'}]},
                         {'records':[{'granule_id':d['granule_concept_id']}]})
    assert [r['granule_concept_id'] for q in queues.values() for r in q]==[b['granule_concept_id']]


def test_twilight_fit_night_and_reserved_year_excluded():
    a=row();a['solar_elevation_min_centre_corners']=9
    assert c.make_queues([a,row(2,phase='night')],{'dates':[]},{'records':[]})=={}
    with pytest.raises(ValueError,match='Reserved'):
        c.make_queues([row(3,'2024-05-01')],{'dates':[]},{'records':[]})


def test_four_acquisition_ranks_per_stratum_and_stable_order():
    rows=[row(i,f'2021-05-{i:02d}') for i in range(1,8)]
    q=c.make_queues(rows,{'dates':[]},{'records':[]})
    assert sum(map(len,q.values()))==4
    assert q==c.make_queues(rows[::-1],{'dates':[]},{'records':[]})


def test_initial_orbit_alternate_tile_and_qualifying_date_are_excluded():
    original=row(1)
    alternate={**original,'granule_concept_id':'Galt-LPCLOUD',
               'granule_title':original['granule_title'].replace('30UXC','30UXD')}
    same_day=row(2)
    other_day=row(3,'2021-05-02')
    initial={'records':[{'granule_id':original['granule_concept_id'],'pilot_id':'greater_london',
                        'title':original['granule_title'],'time_start':original['time_start'],'source_screen_pass':True}]}
    queues=c.make_queues([alternate,same_day,other_day],{'dates':[]},initial)
    assert [r['granule_concept_id'] for q in queues.values() for r in q]==[other_day['granule_concept_id']]


def test_latest_processing_before_greatest_overlap():
    rows=[row(1,build='0700',overlap=50000),row(1,build='0712',overlap=10000)]
    q=c.make_queues(rows,{'dates':[]},{'records':[]})
    assert next(iter(q.values()))[0]['granule_title'].endswith('_0712_01')


def test_date_target_counts_once_and_cursor_skips():
    a,b=row(),row(2,'2021-05-02')
    key=c.queue_key(a)
    plan={'queue_order':[key],'queues':{key:[a,b]},'targets':{key:2}}
    state={'rotation':0,'cursors':{},'seconds_by_pilot_phase':{}}
    records=[{**a,'qualifying_independent_date':True},{**a,'qualifying_independent_date':True}]
    assert c.next_candidate(plan,state,records)==b
    assert state['cursors'][key]==2


def test_soft_phase_deadline_prevents_another_candidate():
    a=row();key=c.queue_key(a)
    plan={'queue_order':[key],'queues':{key:[a]},'targets':{key:12}}
    state={'rotation':0,'cursors':{},'seconds_by_pilot_phase':{'greater_london:day':1200}}
    assert c.next_candidate(plan,state,[]) is None


def test_changed_plan_stops_before_credential_client(tmp_path,monkeypatch):
    (tmp_path/'plan.json').write_text(json.dumps({'source_hashes':{}}))
    (tmp_path/'state.json').write_text(json.dumps({'plan_sha256':'wrong'}))
    (tmp_path/'manifest.json').write_text(json.dumps({'plan_sha256':'wrong'}))
    monkeypatch.setattr(c,'Downloader',lambda *_:pytest.fail('Credential client must not initialize'))
    with pytest.raises(ValueError,match='plan changed'):
        c.run(tmp_path,tmp_path,1)


def test_result_commit_recovers_inflight_at_cap(tmp_path,monkeypatch):
    plan={'source_hashes':{},'source_counts':{}}
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan));sha=c.eco.digest(p)
    state={'plan_sha256':sha,'network_bytes_charged':0,'protected_http_requests':0,'public_http_requests':0,
           'candidate_attempts':200,'rotation':0,'cursors':{},'inflight':{'granule_concept_id':'G1'},'seconds_by_pilot_phase':{}}
    (tmp_path/'state.json').write_text(json.dumps(state))
    (tmp_path/'manifest.json').write_text(json.dumps({'plan_sha256':sha,'records':[{'granule_id':'G1'}]}))
    monkeypatch.setattr(c,'Downloader',lambda *_:None)
    c.run(tmp_path,tmp_path,1)
    assert json.loads((tmp_path/'state.json').read_text())['inflight'] is None
    assert len(json.loads((tmp_path/'manifest.json').read_text())['records'])==1
