from lst_pilot.coarse_context_plan import select_past


def row(start,end,tag='2022001000000'):
    return {'granule_start_utc':start,'granule_end_utc':end,'production_tag':tag,'cmr_revision':1}


def test_latest_completed_native_interval_only():
    old=row('2022-01-01T23:54Z','2022-01-02T00:00Z')
    latest=row('2022-01-02T10:54Z','2022-01-02T11:00Z')
    future=row('2022-01-02T11:00Z','2022-01-02T11:06Z')
    assert select_past([old,future,latest],'2022-01-02T11:03Z') is latest
    assert select_past([old,latest],'2022-01-03T12:00Z') is None


def test_revision_order_cannot_beat_observation_time():
    newer_production=row('2022-01-02T10:00Z','2022-01-02T10:06Z','2023001000000')
    newer_observation=row('2022-01-02T11:00Z','2022-01-02T11:06Z','2022001000000')
    assert select_past([newer_production,newer_observation],'2022-01-02T12:00Z') is newer_observation
