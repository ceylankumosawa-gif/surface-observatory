"""Prospective native-support weighting and leakage-safe split definitions."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = ('2021-01', '2021-04', '2021-07', '2021-10')
REFERENCE_ONLY = frozenset({'cabauw'})


def balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal pilot/phase, date, acquisition; represented area within each one.

    Physical source/cell IDs must be unique. Distinct native observations may
    have overlapping support: this is normalized observation-area weighting,
    not a union-area integral. Overlap is reported separately; neither these
    weights nor adjacent observations create independent dates.
    """
    if frame.empty:
        return np.empty(0, dtype=float)
    keys = ['region_id', 'phase', 'utc_date', 'acquisition_id']
    if frame[keys].isna().any().any():
        raise ValueError('Missing weighting identity')
    area = frame.native_footprint_area_m2.to_numpy(float)
    if not (np.isfinite(area) & (area > 0)).all():
        raise ValueError('Nonpositive or missing represented area')
    group_count = len(frame[['region_id', 'phase']].drop_duplicates())
    dates = frame.groupby(keys[:2], observed=True).utc_date.transform('nunique').to_numpy()
    acquisitions = frame.groupby(keys[:3], observed=True).acquisition_id.transform('nunique').to_numpy()
    total_area = frame.groupby(keys, observed=True).native_footprint_area_m2.transform('sum').to_numpy()
    weight = area / total_area / acquisitions / dates / group_count
    if not np.isfinite(weight).all() or abs(weight.sum() - 1.) > 1e-12:
        raise ValueError('Weight normalization failed')
    return weight


def split_definitions(frame: pd.DataFrame):
    """Return prospective masks before feature/QA missingness is applied.

    A granule shared by pilots or spanning a held month is excluded wholesale
    from fitting. Cabauw always stays outside every fitting population.
    """
    required = ['region_id', 'granule_id', 'granule_start_utc', 'granule_end_utc']
    if frame[required].isna().any().any():
        raise ValueError('Missing split identity')
    start = pd.to_datetime(frame.granule_start_utc, utc=True)
    end = pd.to_datetime(frame.granule_end_utc, utc=True)
    if (end < start).any() or ((end-start).dt.total_seconds() > 600).any():
        raise ValueError('Unsupported native observation interval')
    membership = frame.assign(_start=start, _end=end).groupby('granule_id')[['_start', '_end']].nunique()
    if not membership.eq(1).all().all():
        raise ValueError('Granule interval disagrees between rows')
    fit_population = ~frame.region_id.isin(REFERENCE_ONLY).to_numpy()
    for region in sorted(set(frame.region_id) - REFERENCE_ONLY):
        held = frame.region_id.eq(region).to_numpy()
        excluded = frame.loc[held, 'granule_id'].unique()
        training = fit_population & ~frame.granule_id.isin(excluded).to_numpy()
        yield 'pilot', region, training, held
    for region in sorted(set(frame.region_id) & REFERENCE_ONLY):
        held = frame.region_id.eq(region).to_numpy()
        excluded = frame.loc[held, 'granule_id'].unique()
        training = fit_population & ~frame.granule_id.isin(excluded).to_numpy()
        yield 'reference', region, training, held
    start_month = start.dt.strftime('%Y-%m')
    end_month = end.dt.strftime('%Y-%m')
    for month in MONTHS:
        # Midpoint defines the evaluation month. Either endpoint excludes fit.
        midpoint_month = (start+(end-start)/2).dt.strftime('%Y-%m')
        held = midpoint_month.eq(month).to_numpy() & fit_population
        touches = start_month.eq(month) | end_month.eq(month)
        excluded = frame.loc[touches, 'granule_id'].unique()
        training = fit_population & ~frame.granule_id.isin(excluded).to_numpy()
        yield 'month', month, training, held
    yield 'full', 'all', fit_population, np.zeros(len(frame), dtype=bool)


def design_checks():
    """Hand-worked unequal-area/date case and leakage boundary regressions."""
    f = pd.DataFrame([
        ('a', 'day', 'd1', 'g1', 1.),
        ('a', 'day', 'd1', 'g1', 3.),
        ('a', 'day', 'd1', 'g2', 1.),
        ('a', 'day', 'd2', 'g3', 2.),
        ('a', 'night', 'd1', 'g4', 1.),
        ('b', 'day', 'd1', 'g5', 1.),
    ], columns=['region_id','phase','utc_date','acquisition_id','native_footprint_area_m2'])
    np.testing.assert_allclose(balanced_weights(f), [1/48, 1/16, 1/12, 1/6, 1/3, 1/3], rtol=0, atol=1e-15)
    s = pd.DataFrame([
        ('a', 'shared', '2021-01-01T00:00Z', '2021-01-01T00:06Z'),
        ('b', 'shared', '2021-01-01T00:00Z', '2021-01-01T00:06Z'),
        ('b', 'cross', '2021-03-31T23:58Z', '2021-04-01T00:04Z'),
        ('b', 'july', '2021-07-01T00:00Z', '2021-07-01T00:06Z'),
        ('cabauw', 'reference', '2021-10-01T00:00Z', '2021-10-01T00:06Z'),
    ], columns=['region_id','granule_id','granule_start_utc','granule_end_utc'])
    cases = {(mode,key):(train,held) for mode,key,train,held in split_definitions(s)}
    np.testing.assert_array_equal(cases['pilot','a'][0], [False, False, True, True, False])
    np.testing.assert_array_equal(cases['month','2021-04'][0], [True, True, False, True, False])
    np.testing.assert_array_equal(cases['month','2021-04'][1], [False, False, True, False, False])
    for train, held in cases.values():
        assert not train[-1] and not (train & held).any()
        assert set(s.loc[train,'granule_id']).isdisjoint(s.loc[held,'granule_id'])
    # A source crossing the reference and a training pilot must stay wholly out
    # of the reference fit, even though its non-reference cells are elsewhere.
    shared_reference = s.copy()
    shared_reference.loc[4, ['granule_id','granule_start_utc','granule_end_utc']] = s.loc[0, ['granule_id','granule_start_utc','granule_end_utc']].tolist()
    reference = {(mode,key):(train,held) for mode,key,train,held in split_definitions(shared_reference)}
    np.testing.assert_array_equal(reference['reference','cabauw'][0], [False,False,True,True,False])
    return {'unequal_area_date_weights':'passed','shared_granule_exclusion':'passed',
            'month_crossing_exclusion':'passed','reference_only_exclusion':'passed',
            'reference_shared_granule_exclusion':'passed'}


if __name__ == '__main__':
    import json
    print(json.dumps(design_checks(), sort_keys=True))
