import numpy as np
import pandas as pd
from shapely.geometry import box
from pyproj import Transformer
from lst_pilot.coarse_context import annotate,context_margin,point_context


def sample():
    p=box(512500,54500,513500,55500)
    return {'native_footprint_wkb':p.wkb,'native_footprint_area_m2':p.area,'view_zenith_deg':0.,'product':'MOD21','native_qa_valid':True,
        'native_cell_id':'nativeA','acquisition_id':'acqA','granule_start_utc':'2022-01-02T11:00Z','granule_end_utc':'2022-01-02T11:06Z',
        'lst_c':20.,'lst_error_k':1.,'source_sha256':'sourceA'}


def test_context_margin_increases_with_real_edges_and_angle():
    p=box(0,0,1000,1000)
    assert context_margin(p,0,'MOD21',50)==650
    assert context_margin(p,30,'MOD21',50)>650
    assert context_margin(p,0,'VNP21')==875
    assert context_margin(p,0,'MOD21',None) is None
    assert context_margin(p,31,'VNP21') is None


def test_context_and_coarse_label_eligibility_are_separate():
    area={'id':'greater_london','extent_m':[500000,50000,530000,80000]}
    s={'metadata':{'geolocation_rms_error_m':'50'}};frame=annotate(pd.DataFrame([sample()]),s,area)
    assert frame.context_eligible.iloc[0] and frame.context_fit_eligible.iloc[0]
    assert not frame.coarse_label_training_eligible.iloc[0]
    held=sample();held['native_footprint_wkb']=box(504500,54500,505500,55500).wkb
    f=annotate(pd.DataFrame([held]),s,area)
    assert f.context_eligible.iloc[0] and not f.context_fit_eligible.iloc[0]


def test_point_join_rejects_future_and_uses_native_membership_not_padding():
    area={'id':'greater_london','extent_m':[500000,50000,530000,80000]}
    frame=annotate(pd.DataFrame([sample()]),{'metadata':{'geolocation_rms_error_m':'50'}},area)
    lon,lat=Transformer.from_crs(32631,4326,always_xy=True).transform(513000,55000)
    assert not point_context(frame,'2022-01-02T11:03Z',lon,lat,32631)['context_available']
    result=point_context(frame,'2022-01-02T12:00Z',lon,lat,32631,True)
    assert result['context_available'] and result['age_min_hours']==.9 and result['age_max_hours']==1
    assert result['context_native_cell_id']=='nativeA'
    assert not point_context(frame,'2022-01-03T11:03Z',lon,lat,32631)['context_available']
    lon,lat=Transformer.from_crs(32631,4326,always_xy=True).transform(513600,55000)
    assert not point_context(frame,'2022-01-02T12:00Z',lon,lat,32631)['context_available']


def test_empty_geometry_support_stays_joinable_and_has_zero_eligibility():
    from lst_pilot.coarse_join import CONTEXT_COLUMNS
    frame=annotate(pd.DataFrame(),{'metadata':{}},{'id':'greater_london','extent_m':[0,0,30000,30000]})
    assert all(k in frame for k in CONTEXT_COLUMNS)
    assert frame.context_eligible.dtype==bool
    assert not len(frame.loc[frame.context_eligible,'native_footprint_wkb'])
