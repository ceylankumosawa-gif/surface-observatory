import pandas as pd
from pyproj import Transformer
from shapely.geometry import box
import pytest
from lst_pilot.coarse_join import join_frame


def inputs():
    lon,lat=Transformer.from_crs(32631,4326,always_xy=True).transform(500000,5500000)
    samples=pd.DataFrame({'sample_id':['future','past','old'],'region_id':'london','datetime_utc':['2022-01-01T11:03Z','2022-01-01T12:00Z','2022-01-02T11:03Z'],
        'latitude':lat,'longitude':lon,'epsg':32631})
    p=box(499500,5499500,500500,5500500)
    native=pd.DataFrame([{'native_cell_id':'native','region_id':'london','acquisition_id':'source','granule_start_utc':'2022-01-01T11:00Z',
        'granule_end_utc':'2022-01-01T11:06Z','native_footprint_wkb':p.wkb,'native_footprint_area_m2':p.area,'footprint_epsg':32631,
        'view_zenith_deg':0.,'product':'MOD21','lst_c':20.,'lst_error_k':1.,'source_sha256':'hash','context_support_wkb':p.buffer(650).wkb,
        'context_support_margin_m':650.,'context_eligible':True,'context_fit_eligible':True}])
    return samples,native


def test_join_preserves_sample_identity_and_explicit_missingness():
    s,n=inputs();r=join_frame(s,n,True)
    assert list(r.sample_id)==list(s.sample_id)
    assert list(r.coarse_context_eligible)==[False,True,False]
    assert r.coarse_lst_c.iloc[1]==20 and r.coarse_age_hours.iloc[1]==1
    assert r.coarse_age_min_hours.iloc[1]==.9
    assert r.coarse_native_id.iloc[1]=='native' and r.coarse_source_sha256.iloc[1]=='hash'


def test_fit_context_cannot_use_holdout_only_context():
    s,n=inputs();n['context_fit_eligible']=False
    assert join_frame(s,n,True).coarse_context_eligible.sum()==0
    assert join_frame(s,n,False).coarse_context_eligible.sum()==1


def test_duplicates_or_ambiguous_times_fail():
    s,n=inputs();s.loc[0,'sample_id']='past'
    with pytest.raises(ValueError):join_frame(s,n)
    s,n=inputs();s.loc[0,'datetime_utc']='2022-01-01T11:03'
    with pytest.raises(ValueError):join_frame(s,n)
