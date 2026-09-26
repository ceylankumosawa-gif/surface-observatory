import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from lst_pilot.coarse_lst import decode, age_interval, native_shape, footprint, partition, validate_scale, attr, validate_interval


def raw():
    return {'LST':np.array([15000]*8,dtype='uint16'),'LST_err':np.array([25]*8,dtype='uint8'),
            'QC':np.array([3<<14]*8,dtype='uint16'),'View_angle':np.zeros(8,dtype='uint8'),'oceanpix':np.zeros(8,dtype='uint8')}


def test_native_scaling_and_independent_qa_gates():
    r=raw();r['LST'][1]=0;r['LST_err'][2]=0;r['QC'][3]|=1<<4;r['QC'][4]|=2<<4;r['View_angle'][5]=61;r['oceanpix'][6]=2;r['QC'][7]=1<<14
    d=decode(r)
    np.testing.assert_array_equal(d['native_qa_valid'],[1,0,0,0,0,0,0,0])
    assert d['lst_c'][0]==pytest.approx(26.85)
    assert d['lst_error_k'][0]==1
    assert np.isnan(d['lst_c'][1]) and np.isnan(d['lst_error_k'][2])
    assert d['qc_cloud'][4]==2 and not d['qa_cloud_clear'][4]
    assert d['qc_lst_error'][7]==1 and not d['qa_error_class_ok'][7]


def test_qc_high_bits_are_error_class_not_low_emissivity_bits():
    r=raw();r['QC'][0]=2<<14;r['QC'][1]=3<<12;r['LST_err'][2]=38
    d=decode(r)
    assert d['native_qa_valid'][0]
    assert not d['native_qa_valid'][1]
    assert not d['qa_error_ok'][2]


def test_whole_observation_interval_is_causal():
    r={'granule_start_utc':'2022-01-15T11:00:00Z','granule_end_utc':'2022-01-15T11:06:00Z'}
    assert not age_interval(r,'2022-01-15T11:03:00Z')['available']
    a=age_interval(r,'2022-01-15T12:00:00Z')
    assert a['available'] and a['age_min_hours']==.9 and a['age_max_hours']==1
    assert age_interval(r,'2022-01-15T11:06:00Z')['available']
    assert not age_interval(r,'2022-01-16T11:03:00Z')['available']
    with pytest.raises(ValueError):age_interval(r,'2022-01-15T12:00:00')


def test_sparse_modis_geolocation_is_rejected():
    with pytest.raises(ValueError,match='interpolation'):
        native_shape(np.empty((20,15)),np.empty((4,3)),np.empty((4,3)))
    native_shape(np.empty((20,15)),np.empty((20,15)),np.empty((20,15)))


def test_footprints_conserve_regular_native_area_and_exclude_scan_edges():
    x,y=np.meshgrid(np.arange(12)*1000.,np.arange(20)*1000.)
    p=footprint(x,y,5,5,10,1000)
    assert p.area==1000000 and p.bounds==(4500,4500,5500,5500)
    assert footprint(x,y,9,5,10,1000) is None
    assert footprint(x,y,10,5,10,1000) is None
    x[4,4]=np.inf
    assert footprint(x,y,5,5,10,1000) is None


def test_whole_native_support_respects_holdout_and_buffer():
    area={'id':'greater_london','extent_m':[0,0,30000,30000]}
    assert partition(box(10500,4500,11500,5500),area,500)=='holdout_buffer'
    assert partition(box(9500,4500,10500,5500),area,500)=='spatial_holdout'
    assert partition(box(12500,4500,13500,5500),area,500)=='fit_safe_geometry_only'
    assert partition(box(500,12000,1000,13000),area,600)=='pilot_boundary'
    assert partition(box(12500,4500,13500,5500),{**area,'id':'cabauw'},500)=='cabauw_reference_only'


def test_scale_and_metadata_identity_are_checked():
    validate_scale({'_Scale':.02,'_Offset':0,'_FillValue':0},.02,0)
    with pytest.raises(ValueError):validate_scale({'scale_factor':.2,'add_offset':0},.02)
    attrs={'CoreMetadata.0':'OBJECT = RANGEBEGINNINGDATE\n VALUE = "2021-07-15"\nEND_OBJECT = RANGEBEGINNINGDATE'}
    assert attr(attrs,'RangeBeginningDate')=='2021-07-15'
    attrs={'RangeBeginningDate':b'2021-07-15','RangeBeginningTime':b'11:05:00.000',
           'RangeEndingDate':b'2021-07-15','RangeEndingTime':b'11:10:00.000','LocalGranuleID':'MOD21.example.hdf'}
    r={'product':'MOD21','stem':'MOD21.example','granule_start_utc':'2021-07-15T11:05Z','granule_end_utc':'2021-07-15T11:10Z'}
    validate_interval(attrs,r)
    with pytest.raises(ValueError,match='interval'):validate_interval(attrs,{**r,'granule_end_utc':'2021-07-15T11:11Z'})
