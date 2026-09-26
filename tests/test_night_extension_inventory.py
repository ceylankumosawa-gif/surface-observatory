from copy import deepcopy

from lst_pilot.night_extension_inventory import select


def row(orbit='12345', tile='30UXC', build='0712', gid='G1-LPCLOUD', date='2021-01-01'):
    return {'pilot_id':'greater_london', 'product':'ecostress_v2', 'temporal_split':'fit',
            'time_start':date+'T01:00:00Z', 'utc_date':date, 'granule_concept_id':gid,
            'granule_title':f'ECOv002_L2T_LSTE_{orbit}_004_{tile}_{date.replace("-", "")}T010000_{build}_01',
            'identity':{'acquisition_group':f'ECOSTRESS:orbit:{orbit}'},
            'solar_elevation_max_centre_corners':-20, 'actual_phase':'night',
            'local_solar_3hour_bin':0, 'metadata_overlap_m2':100,
            'footprint_qa':{'status':'metadata_consistent_pixels_unverified'}}


def test_excludes_whole_orbit_including_alternate_native_tiles():
    records=[row(), row(tile='30UXB',gid='G2-LPCLOUD')]
    assert select(records,{('greater_london','12345')},set())==[]


def test_registered_whole_pilot_date_excludes_unattempted_orbit():
    assert select([row()],set(),{('greater_london','2021-01-01')})==[]
    assert len(select([row()],set(),{('sioux_falls','2021-01-01')}))==1


def test_latest_build_before_metadata_overlap():
    old=row(build='0711');old['metadata_overlap_m2']=1000
    new=row(gid='G2-LPCLOUD')
    assert select([old,new],set(),set())[0]['granule_concept_id']=='G2-LPCLOUD'


def test_excludes_reserved_years_and_actual_twilight():
    records=[row(date='2024-01-01'),row(date='2023-01-01')]
    twilight=deepcopy(row());twilight['solar_elevation_max_centre_corners']=-5.99
    assert select(records+[twilight],set(),set())==[]


def test_metadata_inconsistent_footprint_cannot_form_candidate():
    candidate=row();candidate['footprint_qa']['status']='metadata_mismatch'
    assert select([candidate],set(),set())==[]
