import numpy as np
import pandas as pd
from lst_pilot.option_b_cohort import choose_night_positions, spatial_flags


def test_sampling_is_deterministic_bounded_and_only_uses_validity():
    valid=np.ones((500,500),bool);valid[:100,:200]=False
    first,audit=choose_night_positions(valid,'scene')
    second,_=choose_night_positions(valid.copy(),'scene')
    np.testing.assert_array_equal(first,second)
    assert len(first)==400 and len(audit)==4
    assert valid[first[:,0],first[:,1]].all()
    assert len({tuple(p) for p in first})==400
    assert all(t['sampled_cells']<=100 for t in audit)


def test_empty_and_sparse_quadrants_do_not_fabricate_samples():
    valid=np.zeros((500,500),bool)
    assert choose_night_positions(valid,'scene')[0].shape==(0,2)
    valid[20:23,20:25]=True
    positions,_=choose_night_positions(valid,'scene')
    assert len(positions)==15


def test_safe_sampling_survives_more_abundant_heldout_coverage():
    valid=np.ones((500,500),bool)
    safe=np.zeros_like(valid);safe[140:160,140:160]=True
    heldout=valid & ~safe
    positions,audit=choose_night_positions(valid,'scene',safe=safe,reserved=heldout)
    assert safe[positions[:,0],positions[:,1]].sum()==80
    assert heldout[positions[:,0],positions[:,1]].sum()==80
    assert len(positions)<=400
    assert len({tuple(p) for p in positions})==len(positions)


def test_permanent_blocks_use_south_origin_and_exact_outside_buffer():
    areas={'greater_london':{'extent_m':[0,0,50000,50000]}}
    # Cell footprints in reserved r0c0; 900 m / 1000 m east; distant safe r1c0.
    rows=pd.DataFrame({'region_id':['greater_london']*5,'grid_row':[450,450,450,450,350],
                       'grid_col':[50,109,110,120,50]})
    result=spatial_flags(rows,areas)
    assert result.spatial_holdout.tolist()==[True,False,False,False,False]
    assert result.in_holdout_buffer.tolist()==[False,True,True,False,False]
    assert result.block_id.iloc[0]=='greater_london_r00_c00'
    assert result.block_id.iloc[4]=='greater_london_r01_c00'
    pd.testing.assert_frame_equal(result,spatial_flags(rows,areas))


def test_diagonal_cell_corner_inside_buffer_is_excluded_even_if_centre_is_outside():
    area={'greater_london':{'extent_m':[0,0,50000,50000]}}
    frame=pd.DataFrame({'region_id':['greater_london'],'grid_row':[392],'grid_col':[107]})
    # (10750,10750) centre is1061m from block; closest corner is990m away.
    result=spatial_flags(frame,area)
    assert not result.spatial_holdout.iloc[0]
    assert result.in_holdout_buffer.iloc[0]
