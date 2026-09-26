"""Day expansion preserves optical independence and pilot pixel identity."""
import numpy as np
import pytest
from rasterio.transform import from_origin
from lst_pilot.option_b_day import fixed_windows, aggregate_labels, sample_scene
from lst_pilot.raster import RasterGrid
from lst_pilot.satellite import SR_BANDS


def test_thermal_quality_changes_labels_but_never_optical_predictors():
    grid = RasterGrid(32631, from_origin(0,200,100,100), 2,2,(0,0,200,200),None,None,np.ones((2,2),bool),0,0)
    raw = {b: np.full((20,20),18000,np.uint16) for b in SR_BANDS}
    raw.update(qa_pixel=np.zeros((20,20),np.uint16),qa_radsat=np.zeros((20,20),np.uint16),
               lwir11=np.full((20,20),42000,np.uint16),qa=np.full((20,20),100,np.uint16))
    a, labels_a = aggregate_labels(raw,from_origin(0,200,10,10),32631,grid)
    raw['qa'][:3,:10] = 400
    raw['lwir11'][10:,10:] = 0
    b, labels_b = aggregate_labels(raw,from_origin(0,200,10,10),32631,grid)
    for name in a: np.testing.assert_array_equal(a[name],b[name])
    assert np.isfinite(labels_a['lst_c']).all()
    assert np.isnan(labels_b['lst_c'][0,0]) and np.isnan(labels_b['lst_c'][1,1])
    assert np.isclose(labels_b['label_valid_fraction'][0,0],.7)
    assert np.isclose(labels_b['max_source_lst_error_k'][0,0],1)


def test_windows_are_unique_inside_and_aligned_to_pilot_grid():
    region={'id':'greater_london','epsg':27700,'extent_m':[492800,138400,572800,218400], 'grid_shape':[800,800]}
    windows=list(fixed_windows(region))
    assert len(windows)==8
    assert len({tuple(g.bounds) for _,g in windows})==8
    for _,g in windows:
        assert g.shape==(20,20)
        assert 0<=g.row_offset<=780 and 0<=g.col_offset<=780
        assert g.transform.c==492800+g.col_offset*100
        assert g.transform.f==218400-g.row_offset*100


@pytest.mark.parametrize('year',[2020,2024,2025])
def test_reserved_years_rejected_before_source_open(year):
    scene={'assets':{b:{'href':'must-not-open'} for b in (*SR_BANDS,'qa_pixel','qa_radsat','lwir11','qa')},
           'properties':{'datetime':f'{year}-06-01T10:00:00Z'}}
    with pytest.raises(ValueError,match='reserved test years'):
        sample_scene(scene,{'id':'test'})
