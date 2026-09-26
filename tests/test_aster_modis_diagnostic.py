import numpy as np
import pytest
from affine import Affine
from shapely.geometry import Polygon,box
from lst_pilot.aster_modis_diagnostic import aggregate_polygon


def test_exact_partial_native_area_and_invalid_support():
    affine=Affine.translation(0,90)*Affine.scale(90,-90)
    values=np.array([[10.,30.]])
    estimate,fraction,count=aggregate_polygon(box(45,0,180,90),values,np.ones((1,2),bool),affine)
    assert estimate==pytest.approx(70/3)
    assert fraction==pytest.approx(1)
    assert count==2
    estimate,fraction,count=aggregate_polygon(box(45,0,180,90),values,np.array([[True,False]]),affine)
    assert estimate==10
    assert fraction==pytest.approx(1/3)
    assert count==1


def test_rotated_native_grid_retains_area():
    affine=Affine.rotation(12)*Affine.scale(90,-90)
    poly=Polygon([affine*p for p in ((0,0),(2,0),(2,1),(0,1),(0,0))])
    estimate,fraction,count=aggregate_polygon(poly,np.array([[10.,30.]]),np.ones((1,2),bool),affine)
    assert estimate==pytest.approx(20)
    assert fraction==pytest.approx(1)
    assert count==2
