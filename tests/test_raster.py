"""Scientific contracts for bounded inference; no remote source reads in tests."""
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
from shapely.ops import transform

from lst_pilot.raster import (RasterGrid, build_grid, validate_time, aggregate_optical,
                              worldcover_fractions, terrain_arrays, feature_frame,
                              manual_air_reference, _overlay)
from lst_pilot.satellite import SR_BANDS
from lst_pilot.terrain import terrain_descriptors

REGION = {"id": "greater_london", "epsg": 27700, "extent_m": [492800, 138400, 572800, 218400]}


def selection(bounds):
    return mapping(transform(Transformer.from_crs(27700, 4326, always_xy=True).transform, box(*bounds).segmentize(100)))


def simple_grid(n=2, resolution=100):
    t = from_origin(0, n*resolution, resolution, resolution)
    return RasterGrid(32631, t, n, n, (0, 0, n*resolution, n*resolution), None, None, np.ones((n,n), bool), 0, 0)


def test_geometry_bounds_count_aligned_bbox_and_preserve_holes():
    grid = build_grid(REGION, selection((528800, 178400, 533800, 183400)))
    assert grid.shape == (50, 50)
    assert grid.inside.all()
    assert grid.bounds == (528800.,178400.,533800.,183400.)
    with pytest.raises(ValueError, match="fully inside"):
        build_grid(REGION, selection((491800, 178400, 493800, 180400)))
    # The large-area limit counts every cell in the complete raster bbox.
    oversized_region = {**REGION,"extent_m":[492800,138400,582800,228400]}
    with pytest.raises(ValueError, match="640,000"):
        build_grid(oversized_region, selection((492800,138400,582800,228400)))
    projected = box(528800,178400,529100,178700).difference(box(528900,178500,529000,178600))
    polygon = mapping(transform(Transformer.from_crs(27700,4326,always_xy=True).transform,projected))
    holed = build_grid(REGION,polygon)
    assert holed.shape == (3,3) and holed.inside.sum() == 8


def test_time_contract_blocks_future_scenes_stale_optics_and_implicit_experimental():
    scene = {"properties":{"datetime":"2023-06-01T10:00:00.123456Z"}}
    _, _, age = validate_time(scene,"2023-06-01T10:00:00.123456Z","observed")
    assert age == 0
    with pytest.raises(ValueError,match="exact"):
        validate_time(scene,"2023-06-01T11:00:00Z","observed")
    with pytest.raises(ValueError,match="future"):
        validate_time(scene,"2023-06-01T09:00:00Z","experimental")
    with pytest.raises(ValueError,match="90-day"):
        validate_time(scene,"2023-09-01T10:00:00Z","experimental")
    with pytest.raises(ValueError,match="timezone"):
        validate_time(scene,"2023-06-01T10:00:00","observed")
    assert validate_time(scene,"2023-06-02T10:00:00.123456Z","experimental")[2] == 1


def test_optical_aggregation_never_uses_thermal_values_or_qa_uncertainty():
    raw = {band: np.full((20,20),18000,np.uint16) for band in SR_BANDS}
    raw.update(qa_pixel=np.zeros((20,20),np.uint16), qa_radsat=np.zeros((20,20),np.uint16),
               lwir11=np.full((20,20),42000,np.uint16),qa=np.zeros((20,20),np.uint16))
    grid = simple_grid()
    a, _ = aggregate_optical(raw,from_origin(0,200,10,10),32631,grid)
    raw["lwir11"][:] = 0
    raw["qa"][:] = 65535
    b, _ = aggregate_optical(raw,from_origin(0,200,10,10),32631,grid)
    assert a.keys() == b.keys()
    for feature in a:
        np.testing.assert_array_equal(a[feature],b[feature])
    # Optical clouds actually change the valid-area calculation.
    raw["qa_pixel"][:3,:10] = 8
    c, _ = aggregate_optical(raw,from_origin(0,200,10,10),32631,grid)
    assert np.isclose(c["optical_valid_fraction"][0,0],.7)
    assert np.isnan(c["ndvi"][0,0])
    assert np.isfinite(c["ndvi"][1,1])


def test_land_mask_keeps_snow_mangroves_and_counts_nodata_in_denominator():
    grid = simple_grid()
    classes = np.full((20,20),10,np.uint8)
    classes[:10,:10] = 70
    classes[:10,10:] = 95
    classes[10:,:10] = 80
    classes[10:13,10:] = 0
    land, water = worldcover_fractions(classes,from_origin(0,200,10,10),32631,grid)
    np.testing.assert_allclose(land,[[1,1],[0,.7]],atol=1e-6)
    np.testing.assert_allclose(water,[[0,0],[1,0]])


def test_vectorized_terrain_matches_training_stencils_and_preserves_missing():
    z = np.arange(35,dtype=float).reshape(5,7)
    z[0,0] = np.nan
    result = terrain_arrays(z)
    aliases = {"elevation":"elevation_m","slope":"slope_deg","aspect_sin":"aspect_sin","aspect_cos":"aspect_cos","terrain_relief_300m":"terrain_relief_300m"}
    for row in range(3):
        for col in range(5):
            expected = terrain_descriptors(z[row:row+3,col:col+3])
            for name,old in aliases.items():
                np.testing.assert_allclose(result[name][row,col],expected[old],equal_nan=True)


def test_row_identity_and_manual_air_preserve_background_spatial_gradient(monkeypatch):
    grid = build_grid(REGION,selection((528800,178400,529000,178600)))
    target = pd.Timestamp("2023-06-01T10:00:00Z")
    frame = feature_frame(REGION,grid,{"ndvi":np.array([[.1,.2],[.3,.4]])},target)
    assert frame._raster_position.tolist() == [0,1,2,3]
    assert frame.ndvi.tolist() == [.1,.2,.3,.4]
    frame["background_air_temperature_c"] = [19,20,21,22]
    def fake(query,*args,**kwargs):
        query = query.copy()
        query["background_air_temperature_c"] = 20
        return query,[]
    monkeypatch.setattr("lst_pilot.weather.enrich_weather",fake)
    actual,audit = manual_air_reference(frame,25,grid,target,"cache")
    np.testing.assert_array_equal(actual.air_temperature_c,[24,25,26,27])
    assert audit["anchor"] == "projected polygon centroid"
    assert actual.station_id.eq("").all()


def test_overlay_is_web_mercator_north_up_with_transparent_nodata(tmp_path):
    from PIL import Image
    grid = build_grid(REGION,selection((528800,178400,529000,178600)))
    data = np.array([[10,np.nan],[20,30]],np.float32)
    path = tmp_path/"overlay.png"
    bounds = _overlay(path,data,grid,10,30)
    image = np.asarray(Image.open(path))
    assert bounds[0] < bounds[2] and bounds[1] < bounds[3]
    assert image.shape[2] == 4 and (image[...,3] == 0).any()


def test_comparison_legends_share_full_union_and_ignore_unpaired_observations():
    from lst_pilot.raster import comparison_legends
    prediction = np.array([[30.8,35.0],[33.0,np.nan]])
    observed = np.array([[32.12,40.85],[np.nan,80.0]])
    legends = comparison_legends(prediction,observed)
    assert legends["prediction"] == legends["observed"]
    assert legends["prediction"] == {"min_c":30.8,"max_c":40.85,"cmap":"inferno"}
    assert np.isclose(legends["residual"]["max_c"],5.85)
    assert legends["residual"]["min_c"] == -legends["residual"]["max_c"]
    assert legends["residual"]["cmap"] == "RdBu_r"
    # Narrow ranges receive a five-degree display span; values remain untouched.
    for reference in (None,np.full((2,2),np.nan)):
        legend = comparison_legends(prediction,reference)["prediction"]
        assert np.isclose(legend["max_c"]-legend["min_c"],5.)
        assert legend["min_c"] <= 30.8 and legend["max_c"] >= 35.


def test_subdegree_variation_does_not_fill_temperature_colour_scale():
    from lst_pilot.raster import comparison_legends
    values = np.array([13.22,13.86])
    original = values.copy()
    legend = comparison_legends(values)["prediction"]
    assert legend["max_c"]-legend["min_c"] == 5.
    np.testing.assert_array_equal(values,original)


def test_actual_feature_summary_uses_selected_rows_and_serializes_missing_values():
    import json
    from lst_pilot.raster import summarize_features
    frame = pd.DataFrame({"air_temperature_c":[20.,22.,24.,100.],
                          "ndvi":[.2,np.nan,np.inf,.9],
                          "climate_class":["Cfb","Aw","Cfb","ET"]})
    result = summarize_features(frame.iloc[:3],["air_temperature_c","ndvi","climate_class"])
    assert result["air_temperature_c"] == {"min":20.,"max":24.,"mean":22.,"missing":0}
    assert result["ndvi"] == {"min":.2,"max":.2,"mean":.2,"missing":2}
    assert result["climate_class"] == {"counts":{"Aw":1,"Cfb":2}}
    assert json.loads(json.dumps(result,allow_nan=False)) == result


def test_full_pilot_canonical_boundary_fits_exact_grid_and_tiles_cover_once():
    from lst_pilot.raster import grid_tiles
    # Same 20 segments per edge used by the public pilot catalog.
    canonical = mapping(transform(Transformer.from_crs(27700,4326,always_xy=True).transform,
                                  box(*REGION["extent_m"]).segmentize(4000)))
    grid = build_grid(REGION,canonical)
    assert grid.shape == (800,800) and grid.inside.all()
    assert grid.bounds == tuple(REGION["extent_m"])
    visits = np.zeros(grid.shape,np.uint8)
    for slices,tile in grid_tiles(grid):
        assert tile.height <= 128 and tile.width <= 128
        assert tile.transform*(.5,.5) == grid.transform*(slices[1].start+.5,slices[0].start+.5)
        visits[slices] += 1
    assert (visits == 1).all()


def test_tiled_worldcover_matches_full_window_including_tile_edges(tmp_path,monkeypatch):
    import rasterio
    import lst_pilot.raster as module
    rng = np.random.default_rng(4)
    classes = rng.choice(np.array([0,10,70,80,95,100],np.uint8),(60,60))
    t = from_origin(0,600,10,10)
    path = tmp_path/"worldcover.tif"
    with rasterio.open(path,"w",driver="GTiff",width=60,height=60,count=1,dtype="uint8",nodata=0,crs="EPSG:32631",transform=t) as dst:
        dst.write(classes,1)
    grid = simple_grid(6)
    expected = worldcover_fractions(classes,t,32631,grid)
    item = {"id":"synthetic","properties":{"esa_worldcover:product_version":"2.0.0"},"assets":{"map":{"href":str(path)}}}
    monkeypatch.setattr(module,"_stac_tiles",lambda *a,**k:[item])
    original = module.grid_tiles
    monkeypatch.setattr(module,"grid_tiles",lambda g:original(g,tile_size=2))
    land,water,audit = module.read_land_mask(grid,tmp_path)
    np.testing.assert_allclose(land,expected[0],atol=1e-6)
    np.testing.assert_allclose(water,expected[1],atol=1e-6)
    assert audit["processing_tiles"] == 9


def test_window_climate_and_chunked_solar_equal_existing_point_context(tmp_path):
    import rasterio
    from lst_pilot.context import add_context
    from lst_pilot.raster import add_raster_context
    path = tmp_path/"climate.tif"
    with rasterio.open(path,"w",driver="GTiff",width=3,height=3,count=1,dtype="uint8",nodata=0,
                       crs="EPSG:4326",transform=from_origin(-.3,51.8,.1,.1)) as dst:
        dst.write(np.array([[15,0,27],[3,4,29],[15,15,15]],np.uint8),1)
    frame = pd.DataFrame({"longitude":[-.25,-.15,-.05,-.3,0.05],"latitude":[51.75,51.75,51.65,51.55,51.7],
                          "datetime_utc":pd.to_datetime(["2023-09-07T10:52:24Z"]*5,utc=True)})
    expected = add_context(frame,path)
    actual = add_raster_context(frame,path,chunk_size=2)
    pd.testing.assert_series_equal(actual.climate_class,expected.climate_class)
    assert actual.climate_class.tolist() == ["Cfb","unknown","ET","Cfb","unknown"]
    for column in expected.columns.difference(frame.columns):
        if expected[column].dtype.kind in "fc":
            np.testing.assert_allclose(actual[column],expected[column],rtol=0,atol=1e-10)


def test_chunked_prediction_keeps_noncontiguous_row_identity(monkeypatch):
    from lst_pilot.raster import predict_chunks
    frame = pd.DataFrame({"air_temperature_c":[20.,21.,22.,23.,24.]},index=[7,2,10,4,20])
    lengths = []
    def fake(part,bundle):
        lengths.append(len(part))
        return pd.DataFrame({"predicted_lst_c":part.air_temperature_c+3},index=part.index)
    monkeypatch.setattr("lst_pilot.model.predict_frame",fake)
    result = predict_chunks(frame,{},chunk_size=2)
    assert lengths == [2,2,1]
    assert result.index.tolist() == frame.index.tolist()
    assert result.predicted_lst_c.tolist() == [23.,24.,25.,26.,27.]


def test_exact_warp_keeps_overlapping_pixels_invariant_to_request_extent():
    import math
    from types import SimpleNamespace
    from rasterio.warp import transform_bounds
    import lst_pilot.raster as module
    big = build_grid(REGION,selection((522800,168400,542800,188400)))
    small = build_grid(REGION,selection((530300,175900,535300,180900)))
    w,s,e,n = transform_bounds(27700,32631,*big.bounds,densify_pts=21)
    left,top = math.floor(w/30)*30-300,math.ceil(n/30)*30+300
    native_transform = from_origin(left,top,30,30)
    rows,cols = math.ceil((top-s)/30)+10,math.ceil((e-left)/30)+10
    values = np.random.default_rng(123).uniform(.1,.9,(rows,cols)).astype("float32")
    source = SimpleNamespace(crs="EPSG:32631",transform=native_transform)
    win = module._native_window(source,small)
    row,col = int(win.row_off),int(win.col_off)
    subset = values[row:row+int(win.height),col:col+int(win.width)]
    subset_transform = native_transform*module.rasterio.Affine.translation(col,row)
    full = module._warp(values,native_transform,32631,big)
    cropped = module._warp(subset,subset_transform,32631,small)
    np.testing.assert_array_equal(full[75:125,75:125],cropped)
