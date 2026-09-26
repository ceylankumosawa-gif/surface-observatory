"""Native raster cell lookup: independent geometry, mask and value checks."""
import math

from affine import Affine
import numpy as np
from pyproj import Transformer
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from lst_pilot.raster_pixel import read_pixel


def write_raster(path, data, descriptions, *, transform=None, crs="EPSG:4326", nodata=-9999):
    data = np.asarray(data, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                       count=len(data), dtype="float32", nodata=nodata, crs=crs,
                       transform=transform or from_origin(10, 50, 1, 1)) as out:
        out.write(data)
        for index, description in enumerate(descriptions, 1):
            if description is not None:
                out.set_band_description(index, description)
    return path


def test_named_bands_reordered_preserve_zero_negative_and_residual_sign(tmp_path):
    path = write_raster(tmp_path/"x.tif", [[[-2, 5]], [[99, 99]], [[0, -3]], [[2, -8]]],
                        ["observed_lst_c", "upper_lst_c", "predicted_lst_c", "residual_prediction_minus_observed_c"])
    left = read_pixel(path, 10.5, 49.5)
    assert left["status"] == "ok" and left["units"] == "degC"
    assert left["values"] == {"prediction": 0., "observed": -2., "residual": 2.}
    right = read_pixel(path, 11.5, 49.5)
    assert right["values"] == {"prediction": -3., "observed": 5., "residual": -8.}
    assert right["pixel"] == {"row": 0, "column": 1, "center": [11.5, 49.5],
                              "corners": [[11.,50.],[12.,50.],[12.,49.],[11.,49.],[11.,50.]]}


@pytest.mark.parametrize("invalid", [-9999, float("nan"), float("inf"), -float("inf")])
def test_nodata_and_nonfinite_are_null_with_native_cell_geometry(tmp_path, invalid):
    path = write_raster(tmp_path/"x.tif", [[[invalid]]], ["predicted_lst_c"])
    result = read_pixel(path, 10.5, 49.5)
    assert result["status"] == "nodata"
    assert result["values"] == {"prediction": None, "observed": None, "residual": None}
    assert result["pixel"]["row"] == 0 and len(result["pixel"]["corners"]) == 5


def test_valid_observation_with_masked_prediction_and_missing_residual(tmp_path):
    path = write_raster(tmp_path/"x.tif", [[[-9999]], [[0]]], ["predicted_lst_c", "observed_lst_c"])
    result = read_pixel(path, 10.5, 49.5)
    assert result["status"] == "ok"
    assert result["values"] == {"prediction": None, "observed": 0., "residual": None}


def test_band_positions_are_never_guessed_and_ambiguous_names_fail(tmp_path):
    path = write_raster(tmp_path/"x.tif", [[[20]], [[21]]], [None, "upper_lst_c"])
    assert read_pixel(path, 10.5, 49.5)["status"] == "nodata"
    write_raster(path, [[[20]], [[21]]], ["predicted_lst_c", "predicted_lst_c"])
    with pytest.raises(ValueError, match="ambiguous"):
        read_pixel(path, 10.5, 49.5)


@pytest.mark.parametrize("lon,lat,row,column", [
    (10,50,0,0), (11,49,1,1), (12.999,48.001,1,2),
    (13,49,None,None), (11,48,None,None), (9.999,49,None,None), (11,50.001,None,None),
])
def test_half_open_native_edges_and_floor_ownership(tmp_path, lon, lat, row, column):
    path = write_raster(tmp_path/"x.tif", [np.arange(6).reshape(2,3)], ["predicted_lst_c"])
    result = read_pixel(path, lon, lat)
    if row is None:
        assert result["status"] == "outside" and result["pixel"] is None
        assert all(value is None for value in result["values"].values())
    else:
        assert result["pixel"]["row"] == row and result["pixel"]["column"] == column
        assert result["values"]["prediction"] == row*3+column


def test_projected_rotated_cell_corners_and_one_pixel_read(tmp_path, monkeypatch):
    affine = Affine.translation(700000,5700000) * Affine.rotation(23) * Affine.scale(100,-100)
    path = write_raster(tmp_path/"x.tif", [np.arange(6).reshape(2,3)], ["predicted_lst_c"],
                        transform=affine, crs="EPSG:32630")
    geographic = Transformer.from_crs(32630,4326,always_xy=True)
    projected = Transformer.from_crs(4326,32630,always_xy=True)
    query = geographic.transform(*(affine*(1.2,.7)))
    real_open = rasterio.open
    calls = []

    class Reader:
        def __init__(self, dataset): self.dataset = dataset
        def __getattr__(self, name): return getattr(self.dataset, name)
        def __enter__(self): return self
        def __exit__(self, *args): self.dataset.close()
        def read(self, indexes, **kwargs):
            calls.append((indexes, kwargs))
            assert kwargs["window"] == Window(1,0,1,1)
            assert kwargs["masked"] is True
            return self.dataset.read(indexes, **kwargs)

    monkeypatch.setattr(rasterio,"open",lambda *args,**kwargs:Reader(real_open(*args,**kwargs)))
    result = read_pixel(path,*query)
    assert result["pixel"]["row"] == 0 and result["pixel"]["column"] == 1
    assert result["values"]["prediction"] == 1 and len(calls) == 1
    assert projected.transform(*result["pixel"]["center"]) == pytest.approx(affine*(1.5,.5), abs=1e-6)
    for corner, offset in zip(result["pixel"]["corners"],[(1,0),(2,0),(2,1),(1,1),(1,0)]):
        assert projected.transform(*corner) == pytest.approx(affine*offset, abs=1e-6)
    # Exact transformed upper-left is included, lower/right outer edges excluded.
    monkeypatch.setattr(rasterio,"open",real_open)
    assert read_pixel(path,*geographic.transform(*(affine*(0,0))))["status"] == "ok"
    for offset in [(3,1),(1,2),(-.2,1.2)]:
        assert read_pixel(path,*geographic.transform(*(affine*offset)))["status"] == "outside"


def test_internal_mask_is_respected(tmp_path):
    path = write_raster(tmp_path/"x.tif", [[[23,0]]], ["predicted_lst_c"])
    with rasterio.open(path,"r+") as out:
        out.write_mask(np.array([[0,255]],dtype="uint8"))
    assert read_pixel(path,10.5,49.5)["status"] == "nodata"
    assert read_pixel(path,11.5,49.5)["values"]["prediction"] == 0


@pytest.mark.parametrize("lon,lat", [(math.nan,0),(0,math.inf),(181,0),(0,-91)])
def test_invalid_coordinates_rejected_before_file_open(tmp_path,lon,lat):
    with pytest.raises(ValueError, match="Coordinates"):
        read_pixel(tmp_path/"does-not-exist.tif",lon,lat)


def test_disguised_vrt_is_not_opened_as_a_local_temperature_raster(tmp_path):
    path=tmp_path/"prediction.tif"
    path.write_text('<VRTDataset rasterXSize="1" rasterYSize="1"><VRTRasterBand dataType="Float32" band="1"/></VRTDataset>')
    with pytest.raises(rasterio.errors.RasterioIOError):
        read_pixel(path,0,0)
