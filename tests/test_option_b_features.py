"""Research pairing contracts: causality, label independence and stable identity."""
import json

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin

from lst_pilot import option_b_features as features


AREA = {"id": "test", "epsg": 32614, "extent_m": [658000, 4813800, 708000, 4863800], "grid_shape": [500, 500]}


def samples(n=3):
    row, col = np.arange(n) + 249, np.arange(n) + 100
    lon, lat = Transformer.from_crs(32614, 4326, always_xy=True).transform(658000 + (col + .5) * 100, 4863800 - (row + .5) * 100)
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "region_id": "test", "acquisition_id": "scene",
                         "datetime_utc": pd.Timestamp("2022-07-02T03:30:00Z"), "grid_row": row, "grid_col": col,
                         "epsg": 32614, "latitude": lat, "longitude": lon, "lst_c": np.arange(n) + 19., "label_product": "ECOSTRESS"})


def optical_item(name, date, thermal=False):
    item = {"id": name, "collection": "landsat-c2-l2", "properties": {"datetime": date, "platform": "landsat-8",
            "landsat:collection_category": "T1", "landsat:correction": "L2SR"},
            "assets": {name: {"href": f"https://example.test/{name}.tif"} for name in (*features.satellite.SR_BANDS, "qa_pixel", "qa_radsat")}}
    if thermal:
        item["assets"]["lwir11"] = {"href": "https://example.test/NEVER_READ.tif"}
    return item


def optical(value, valid=1):
    return {**{name: np.full((1, 2), value) for name in features.SR_FIELDS},
            "optical_valid_fraction": np.full((1, 2), valid), "water_fraction": np.zeros((1, 2))}


def test_sample_identity_and_fixed_centres():
    frame = samples()
    assert features.validate_samples(frame, {"test": AREA}).sample_id.tolist() == frame.sample_id.tolist()
    with pytest.raises(ValueError, match="unique"):
        features.validate_samples(pd.concat([frame, frame]), {"test": AREA})
    shifted = frame.copy()
    shifted.loc[0, "latitude"] += .01
    with pytest.raises(ValueError, match="centre"):
        features.validate_samples(shifted, {"test": AREA})
    naive = frame.copy()
    naive["datetime_utc"] = pd.Timestamp("2022-07-02")
    with pytest.raises(ValueError, match="timezone"):
        features.validate_samples(naive, {"test": AREA})


def test_adapter_reorder_and_identity_corruption():
    original = samples()
    reverse = original.iloc[::-1].copy()
    reverse["added"] = [1, 2, 3]
    result = features.preserve_identity(original, reverse)
    assert result.sample_id.tolist() == ["s0", "s1", "s2"]
    assert result.added.tolist() == [3, 2, 1]
    reverse.loc[0, "lst_c"] = 300
    with pytest.raises(ValueError, match="immutable"):
        features.preserve_identity(original, reverse)
    with pytest.raises(ValueError, match="duplicated"):
        features.preserve_identity(original, pd.concat([original.iloc[:2], original.iloc[:1]]))


def test_sr_search_is_past_only_and_does_not_require_thermal():
    items = [optical_item("past", "2022-06-22T12:00:00Z"), optical_item("future", "2022-07-03T12:00:00Z"),
             optical_item("old", "2022-05-01T12:00:00Z"), optical_item("now", "2022-07-02T03:30:00Z", thermal=True)]
    chosen, audit = features.choose_optical(items, "2022-07-02T03:30:00Z")
    assert [item["id"] for item in chosen] == ["now", "past"]
    assert audit["eligible_scenes"] == 2
    # Changing thermal asset availability cannot affect selection.
    items[-1]["assets"].pop("lwir11")
    assert features.choose_optical(items, "2022-07-02T03:30:00Z")[0] == chosen


def test_composite_masks_before_median_and_records_actual_time_support():
    first, second, cloudy = optical(.2), optical(.6), optical(.99, .79)
    second["sr_blue"][0, 1] = np.nan
    arrays = features.composite_optical([first, second, cloudy],
                                       ["2022-06-10T12:00Z", "2022-06-20T12:00Z", "2022-06-30T12:00Z"],
                                       "2022-07-02T03:30Z", (1, 2))
    np.testing.assert_allclose(arrays["sr_red"], [[.4, .2]])
    np.testing.assert_equal(arrays["optical_observation_count"], [[2, 1]])
    assert arrays["optical_latest_epoch_s"][0, 1] == pd.Timestamp("2022-06-10T12:00Z").timestamp()
    with pytest.raises(ValueError, match="Future"):
        features.composite_optical([first], ["2022-07-03T12:00Z"], "2022-07-02T03:30Z", (1, 2))
    empty = features.composite_optical([], [], "2022-07-02T03:30Z", (1, 2))
    assert np.isnan(empty["ndvi"]).all() and not empty["optical_observation_count"].any()


def test_optical_reuse_rejects_future_and_stale_explicit_source():
    data = samples()
    for name in features.SR_FIELDS:
        data[name] = .2
    data["water_fraction"] = 0.
    data["optical_source_id"] = "independent_scene"
    data["optical_source_datetime_utc"] = pd.Timestamp("2022-06-28T12:00Z")
    assert features._reuse_optical(data, 32)
    for date in ("2022-07-02T03:31Z", "2022-04-01T00:00Z"):
        data["optical_source_datetime_utc"] = pd.Timestamp(date)
        with pytest.raises(ValueError, match="future or outside"):
            features._reuse_optical(data, 32)


def test_worldcover_denominator_and_support_not_relabelled_land():
    # Each target pixel contains 100 native cells. Unknown values stay in the denominator.
    grid = features.raster.RasterGrid(32614, from_origin(0, 100, 100, 100), 1, 3, (0, 0, 300, 100), None, None, None, 0, 0)
    classes = np.full((10, 30), 90, np.uint8)  # Other valid wetland class.
    classes[:, :5] = 10
    classes[:, 10:20] = 50
    classes[:1, 10:11] = 0  # 99% classified: acceptable denominator stays 100.
    classes[:, 20:30] = 40
    classes[:1, 20:30] = 0  # 90% classified: five added predictors stay missing.
    result = features.cover_arrays(classes, from_origin(0, 100, 10, 10), "EPSG:32614", grid)
    assert result["worldcover_tree_class_fraction"][0, 0] == pytest.approx(.5)
    assert sum(result[name][0, 0] for name in features.COVER_FEATURES) == pytest.approx(.5)
    assert result["worldcover_built_class_fraction"][0, 1] == pytest.approx(.99)
    assert np.isnan(result["worldcover_crop_class_fraction"][0, 2])
    assert result["worldcover_classified_fraction"][0, 2] == pytest.approx(.9)


def test_surface_weight_group_ignores_temperature_and_unknown_is_explicit():
    frame = pd.DataFrame([[.7, .1, 0., .2, 0.], [.1, .1, .1, .1, .1], [np.nan] * 5], columns=features.COVER_FEATURES)
    frame["lst_c"] = [-80, 100, 1]
    assert features.surface_group(frame).tolist() == ["tree", "other_or_mixed", "unknown"]
    frame["lst_c"] = [100, -80, 50]
    assert features.surface_group(frame).tolist() == ["tree", "other_or_mixed", "unknown"]


def test_fixed_tiles_do_not_depend_on_sample_selection():
    first, neighbour = features.tile_grid(AREA, 249, 100), features.tile_grid(AREA, 250, 101)
    assert first.transform == neighbour.transform
    assert (first.row_offset, first.col_offset) == (128, 0)
    edge = features.tile_grid(AREA, 499, 499)
    assert edge.shape == (116, 116)


def test_memory_deduplicates_joint_grids_and_never_passes_current_air(monkeypatch):
    data = samples()
    data["air_temperature_c"] = [100, 0, -80]
    calls = []
    def memory(query, cache, **kwargs):
        assert "air_temperature_c" not in query
        calls.append(query.copy())
        result = query.copy()
        result["memory_air_mean_6h_c"] = 11.
        result["memory_status"] = "complete"
        return result, {}
    monkeypatch.setattr(features.thermal_memory, "add_thermal_memory", memory)
    output, report = features.append_memory(data, "/unused")
    assert len(calls[0]) == 1 and report["sample_rows"] == 3
    assert output.sample_id.tolist() == data.sample_id.tolist()
    assert output.air_temperature_c.tolist() == [100, 0, -80]
    assert output.memory_air_mean_6h_c.tolist() == [11, 11, 11]


def test_legacy_base_is_bitwise_preserved_with_missingness_diagnostics(monkeypatch):
    data = samples()
    for index, name in enumerate(features.BASE_FEATURES):
        data[name] = "Cfb" if name == "climate_class" else np.array([index + .1, index + .2, np.nan])
    before = data[list(features.BASE_FEATURES)].copy()
    def surfaces(frame, *args, **kwargs):
        assert kwargs["preserve_existing_base"]
        for name in features.COVER_FEATURES:
            frame[name] = [0., 0., np.nan]
        return frame, {}
    def memory(frame, cache):
        for name in features.MEMORY_FEATURES:
            frame[name] = [1., np.nan, 1.]
        return frame, {}
    monkeypatch.setattr(features, "append_surfaces", surfaces)
    monkeypatch.setattr(features, "append_memory", memory)
    result, report = features.assemble_acquisition(data, {"test": AREA}, "/unused", preserve_existing_base=True)
    pd.testing.assert_frame_equal(before, result[list(features.BASE_FEATURES)], check_exact=True)
    assert result.features_A_complete.tolist() == [True, True, False]
    assert result.features_D_complete.tolist() == [True, False, False]
    assert len(result) == len(data)


def test_tile_cache_hash_verification(tmp_path):
    grid = features.tile_grid(AREA, 0, 0)
    count = []
    def reader():
        count.append(1)
        return {"x": np.ones(grid.shape)}, {"source": "independent"}
    arrays, audit = features._cached_tile("test", grid, tmp_path, "one", reader)
    repeated, _ = features._cached_tile("test", grid, tmp_path, "one", reader)
    assert len(count) == 1
    np.testing.assert_equal(arrays["x"], repeated["x"])
    from pathlib import Path
    Path(audit["array_path"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash differs"):
        features._cached_tile("test", grid, tmp_path, "one", reader)


def test_checkpoint_fingerprint_refuses_different_source(tmp_path, monkeypatch):
    frame = samples()
    for name in features.BASE_FEATURES:
        frame[name] = "Cfb" if name == "climate_class" else 1.
    input_path, areas_path = tmp_path / "input.parquet", tmp_path / "areas.json"
    frame.to_parquet(input_path)
    areas_path.write_text(json.dumps({"areas": [AREA]}))
    calls = []
    def assemble(frame, *args, **kwargs):
        calls.append(1)
        for name in ("features_A_complete", "features_B_complete", "features_C_complete", "features_D_complete", "station_pair_available"):
            frame[name] = True
        return frame, {"completeness": {"features_A_complete": len(frame)}}
    monkeypatch.setattr(features, "assemble_acquisition", assemble)
    args = (input_path, tmp_path / "output", areas_path, tmp_path / "cache")
    features.build(*args, preserve_existing_base=True)
    features.build(*args, preserve_existing_base=True)
    assert len(calls) == 1
    frame.loc[0, "lst_c"] += 1
    frame.to_parquet(input_path)
    with pytest.raises(ValueError, match="different inputs"):
        features.build(*args, preserve_existing_base=True)
