"""Small synthetic scientific checks; no network or raster downloads."""
import numpy as np
from rasterio.transform import from_origin
from lst_pilot.satellite import (SR_BANDS, aggregate_patch, qa_valid,
                                 scale_reflectance, scale_temperature, patch_origins, _redact_url_queries)


def test_usgs_scaling_examples():
    assert np.isclose(scale_reflectance(np.array([18639]))[0], .3125725)
    assert np.isclose(scale_temperature(np.array([44947]))[0], 29.47974494, atol=.001)


def test_qa_retains_snow_water_rejects_cloud_saturation():
    qa = np.array([64, 64 | 32, 64 | 128, 8, 16, 4, 2, 1, 64, 2 << 8], dtype=np.uint16)
    sat = np.zeros_like(qa)
    sat[8] = 1 << 3
    assert qa_valid(qa, sat).tolist() == [True, True, True, False, False, False, False, False, False, False]


def test_mask_precedes_aggregation_and_requires_80_percent_coverage():
    shape = (10, 20)
    raw = {b: np.full(shape, 20000, dtype=np.uint16) for b in SR_BANDS}
    raw.update(lwir11=np.full(shape, 45000, dtype=np.uint16),
               qa_pixel=np.full(shape, 64 | 32, dtype=np.uint16),
               qa_radsat=np.zeros(shape, dtype=np.uint16))
    # First destination cell has 90% valid samples, second only 70%.
    raw["qa_pixel"][:, 0] = 8
    raw["qa_pixel"][:, 10:13] = 8
    raw["lwir11"][:, 0] = 65000  # Would warm the average if masking happened later.
    result = aggregate_patch(raw, from_origin(0, 100, 10, 10), "EPSG:32631",
                             from_origin(0, 100, 100, 100), "EPSG:32631", (1, 2))
    assert np.isclose(result["valid_fraction"][0, 0], .9)
    assert np.isclose(result["lst_c"][0, 0], scale_temperature(np.array([45000]))[0])
    assert np.isnan(result["lst_c"][0, 1])
    assert np.isclose(result["snow_fraction"][0, 0], 1.0)


def test_spatial_patches_do_not_overlap_or_escape_region():
    origins = patch_origins({"grid_shape": [500, 500]}, 24, 8, np.random.default_rng(42))
    occupied = set()
    for r, c in origins:
        assert 0 <= r <= 492 and 0 <= c <= 492
        cells = {(rr, cc) for rr in range(r, r + 8) for cc in range(c, c + 8)}
        assert not (occupied & cells)
        occupied.update(cells)


def test_signed_url_queries_are_removed_from_library_log_text():
    message = "Skipping source 'https://example.invalid/data.tif?st=example&sig=test-signature'"
    safe = _redact_url_queries(message)
    assert "sig=" not in safe and "test-signature" not in safe
    assert "https://example.invalid/data.tif?[redacted]" in safe
    assert _redact_url_queries(safe) == safe
