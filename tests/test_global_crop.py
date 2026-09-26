from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import rasterio

from lst_global.crop import crop_tile
from lst_global import patch
from lst_global.planner import sha
from lst_global.model import MODEL_SHA256


@pytest.mark.parametrize('row,column,cells', [(-1, 0, 32), (0, 0, 0), (500, 0, 32), (0, 500, 32), (True, 0, 32)])
def test_invalid_crop_fails_before_opening_inputs(tmp_path, row, column, cells):
    with pytest.raises(ValueError):
        crop_tile(tmp_path / 'absent', tmp_path / 'out', row, column, cells)


def test_public_preparation_cannot_reselect_sources_for_a_partial_area(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Partial request reached preparation')
    monkeypatch.setattr(patch, '_run_bounded', forbidden)
    with pytest.raises(ValueError, match='full 512-cell'):
        patch.run(tmp_path / 'out', -1.25, 51.75, '2023-06-21T00:00Z', cells=32)


PARENT = Path('/opt/lst-pilot/runs/global_v1_20260915/oxford_full_tile_v2')


@pytest.mark.skipif(not PARENT.exists(), reason='Remote completed engineering tile fixture only')
def test_actual_overlapping_crops_are_bit_exact_and_do_not_run_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Cropping must not run the estimator')
    monkeypatch.setattr(patch.FrozenF, 'load', forbidden)
    before = sha(PARENT / 'provenance.json')
    large = crop_tile(PARENT, tmp_path / 'large', 480, 48, 32)
    small = crop_tile(PARENT, tmp_path / 'small', 480, 52, 16)
    with rasterio.open(tmp_path / 'large' / 'lst.tif') as a, rasterio.open(tmp_path / 'small' / 'lst.tif') as b:
        np.testing.assert_array_equal(a.read(1)[:16, 4:20], b.read(1))
    assert large['crop_network_requests'] == small['crop_model_predictions'] == 0
    assert sha(PARENT / 'provenance.json') == before


@pytest.mark.skipif(not PARENT.exists(), reason='Remote completed engineering tile fixture only')
def test_all_masked_parent_can_be_cropped_without_an_inference_record(tmp_path):
    parent = tmp_path / 'all_masked_fixture'
    parent.mkdir()
    for name in ('inputs.parquet', 'features.parquet'):
        shutil.copyfile(PARENT / name, parent / name)  # Unchanged fixture columns; /tmp may be another filesystem.
    proof = json.loads((PARENT / 'provenance.json').read_text())
    proof.pop('inference')
    proof['model_sha256'] = MODEL_SHA256
    proof['predicted_pixels'] = 0
    for name, value in (('lst.tif', -9999), ('support.tif', 8)):
        with rasterio.open(PARENT / name) as original:
            with rasterio.open(parent / name, 'w', **original.profile) as dest:
                dest.write(np.full((512, 512), value, dtype=original.dtypes[0]), 1)
    pixels = pd.read_parquet(PARENT / 'pixels.parquet')
    pixels['predicted_lst_c'] = np.nan
    pixels['support_code'] = np.uint8(8)
    pixels.to_parquet(parent / 'pixels.parquet', index=False)
    for name in proof['artifacts']:
        proof['artifacts'][name] = {'sha256': sha(parent / name), 'bytes': (parent / name).stat().st_size}
    (parent / 'provenance.json').write_text(json.dumps(proof))
    result = crop_tile(parent, tmp_path / 'out', 480, 48, 32)
    assert result['predicted_pixels'] == 0
    assert result['support_counts']['no_verified_recent_station'] == 1024
    assert result['crop_model_predictions'] == 0
