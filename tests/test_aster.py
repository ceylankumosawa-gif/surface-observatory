import numpy as np
import pytest
from affine import Affine

from lst_pilot import aster


def raw():
    return {'SKT': np.full((30, 30), 3000, dtype='uint16'),
            'SKT_QA_DataPlane': np.zeros((30, 30), dtype='uint8'),
            'SKT_QA_DataPlane2': np.zeros((30, 30), dtype='uint8')}


def test_native_zero_qa_is_good_kelvin_scale_and_edges():
    valid, kelvin, _ = aster.native_valid(raw(), Affine(90, 0, 0, 0, -90, 2700))
    assert kelvin[15, 15] == 300
    assert valid[15, 15]
    assert not valid[0, 0]


def test_cloud_and_fill_are_buffered_without_value_selection():
    source = raw()
    source['SKT_QA_DataPlane'][15, 15] = 4
    valid, _, _ = aster.native_valid(source, Affine(90, 0, 0, 0, -90, 2700))
    assert not valid[15, 18]
    assert valid[15, 20]
    source['SKT'][15, 15] = 0
    source['SKT_QA_DataPlane'][15, 15] = 0
    valid2, _, _ = aster.native_valid(source, Affine(90, 0, 0, 0, -90, 2700))
    assert np.array_equal(valid, valid2)


def test_rotation_preserves_90m_spacing():
    transform = Affine.translation(500000, 6000000) * Affine.rotation(12) * Affine.scale(90, -90)
    valid, _, _ = aster.native_valid(raw(), transform)
    assert valid[15, 15]
    with pytest.raises(aster.eco.AcquisitionError, match='orthogonal'):
        aster.native_valid(raw(), Affine(90, 20, 0, 0, -90, 0))


def test_aggregation_masks_partial_native_support():
    values, _, _ = aster.aggregate(raw(), Affine(90, 0, 500000, 0, -90, 6000000),
                                    'EPSG:32630', [499900, 5997300, 502700, 6000100], 32630)
    assert np.isnan(values['lst_c'][0]).all()
    assert np.isnan(values['lst_c'][:, 0]).all()
    assert np.isclose(values['lst_c'][14, 14], 26.85, atol=1e-4)


def test_identity_and_unsupported_dtype():
    ident = aster.identity('AST_08_00406122021213543_20250915063545')
    assert ident['acquisition_utc'] == '2021-06-12T21:35:43+00:00'
    with pytest.raises(aster.eco.AcquisitionError):
        aster.identity('AST_08_00306122021213543_20250915063545')
    values = raw()
    values['SKT'] = values['SKT'].astype('float32')
    with pytest.raises(aster.eco.AcquisitionError, match='types'):
        aster.native_valid(values, Affine(90, 0, 0, 0, -90, 0))


def test_single_and_range_metadata_preserve_fractional_seconds():
    assert str(aster.acquisition_time({'TemporalExtent': {'SingleDateTime': '2021-06-12T21:35:43.96200Z'}})) == '2021-06-12 21:35:43.962000+00:00'
    assert str(aster.acquisition_time({'TemporalExtent': {'RangeDateTime': {'BeginningDateTime': '2021-06-12T21:35:43.96200Z'}}})) == '2021-06-12 21:35:43.962000+00:00'
