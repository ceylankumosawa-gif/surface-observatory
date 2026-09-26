"""Actual pinned-model parity and native-call boundary checks (Hetzner)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lst_global.compiled import NativeF, LIBRARY_PATH
from lst_global.model import FrozenF, FEATURES, MODEL_PATH, MODEL_SHA256

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists() or not LIBRARY_PATH.exists(),
                                reason='Pinned server-only model/native artifact required')


@pytest.fixture(scope='module')
def engines():
    return FrozenF.load(), FrozenF.load(backend='native')


@pytest.fixture(scope='module')
def features():
    path = MODEL_PATH.parent.parent / 'fit_frame.parquet'
    data = pd.read_parquet(path, columns=list(FEATURES))
    # Both every trained climate and a geographically varied systematic sample.
    return pd.concat([data.groupby('climate_class', observed=True).head(10), data.iloc[::17]]).drop_duplicates()


def test_native_values_and_hash_are_identical_across_chunk_boundaries(engines, features):
    reference, native = engines
    a = reference.predict(features, source_provenance={'test': True}, chunk_size=317)
    b = native.predict(features, source_provenance={'test': True}, chunk_size=503)
    pd.testing.assert_frame_equal(a.values, b.values, check_exact=True)
    assert a.provenance['input_feature_and_index_sha256'] == b.provenance['input_feature_and_index_sha256']
    assert b.provenance['inference_backend']['name'] == 'native-c-exact-f-v1'
    assert b.provenance['model_sha256'] == MODEL_SHA256


def test_empty_native_batch_preserves_adapter_behavior(engines, features):
    result = engines[1].predict(features.iloc[:0], source_provenance={'test': True})
    assert result.values.empty and result.provenance['input_rows'] == 0


def test_hash_mismatch_stops_before_loading_code(engines, tmp_path):
    path = tmp_path / 'incorrect.so'
    path.write_bytes(b'not executable code')
    with pytest.raises(ValueError, match='differs from the verified artifact'):
        NativeF(engines[0]._estimator, model_sha256=MODEL_SHA256, library_path=path)


def test_model_identity_is_required_before_native_load(engines):
    with pytest.raises(ValueError, match='exact fitted model identity'):
        NativeF(engines[0]._estimator, model_sha256='0' * 64)


@pytest.mark.parametrize('invalid', ['unknown_climate', 'nan', 'infinity'])
def test_invalid_public_features_never_reach_native_code(engines, features, invalid):
    data = features.iloc[:2].copy()
    if invalid == 'unknown_climate':
        data.loc[data.index[0], 'climate_class'] = 'unknown'
    else:
        data.loc[data.index[0], 'ndvi'] = np.nan if invalid == 'nan' else np.inf
    with pytest.raises(ValueError):
        engines[1].predict(data, source_provenance={'test': True})


@pytest.mark.parametrize('invalid', ['shape', 'nan', 'climate'])
def test_native_memory_boundary_checks_preprocessed_arrays(engines, features, monkeypatch, invalid):
    native = engines[1]._native
    data = features.iloc[:2]
    array = np.zeros((2, 40))
    if invalid == 'shape':
        array = array[:, :39]
    elif invalid == 'nan':
        array[0, 2] = np.nan
    else:
        array[0, 0] = 13
    monkeypatch.setattr(native.regressor, '_preprocess_X', lambda *a, **k: array)
    with pytest.raises(ValueError):
        native.predict(data)


def test_backend_selection_is_explicit():
    with pytest.raises(ValueError, match='backend'):
        FrozenF.load(backend='something-else')
