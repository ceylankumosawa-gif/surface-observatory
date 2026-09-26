"""Inference-only tests; never fit an estimator or decode a thermal target."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from lst_global import model as m


@pytest.fixture(scope="module")
def frozen():
    if not m.MODEL_PATH.exists():
        pytest.skip("Pinned research model lives on Hetzner; execute this suite there.")
    return m.FrozenF.load()


@pytest.fixture
def frame():
    data = pd.DataFrame(np.zeros((9, len(m.NUMERIC_FEATURES))), columns=m.NUMERIC_FEATURES)
    data["climate_class"] = ["Cfb", "Dfa", "Af"] * 3
    data["air_temperature_c"] = [-12., 0., 7., 2., -1., 33., 0., 15., -7.]
    data.index = pd.Index([91, 12, 12, 4, 72, 1, 60, 88, 8], name="pixel_identity")
    return data


def test_bad_hash_precedes_deserialization(tmp_path, monkeypatch):
    bad = tmp_path / "untrusted.joblib"
    bad.write_bytes(b"not an authorized estimator")
    monkeypatch.setattr(m.joblib, "load", lambda _: pytest.fail("Must not deserialize an untrusted model"))
    with pytest.raises(ValueError, match="SHA-256"):
        m.FrozenF.load(bad)


def test_pinned_schema_and_estimator(frozen):
    assert len(m.FEATURES) == 40
    assert tuple(frozen._estimator.feature_names_in_) == m.FEATURES
    assert len(m.CLIMATE_CLASSES) == 13
    assert hashlib.sha256(m.MODEL_PATH.read_bytes()).hexdigest() == m.MODEL_SHA256


def test_chunks_are_exact_with_zero_negative_air_and_duplicate_index(frozen, frame):
    before = frame.copy(deep=True)
    expected_offset = frozen._estimator.predict(frame)
    results = [frozen.predict(frame, chunk_size=n, source_provenance={"prepared_features_sha256": "example"})
               for n in (1, 4, 9, 32)]
    for result in results:
        np.testing.assert_array_equal(result.values.surface_air_offset_c, expected_offset)
        np.testing.assert_array_equal(result.values.predicted_lst_c, frame.air_temperature_c.to_numpy() + expected_offset)
        pd.testing.assert_index_equal(result.values.index, frame.index, exact=True)
        assert result.values.support_status.eq("global_extrapolation").all()
        assert result.provenance["uncertainty_available"] is False
        assert result.provenance["global_accuracy_validated"] is False
        assert tuple(result.values.columns) == ("predicted_lst_c", "surface_air_offset_c", "support_status")
    assert len({r.provenance["input_feature_and_index_sha256"] for r in results}) == 1
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


@pytest.mark.parametrize("change", ["reorder", "missing", "extra", "duplicate"])
def test_strict_schema(frozen, frame, change):
    if change == "reorder":
        frame = frame[list(reversed(m.FEATURES))]
    elif change == "missing":
        frame = frame.drop(columns="ndvi")
    elif change == "extra":
        frame["lst_c"] = 999  # Label columns are forbidden at this interface.
    else:
        frame.columns = ["air_temperature_c", *m.FEATURES[2:], "climate_class"]
    with pytest.raises(ValueError, match="exact 40"):
        frozen.predict(frame, source_provenance={"source": "test"})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected_before_any_predict(frozen, frame, monkeypatch, bad):
    frame.iloc[-1, 0] = bad
    monkeypatch.setattr(frozen._estimator, "predict", lambda _: pytest.fail("Validate all inputs before prediction"))
    with pytest.raises(ValueError, match="finite"):
        frozen.predict(frame, chunk_size=1, source_provenance={"source": "test"})


@pytest.mark.parametrize("bad", ["", "Cfb ", "__unknown__", "new_climate", None, float("nan")])
def test_unknown_climate_is_not_imputed(frozen, frame, bad):
    frame.iloc[0, -1] = bad
    with pytest.raises(ValueError, match="climate_class"):
        frozen.predict(frame, source_provenance={"source": "test"})


@pytest.mark.parametrize("bad", ["0", True, 1 + 2j])
def test_no_implicit_numeric_conversion(frame, bad):
    frame["ndvi"] = bad
    with pytest.raises(ValueError, match="real numeric"):
        m.validate_features(frame)


@pytest.mark.parametrize("chunk", [0, -1, 1.5, True, 1_048_577])
def test_chunk_bound(frozen, frame, chunk):
    with pytest.raises(ValueError, match="chunk_size"):
        frozen.predict(frame, source_provenance={"source": "test"}, chunk_size=chunk)


def test_empty_and_provenance_copy(frozen, frame):
    source = {"source": {"path": "prepared.parquet", "air_basis": "station_adjusted"}}
    result = frozen.predict(frame.iloc[:0], source_provenance=source)
    assert result.values.empty
    assert result.provenance["input_rows"] == 0
    source["source"]["path"] = "changed"
    assert result.provenance["source"]["source"]["path"] == "prepared.parquet"
    with pytest.raises(ValueError, match="source_provenance"):
        frozen.predict(frame, source_provenance={})
    with pytest.raises(ValueError):
        frozen.predict(frame, source_provenance={"bad": float("nan")})


def test_bad_model_outputs_fail_closed(frozen, frame, monkeypatch):
    monkeypatch.setattr(frozen._estimator, "predict", lambda x: np.full(len(x), np.nan))
    with pytest.raises(ValueError, match="invalid offsets"):
        frozen.predict(frame, source_provenance={"source": "test"})
    monkeypatch.setattr(frozen._estimator, "predict", lambda x: np.zeros(len(x) + 1))
    with pytest.raises(ValueError, match="invalid offsets"):
        frozen.predict(frame, source_provenance={"source": "test"})
    frame["air_temperature_c"] = np.finfo(float).max
    monkeypatch.setattr(frozen._estimator, "predict", lambda x: np.full(len(x), np.finfo(float).max))
    with pytest.raises(ValueError, match="Air plus offset"):
        frozen.predict(frame, source_provenance={"source": "test"})
