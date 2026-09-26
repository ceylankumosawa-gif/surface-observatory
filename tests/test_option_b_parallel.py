from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pandas as pd
import pytest

from lst_pilot import option_b_features as features
from lst_pilot import option_b_parallel as parallel


def test_cross_worker_key_lock_prevents_duplicate_cache_population(tmp_path, monkeypatch):
    calls, stored = [], {}
    def object_reader(fs, key, cache, limit, manifest):
        if key not in stored:
            calls.append(key)
            time.sleep(.04)
            stored[key] = b"validated"
        return stored[key]
    originals = [(features, "_cached_tile"), (features, "search_optical"), (features.raster, "_stac_tiles"),
                 (features.weather, "fetch_archive"), (features.radiation, "_cached_object"), (features.assemble, "station_series")]
    # Register restoration of all process-local wrapper replacements.
    for module, name in originals:
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(features.radiation, "_cached_object", object_reader)
    parallel.install_cache_locks(tmp_path)
    def read(_):
        return features.radiation._cached_object(None, "same-hour", tmp_path, 100, [])
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(read, range(4))) == [b"validated"] * 4
    assert calls == ["same-hour"]


def test_distinct_keys_do_not_share_a_lock_file(tmp_path):
    with parallel.cache_lock(tmp_path, "radiation", ["hour-1"]):
        with parallel.cache_lock(tmp_path, "radiation", ["hour-2"]):
            assert len(list((tmp_path / "option-b-locks" / "radiation").glob("*.lock"))) == 2


def test_worker_bound_checked_before_any_input_reads(tmp_path):
    for count in (0, 5):
        with pytest.raises(ValueError, match="1–4"):
            parallel.run(tmp_path / "missing.parquet", tmp_path / "output", tmp_path / "areas", tmp_path / "cache", workers=count)
