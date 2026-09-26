"""Public ARCO cache sharing without changing general/credential permissions."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import pytest

from lst_pilot.radiation import _cached_object, _publish_public_cache


class PublicStore:
    def __init__(self, payload=b"public ERA5 fixture"):
        self.payload = payload
        self.reads = 0

    def info(self, key):
        return {"size": len(self.payload), "generation": "test-generation"}

    def cat_file(self, key):
        self.reads += 1
        return self.payload


def test_object_and_provenance_share_existing_group_under_restrictive_umask(tmp_path):
    cache=tmp_path/"public-cache"
    cache.mkdir(mode=0o775)
    expected_group=cache.stat().st_gid
    previous=os.umask(0o077)
    try:
        store=PublicStore(); manifest=[]
        assert _cached_object(store,"snow_depth/123.0.0",cache,100,manifest) == store.payload
    finally:
        os.umask(previous)
    path=cache/"snow_depth/123.0.0"
    for artifact in [path,path.with_name(path.name+".provenance.json")]:
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o664
        assert artifact.stat().st_gid == expected_group
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o2775
    assert path.parent.stat().st_gid == expected_group
    record=json.loads(path.with_name(path.name+".provenance.json").read_text())
    assert record["sha256"] == hashlib.sha256(store.payload).hexdigest()
    assert record["cached"] is False and manifest == [record]
    assert _cached_object(store,"snow_depth/123.0.0",cache,100,[]) == store.payload
    assert store.reads == 1


def test_daily_metadata_refresh_replaces_private_object_and_sidecar(tmp_path):
    metadata=tmp_path/".zmetadata"
    metadata.write_bytes(b"old metadata"); metadata.chmod(0o600)
    sidecar=tmp_path/".zmetadata.provenance.json"
    sidecar.write_text("old sidecar"); sidecar.chmod(0o600)
    expired=time.time()-90000
    os.utime(metadata,(expired,expired))
    store=PublicStore(b"new public metadata")
    _cached_object(store,".zmetadata",tmp_path,100,[])
    assert metadata.read_bytes() == store.payload and store.reads == 1
    assert json.loads(sidecar.read_text())["sha256"] == hashlib.sha256(store.payload).hexdigest()
    assert stat.S_IMODE(metadata.stat().st_mode) == stat.S_IMODE(sidecar.stat().st_mode) == 0o664
    assert metadata.stat().st_gid == sidecar.stat().st_gid == tmp_path.stat().st_gid


def test_permission_policy_stays_inside_public_cache_and_credentials_are_untouched(tmp_path):
    credentials=tmp_path/"private-token"
    credentials.write_bytes(b"synthetic private fixture"); credentials.chmod(0o600)
    cache=tmp_path/"public-cache"; cache.mkdir()
    before=credentials.stat()
    for key in ["../private-token",str(credentials)]:
        with pytest.raises(ValueError,match="relative"):
            _cached_object(PublicStore(),key,cache,100,[])
    (cache/"escape").symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match="inside"):
        _publish_public_cache(cache/"escape/private-token",b"public",cache)
    assert credentials.read_bytes() == b"synthetic private fixture"
    assert stat.S_IMODE(credentials.stat().st_mode) == 0o600
    assert credentials.stat().st_mtime_ns == before.st_mtime_ns


def test_failed_atomic_publication_removes_temporary_without_changing_existing_bytes(tmp_path,monkeypatch):
    path=tmp_path/".zmetadata"; path.write_bytes(b"prior public data")
    def fail(*args): raise OSError("simulated atomic replacement failure")
    monkeypatch.setattr("lst_pilot.radiation.os.replace",fail)
    with pytest.raises(OSError,match="replacement"):
        _publish_public_cache(path,b"replacement",tmp_path)
    assert path.read_bytes() == b"prior public data"
    assert list(tmp_path.iterdir()) == [path]
