"""Metadata QA tests use synthetic XML/CMR and a verified public-list excerpt."""
import base64
import copy
import hashlib
import json

import pytest

from lst_pilot import ecostress_geo as geo


TARGET = "ECOv002_L2T_LSTE_00486_001_30UXC_20180807T012533_0702_01"
GEO = "ECOv002_L1B_GEO_00486_001_20180807T012533_0713_01"
PUBLIC_LINE = 'ORB=00486 SCN=001 t1=2018-08-07T01:25:33.512065 t2=2018-08-07T01:26:24.852084 FOV_OBST=YES GeolocationAccuracyQA="Best"\n'


def candidate(name=GEO, revision=1, concept="G123-LPCLOUD"):
    url = "https://" + geo.GEO_HOST + geo.GEO_PREFIX + name + "/" + name + ".h5"
    return {"meta": {"concept-id": concept, "revision-id": revision, "collection-concept-id": geo.GEO_COLLECTION,
                     "revision-date": "2026-09-09T00:00:00Z"},
            "umm": {"GranuleUR": name, "CollectionReference": {"ShortName": "ECO_L1B_GEO", "Version": "002"},
                    "TemporalExtent": {"RangeDateTime": {"BeginningDateTime": "2018-08-07T01:25:33.512Z"}},
                    "RelatedUrls": [{"Type": "GET DATA", "URL": url}]}}


def xml(label="Best", name=GEO, content=None):
    value = base64.b64encode(label.encode()).decode() if content is None else content
    return f'''<?xml version="1.0" encoding="UTF-8"?>
    <Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" xmlns:dmrpp="http://xml.opendap.org/dmrpp/1.0.0#" name="{name}.h5">
      <Group name="L1GEOMetadata"><String name="GeolocationAccuracyQA"><dmrpp:compact>{value}</dmrpp:compact></String></Group>
    </Dataset>'''.encode()


def test_discovery_is_pinned_to_exact_orbit_scene_acquisition_and_geo_collection():
    params = geo.discovery_params(TARGET)
    assert params["collection_concept_id"] == geo.GEO_COLLECTION
    assert params["readable_granule_name"] == "ECOv002_L1B_GEO_00486_001_20180807T012533_*"
    assert params["version"] == "002"
    assert geo.parse_identity(GEO + ".h5.dmrpp")["granule_name"] == GEO
    with pytest.raises(geo.GeoMetadataError):
        geo.discovery_params(TARGET.replace("ECOv002", "ECOv003"))


def test_selects_latest_processing_then_cmr_revision_without_using_qa():
    old = candidate(GEO.replace("0713_01", "0712_99"), revision=99, concept="G124-LPCLOUD")
    latest = candidate(revision=3)
    result = geo.select_geo_candidate([latest, old, candidate(revision=1)], TARGET)
    assert result["status"] == "Matched"
    assert result["cmr_revision_id"] == 3
    assert result["identity"]["build"] == "0713"
    assert result["dmrpp_url"] == result["h5_url"] + ".dmrpp"
    assert result["accepted"] is False and result["qa_not_read"] is True


@pytest.mark.parametrize("fragment,replacement", [("00486", "00487"), ("_001_", "_002_"), ("20180807T012533", "20180807T012534")])
def test_lookup_does_not_match_timestamp_alone_or_neighboring_orbits(fragment, replacement):
    item = candidate(GEO.replace(fragment, replacement))
    assert geo.select_geo_candidate([item], TARGET)["status"] == "Unknown"


@pytest.mark.parametrize("fault", ["time", "collection", "signed_url", "wrong_host", "duplicate_concept", "missing_meta"])
def test_latest_invalid_revision_is_not_replaced_by_older_usable_metadata(fault):
    latest = candidate(revision=2)
    records = [candidate(revision=1), latest]
    if fault == "time":
        latest["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"] = "2018-08-07T01:26:00Z"
    elif fault == "collection":
        latest["umm"]["CollectionReference"]["Version"] = "003"
    elif fault == "signed_url":
        latest["umm"]["RelatedUrls"][0]["URL"] += "?token=not-real"
    elif fault == "wrong_host":
        latest["umm"]["RelatedUrls"][0]["URL"] = latest["umm"]["RelatedUrls"][0]["URL"].replace(geo.GEO_HOST, "example.com")
    elif fault == "duplicate_concept":
        latest["meta"]["concept-id"] = "G999-LPCLOUD"
    else:
        del latest["meta"]["revision-id"]
    assert geo.select_geo_candidate(records, TARGET)["status"] == "Unknown"


@pytest.mark.parametrize("label,accepted,status", [("Best", True, "Accepted"), ("good\x00", True, "Accepted"), ("Suspect", False, "Rejected"), ("Poor", False, "Rejected"), ("NotFound", False, "Unknown"), ("Best or Good", False, "Unknown")])
def test_only_positive_exact_best_good_qa_is_accepted(label, accepted, status):
    payload = xml(label)
    result = geo.parse_geolocation_dmrpp(payload, GEO)
    assert result["accepted"] is accepted
    assert result["status"] == status
    assert result["source_sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("fault", ["explanation_only", "wrong_group", "other_dataset", "invalid_base64", "malformed_xml", "duplicate_scalar", "external_entity", "array"])
def test_unverifiable_or_ambiguous_xml_is_unknown(fault):
    payload = xml()
    if fault == "explanation_only":
        payload = payload.replace(b'"GeolocationAccuracyQA"', b'"GeolocationAccuracyQAExplanation"')
    elif fault == "wrong_group":
        payload = payload.replace(b'"L1GEOMetadata"', b'"StandardMetadata"')
    elif fault == "other_dataset":
        payload = xml(name=GEO.replace("_001_", "_002_"))
    elif fault == "invalid_base64":
        payload = xml(content="not base64@@")
    elif fault == "malformed_xml":
        payload = payload[:70]
    elif fault == "duplicate_scalar":
        payload = payload.replace(b'</Group>', b'<String name="GeolocationAccuracyQA"><v>R29vZA==</v></String></Group>')
    elif fault == "array":
        payload = payload.replace(b'<dmrpp:compact>', b'<Dim size="2"/><dmrpp:compact>')
    else:
        payload = b'<!DOCTYPE Dataset [<!ENTITY x SYSTEM "file:///never-read">]>' + payload
    result = geo.parse_geolocation_dmrpp(payload, GEO)
    assert result["status"] == "Unknown" and result["accepted"] is False


def test_official_obstruction_format_matches_orbit_scene_time_and_preserves_source_sha():
    index = geo.parse_obstruction_list(PUBLIC_LINE.encode(), complete=True)
    result = geo.obstruction_status(index, TARGET)
    assert index["status"] == "Parsed"
    assert result["status"] == "Listed" and result["obstructed"] is True
    assert result["source_sha256"] == hashlib.sha256(PUBLIC_LINE.encode()).hexdigest()
    assert result["record"]["listed_geolocation_qa"] == "Best"  # Does not negate obstruction.
    different = TARGET.replace("20180807T012533", "20180807T012543")
    assert geo.obstruction_status(index, different)["status"] == "Unknown"


def test_partial_missing_malformed_and_conflicting_lists_never_imply_clear():
    for index in [geo.parse_obstruction_list(PUBLIC_LINE.encode()), geo.parse_obstruction_list(b"", complete=True),
                  geo.parse_obstruction_list((PUBLIC_LINE + "not a record\n").encode(), complete=True),
                  geo.parse_obstruction_list((PUBLIC_LINE + PUBLIC_LINE.replace("FOV_OBST=YES", "FOV_OBST=NO")).encode(), complete=True)]:
        assert geo.obstruction_status(index, TARGET)["status"] == "Unknown"
    complete = geo.parse_obstruction_list(PUBLIC_LINE.encode(), complete=True)
    result = geo.obstruction_status(complete, TARGET.replace("00486", "99999"))
    assert result["status"] == "NotListed"
    assert result["obstructed"] is None and result["proves_unobstructed"] is False


def test_full_official_list_variants_optional_qa_long_intervals_and_reprocessing():
    no_qa = PUBLIC_LINE.split(' GeolocationAccuracyQA=')[0] + "\n"
    long_interval = PUBLIC_LINE.replace("01:26:24.852084", "02:26:24.852084")
    subsecond_revision = PUBLIC_LINE.replace("01:25:33.512065", "01:25:33.912065")
    index = geo.parse_obstruction_list((no_qa + long_interval + subsecond_revision).encode(), complete=True)
    assert index["status"] == "Parsed" and not index["errors"]
    result = geo.obstruction_status(index, TARGET)
    assert result["status"] == "Listed" and result["obstructed"] is True
    assert len(result["matching_variants"]) == 3
    assert result["matching_variants"][0]["listed_geolocation_qa"] == "Unknown"


def test_obstruction_source_cache_is_hash_addressed(tmp_path):
    payload = PUBLIC_LINE.encode()
    result = geo.cache_obstruction_list(payload, tmp_path, complete=True)
    assert (tmp_path / result["cache_file"]).read_bytes() == payload
    assert result == geo.cache_obstruction_list(payload, tmp_path, complete=True)
    (tmp_path / result["cache_file"]).write_bytes(b"modified")
    with pytest.raises(geo.GeoMetadataError, match="hash changed"):
        geo.cache_obstruction_list(payload, tmp_path, complete=True)


def test_discovery_caches_frozen_complete_snapshot_with_hash(tmp_path, monkeypatch):
    calls = []
    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return json.dumps({"hits": 1, "items": [candidate()]}).encode(), {"CMR-Hits": "1"}
    monkeypatch.setattr(geo, "_anonymous_bytes", fetch)
    first = geo.discover_geo(TARGET, tmp_path)
    second = geo.discover_geo(TARGET, tmp_path)
    assert first == second and first["status"] == "Matched" and len(calls) == 1
    assert calls[0][0] == geo.CMR_URL
    assert "source_sha256" in first["discovery"]


def test_incomplete_cmr_snapshot_never_establishes_latest_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "_anonymous_bytes", lambda *args, **kwargs: (json.dumps({"hits": 101, "items": [candidate()]}).encode(), {}))
    assert geo.discover_geo(TARGET, tmp_path)["status"] == "Unknown"


def test_anonymous_downloader_refuses_any_protected_url_before_session_creation(monkeypatch):
    monkeypatch.setattr(geo.requests, "Session", lambda: pytest.fail("No session should be opened"))
    with pytest.raises(geo.GeoMetadataError, match="approved public"):
        geo._anonymous_bytes("https://" + geo.GEO_HOST + geo.GEO_PREFIX + "anything.h5.dmrpp", max_bytes=1024)
