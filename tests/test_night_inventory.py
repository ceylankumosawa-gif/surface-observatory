"""Metadata-only checks: no remote API, label reads or authentication."""
from datetime import datetime, timezone

import pytest

from lst_pilot.night_inventory import (
    MetadataClient, PRODUCTS, PreflightError, calendar_candidates, footprint_qa,
    inventory_query, name_identity, normalize, pilot_blocks, split_for_date, summarize,
)


def london():
    return {"type": "Feature", "properties": {
        "id": "greater_london", "center": [-.09, 51.49], "epsg": 27700,
        "extent_m": [492800, 138400, 572800, 218400]}, "geometry": {
        "type": "Polygon", "coordinates": [[[-.7, 51.1], [.5, 51.1], [.5, 51.9], [-.7, 51.9], [-.7, 51.1]]]}}


def eco_entry(tile="30UXC", scene="008", time="20231231T030706", concept="G1-LPCLOUD"):
    stamp = datetime.strptime(time, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return {"id": concept, "title": f"ECOv002_L2T_LSTE_42449_{scene}_{tile}_{time}_0713_01",
            "time_start": stamp.isoformat(), "day_night_flag": "NIGHT",
            "polygons": [["51.0 -1.0 51.0 0.8 52.0 0.8 52.0 -1.0 51.0 -1.0"]]}


def test_split_preserves_2024_and_2025():
    assert split_for_date("2022-12-31") == "fit"
    assert split_for_date("2023-06-30") == "development"
    assert split_for_date("2023-07-01") == "calibration"
    assert split_for_date("2024-02-29") == "reserved_legacy_2024_test"
    assert split_for_date("2025-01-01") == "reserved_blind_2025_test"
    assert split_for_date("2026-01-01") == "outside_frozen_protocol"


def test_orbit_groups_remove_tile_and_adjacent_scene_replication():
    first = normalize(eco_entry(), london(), "ecostress_v2")
    second = normalize(eco_entry(tile="30UYC", scene="009", time="20231231T030759", concept="G2-LPCLOUD"), london(), "ecostress_v2")
    result = summarize([first, second])
    assert result["unique_granule_ids"] == 2
    assert result["unique_acquisition_groups"] == 1
    assert result["unique_utc_dates"] == 1
    assert result["usable_thermal_acquisitions"] is None
    assert len(calendar_candidates([first, second])) == 1


def test_wrong_tile_and_antimeridian_metadata_are_quarantined():
    record = normalize(eco_entry(tile="60KYV"), london(), "ecostress_v2")
    assert record["footprint_qa"]["status"] == "review_required"
    assert any("MGRS" in issue for issue in record["footprint_qa"]["issues"])
    entry = eco_entry()
    entry["polygons"] = [["51 -179 51 179 52 179 52 -179 51 -179"]]
    assert normalize(entry, london(), "ecostress_v2")["footprint_qa"]["status"] == "review_required"


def test_name_and_timestamp_must_match_version():
    entry = eco_entry()
    entry["time_start"] = "2023-12-31T04:00:00Z"
    assert normalize(entry, london(), "ecostress_v2")["footprint_qa"]["status"] == "review_required"
    with pytest.raises(PreflightError, match="version"):
        name_identity(eco_entry()["title"], PRODUCTS["ecostress_v3"], datetime(2023, 12, 31, 3, 7, 6, tzinfo=timezone.utc))


def test_aster_processing_revisions_have_one_timestamp_group():
    stamp = datetime(2024, 8, 1, 20, 55, 27, tzinfo=timezone.utc)
    first = name_identity("AST_08_00408012024205527_20251114030125", PRODUCTS["aster_v4"], stamp)
    second = name_identity("AST_08_00408012024205527_20261114030125", PRODUCTS["aster_v4"], stamp)
    assert first["acquisition_group"] == second["acquisition_group"]
    assert first["orbit_available"] is False


def test_cross_split_overpass_cannot_enter_fit():
    rows = [normalize(eco_entry(time="20221231T235930"), london(), "ecostress_v2"),
            normalize(eco_entry(time="20230101T000030", concept="G2-LPCLOUD"), london(), "ecostress_v2")]
    candidates = calendar_candidates(rows)
    assert len(candidates) == 1
    assert candidates[0]["temporal_split"] == "cross_split_boundary_reserved"


def test_calendar_selection_is_deterministic_and_not_label_driven():
    rows = []
    for i in range(6):
        entry = eco_entry(concept=f"G{i}-LPCLOUD")
        entry["title"] = entry["title"].replace("42449", str(42449 + i))
        rows.append(normalize(entry, london(), "ecostress_v2"))
    first = calendar_candidates(rows, 2)
    assert first == calendar_candidates(list(reversed(rows)), 2)
    assert sum(item["initial_candidate"] for item in first) == 2


def test_spatial_blocks_are_fixed_and_have_buffer():
    blocks = pilot_blocks(london())
    assert len(blocks) == 64
    assert 12 <= sum(block["spatial_holdout"] for block in blocks) <= 14
    assert all(block["holdout_buffer_m"] == 1000 for block in blocks)
    assert blocks[0]["bounds_m"] == [492800, 138400, 502800, 148400]


def test_page_cap_is_explicitly_partial():
    class FakeClient:
        def get(self, endpoint, params):
            return {"feed": {"entry": [eco_entry()]}}, 10
    records, audit = inventory_query(FakeClient(), london(), "ecostress_v2", "NIGHT", "2023-01-01", "2023-12-31", 1, 1)
    assert len(records) == 1
    assert audit["complete"] is False
    assert audit["cmr_hits"] == 10
    assert "lower bounds" in audit["errors"][0]


def test_metadata_client_never_uses_account_or_asset_endpoint():
    client = MetadataClient()
    assert client.session.trust_env is False
    assert client.session.auth is None
    assert "Authorization" not in client.session.headers
    with pytest.raises(PreflightError, match="Only public CMR"):
        client.get("https://data.lpdaac.earthdatacloud.nasa.gov/asset.tif", {})
