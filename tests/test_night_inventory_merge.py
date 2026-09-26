"""Offline merge integrity and coverage checks; no label or network access."""
import json

import pytest

from lst_pilot.night_inventory import PRODUCTS, PROTOCOL
from lst_pilot.night_inventory_merge import build_merge, coverage_gaps
from lst_pilot.night_metadata_campaign import MAX_BYTES, MAX_REQUESTS, MAX_SECONDS, campaign_queries


def shard(path, first, last, *, complete=True, identifier="G1-LPCLOUD", when=None, source_hash="baseline"):
    path.mkdir()
    when = when or first
    row = {"pilot_id": "sioux_falls", "product": "ecostress_v2", "collection": PRODUCTS["ecostress_v2"],
           "granule_concept_id": identifier, "granule_title": identifier,
           "time_start": when + "T04:00:00+00:00", "utc_date": when, "day_night_flag": "NIGHT",
           "local_solar_hour_bin": 3, "temporal_split": "fit", "identity": {"acquisition_group": "ECOSTRESS:orbit:1"},
           "footprint_qa": {"status": "metadata_consistent_pixels_unverified", "issues": []}}
    plan = {"protocol": PROTOCOL, "metadata_only": True, "credentials_used": False, "calendar_candidates_per_stratum": 2,
            "protected_source_hashes": [{"path": "model.joblib", "present": True, "sha256": source_hash}], "blocks": []}
    report = {"protocol": PROTOCOL, "protected_pixels_downloaded": 0, "status": "complete_metadata_only",
              "collection_verification": {"ecostress_v2": {**PRODUCTS["ecostress_v2"], "verified": True}},
              "queries": [{"pilot_id": "sioux_falls", "product": "ecostress_v2", "day_night_flag": "NIGHT",
                           "complete": complete, "errors": [] if complete else ["page cap"], "catalog_records_loaded": 1,
                           "params": {"temporal": f"{first}T00:00:00Z,{last}T23:59:59Z", "collection_concept_id": PRODUCTS["ecostress_v2"]["concept_id"]}}]}
    for name, value in [("plan.json", plan), ("inventory.json", report), ("granules.json", [row]), ("calendar_candidates.json", {})]:
        (path / name).write_text(json.dumps(value))
    return path


def test_adjacent_calendar_shards_cover_year_and_missing_day_remains_gap():
    assert coverage_gaps([("2021-01-01", "2021-06-30"), ("2021-07-01", "2021-12-31")], "2021-01-01", "2021-12-31") == []
    assert coverage_gaps([("2021-01-01", "2021-06-29"), ("2021-07-01", "2021-12-31")], "2021-01-01", "2021-12-31") == [["2021-06-30", "2021-06-30"]]
    assert coverage_gaps([("2020-01-01", "2022-12-31")], "2021-01-01", "2021-12-31") == []


def test_partial_source_does_not_supply_coverage_or_candidates(tmp_path):
    source = shard(tmp_path / "partial", "2021-01-01", "2021-12-31", complete=False)
    rows, candidates, report, _ = build_merge([source], ["sioux_falls"], ["ecostress_v2"], "2021-01-01", "2021-12-31")
    assert rows == [] and candidates["acquisitions"] == []
    assert report["coverage"][0]["gaps"] == [["2021-01-01", "2021-12-31"]]


def test_exact_duplicate_is_counted_once_and_source_hashes_are_recorded(tmp_path):
    sources = [shard(tmp_path / name, "2021-01-01", "2021-12-31") for name in ("one", "two")]
    rows, candidates, report, _ = build_merge(sources, ["sioux_falls"], ["ecostress_v2"], "2021-01-01", "2021-12-31")
    assert len(rows) == len(candidates["acquisitions"]) == 1
    assert report["duplicates_removed"] == 1
    assert report["status"] == "complete_metadata_only"
    assert all(len(source["hashes"]["granules.json"]) == 64 for source in report["source_registry"])
    assert candidates["frozen_for_label_acquisition"] is False


def test_conflicting_record_is_quarantined_and_baseline_change_is_visible(tmp_path):
    first = shard(tmp_path / "one", "2021-01-01", "2021-12-31")
    second = shard(tmp_path / "two", "2021-01-01", "2021-12-31", source_hash="changed")
    rows = json.loads((second / "granules.json").read_text())
    rows[0]["granule_title"] = "different"
    (second / "granules.json").write_text(json.dumps(rows))
    merged, candidates, report, _ = build_merge([first, second], ["sioux_falls"], ["ecostress_v2"], "2021-01-01", "2021-12-31")
    assert report["record_conflicts_quarantined"] == 1
    assert report["protected_source_conflicts"]
    assert candidates["acquisitions"] == []


def test_incompatible_protocol_is_rejected(tmp_path):
    source = shard(tmp_path / "one", "2021-01-01", "2021-12-31")
    plan = json.loads((source / "plan.json").read_text())
    plan["protocol"] = "changed"
    (source / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Incompatible protocol"):
        build_merge([source], ["sioux_falls"], ["ecostress_v2"], "2021-01-01", "2021-12-31")


def test_campaign_respects_fixed_scope_and_reuses_prior_completed_2023_queries():
    assert MAX_REQUESTS == 400 and MAX_BYTES == 256 * 1024 * 1024 and MAX_SECONDS == 1200
    queries = campaign_queries()
    assert len(queries) == 24
    assert queries[0] == ("greater_london", "ecostress_v2", "2023-01-01", "2023-03-31")
    assert all(last <= "2023-12-31" for _, _, first, last in queries)
    assert not any(pilot == "sioux_falls" and first.startswith("2023") for pilot, _, first, _ in queries)
