"""Merge complete anonymous metadata shards without opening label assets.

Incomplete queries never establish coverage. Source hashes, geometry quarantine,
deduplication and exact calendar gaps are retained in an immutable registry.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

from .night_inventory import PRODUCTS, PROTOCOL, calendar_candidates, sha256_file, summarize, write_json


def coverage_gaps(intervals, start, end):
    """Inclusive UTC date intervals; adjacent shards cover their shared calendar."""
    first, last = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    cursor, gaps = first, []
    for left, right in sorted((date.fromisoformat(a), date.fromisoformat(b)) for a, b in intervals):
        left, right = max(first, left), min(last, right)
        if right < first or left > last or right < cursor:
            continue
        if left > cursor:
            gaps.append([str(cursor), str(left - timedelta(days=1))])
        cursor = max(cursor, right + timedelta(days=1))
    if cursor <= last:
        gaps.append([str(cursor), str(last)])
    return gaps


def discover_shards(paths):
    result = []
    for value in paths:
        path = Path(value).resolve()
        if (path / "inventory.json").is_file():
            result.append(path)
        else:
            result.extend(p.parent for p in sorted(path.glob("*/inventory.json")))
    return sorted(set(result))


def build_merge(shards, pilots, products, start, end):
    registry, source_queries, issues = [], [], []
    coverage = defaultdict(list)
    rows_by_key, duplicate_count, conflicting = {}, 0, set()
    protected_hashes, blocks_by_id = {}, {}
    protected_conflicts = []
    for directory in shards:
        plan = json.loads((directory / "plan.json").read_text())
        report = json.loads((directory / "inventory.json").read_text())
        rows = json.loads((directory / "granules.json").read_text())
        source = {"directory": str(directory), "hashes": {name: sha256_file(directory / name) for name in
                  ("plan.json", "inventory.json", "granules.json", "calendar_candidates.json") if (directory / name).is_file()},
                  "catalog_sha256": plan.get("catalog_sha256"), "module_sha256": plan.get("module_sha256"),
                  "protocol_document_sha256": plan.get("protocol_document_sha256"),
                  "source_status": report.get("status")}
        registry.append(source)
        if plan.get("protocol") != PROTOCOL or report.get("protocol") != PROTOCOL:
            raise ValueError(f"Incompatible protocol in {directory}")
        if not plan.get("metadata_only") or plan.get("credentials_used") is not False or report.get("protected_pixels_downloaded") != 0:
            raise ValueError(f"Source is not the expected anonymous metadata-only preflight: {directory}")
        if plan.get("calendar_candidates_per_stratum") != 2:
            raise ValueError("Candidate sampling differs from frozen preflight")
        for artifact in plan.get("protected_source_hashes", []):
            if artifact.get("present"):
                path, digest = artifact["path"], artifact["sha256"]
                protected_hashes.setdefault(path, set()).add(digest)
        for block in plan.get("blocks", []):
            previous = blocks_by_id.get(block["id"])
            if previous is not None and previous != block:
                raise ValueError(f"Spatial block changed: {block['id']}")
            blocks_by_id[block["id"]] = block
        for query in report.get("queries", []):
            pilot, product, flag = query["pilot_id"], query["product"], query["day_night_flag"]
            if pilot not in pilots or product not in products or flag != "NIGHT":
                continue
            params = query["params"]
            first, last = [part[:10] for part in params["temporal"].split(",")]
            expected = PRODUCTS[product]
            verified = report.get("collection_verification", {}).get(product, {})
            identity_ok = (verified.get("verified") is True and all(verified.get(k) == expected[k] for k in
                           ("short_name", "version", "concept_id")) and params.get("collection_concept_id") == expected["concept_id"])
            accepted = query.get("complete") is True and not query.get("errors") and identity_ok
            source_queries.append({"source_directory": str(directory), "pilot_id": pilot, "product": product,
                                   "period": [first, last], "source_complete": query.get("complete"),
                                   "accepted_for_merge": accepted, "collection_identity_valid": identity_ok,
                                   "cmr_hits": query.get("cmr_hits"), "errors": query.get("errors", [])})
            if not accepted:
                continue
            coverage[(pilot, product)].append((first, last))
            query_rows = [r for r in rows if r["pilot_id"] == pilot and r["product"] == product and first <= r["utc_date"] <= last]
            if len(query_rows) != query["catalog_records_loaded"]:
                raise ValueError(f"Record count differs from completed query: {directory}")
            for row in query_rows:
                if not str(start) <= row["utc_date"] <= str(end):
                    continue
                if row.get("day_night_flag") != "NIGHT":
                    issues.append({"granule_id": row.get("granule_concept_id"), "issue": "Night query returned a different metadata day/night flag"})
                    continue
                if row.get("collection") != expected:
                    raise ValueError("Normalized record collection differs from pinned product")
                key = (pilot, product, row["granule_concept_id"])
                if key in rows_by_key:
                    duplicate_count += 1
                    if rows_by_key[key] != row:
                        conflicting.add(key)
                else:
                    rows_by_key[key] = row
    for key in conflicting:
        row = rows_by_key[key]
        row["footprint_qa"]["status"] = "review_required"
        row["footprint_qa"]["issues"].append("Conflicting normalized records in different immutable source shards")
    for path, hashes in sorted(protected_hashes.items()):
        if len(hashes) > 1:
            protected_conflicts.append({"path": path, "hashes": sorted(hashes)})
    rows = sorted(rows_by_key.values(), key=lambda r: (r["pilot_id"], r["product"], r["time_start"], r["granule_concept_id"]))
    periods = []
    for pilot in pilots:
        for product in products:
            gaps = coverage_gaps(coverage[(pilot, product)], start, end)
            subset = [row for row in rows if row["pilot_id"] == pilot and row["product"] == product]
            periods.append({"pilot_id": pilot, "product": product, "required_period": [str(start), str(end)],
                            "complete": not gaps, "gaps": gaps, **summarize(subset)})
    complete = all(period["complete"] for period in periods) and not issues and not protected_conflicts and not conflicting
    report = {"protocol": PROTOCOL, "status": "complete_metadata_only" if complete else "partial_or_review_required",
              "created_utc": datetime.now(timezone.utc).isoformat(), "merge_module_sha256": sha256_file(__file__),
              "period": [str(start), str(end)], "source_registry": registry, "source_queries": source_queries,
              "coverage": periods, "duplicates_removed": duplicate_count, "record_conflicts_quarantined": len(conflicting),
              "protected_source_hashes": {path: sorted(digests) for path, digests in sorted(protected_hashes.items())},
              "protected_source_conflicts": protected_conflicts, "issues": issues,
              "geometry_issue_counts": dict(Counter(issue for row in rows for issue in row["footprint_qa"]["issues"])),
              "metadata_only": True, "protected_pixels_downloaded": 0,
              "label_sample_frozen": False, "fitting_authorized_by_this_manifest": False,
              "note": "Calendar metadata is frozen by exact hashes. Protected-pixel QA, input availability and the fitting addendum remain necessary."}
    candidates = {"protocol": PROTOCOL, "inventory_complete": complete, "frozen_for_label_acquisition": False,
                  "source_registry_sha256": None, "acquisitions": calendar_candidates(rows, 2),
                  "note": "Deterministic metadata candidates; quarantined records excluded. Preserve whole dates/orbits and spatial holdouts for any separately authorized engineering QA."}
    return rows, candidates, report, sorted(blocks_by_id.values(), key=lambda b: b["id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilots", nargs="+", default=["greater_london", "sioux_falls"])
    parser.add_argument("--products", nargs="+", choices=PRODUCTS, default=["ecostress_v2", "aster_v4"])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2023, 12, 31))
    args = parser.parse_args(argv)
    if not date(2021, 1, 1) <= args.start <= args.end <= date(2023, 12, 31):
        parser.error("This merge is restricted to the2021–2023 fitting/development/calibration metadata")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Merge output must be new or empty")
    shards = discover_shards(args.sources)
    if not shards:
        parser.error("No metadata shards found")
    rows, candidates, registry, blocks = build_merge(shards, args.pilots, args.products, args.start, args.end)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "merge_registry.json", registry)
    candidates["source_registry_sha256"] = sha256_file(args.output / "merge_registry.json")
    write_json(args.output / "granules.json", rows)
    write_json(args.output / "calendar_candidates.json", candidates)
    write_json(args.output / "spatial_blocks.json", blocks)
    print(json.dumps({"status": registry["status"], "output": str(args.output.resolve()), "coverage": registry["coverage"],
                      "geometry_issue_counts": registry["geometry_issue_counts"], "source_shards": len(shards)}), flush=True)
    return 0 if registry["status"] == "complete_metadata_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
