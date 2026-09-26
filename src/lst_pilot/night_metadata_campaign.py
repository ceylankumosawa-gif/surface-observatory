"""Anonymous Option B metadata campaign with a single global resource budget.

No protected asset requests, authentication, model fitting or production writes.
Writes new immutable calendar shards for ``night_inventory_merge``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

from .night_inventory import (
    LEGACY_PATHS, MetadataClient, PRODUCTS, PROTOCOL, calendar_candidates,
    inventory_query, pilot_blocks, sha256_file, write_json,
)

MAX_REQUESTS = 400
MAX_BYTES = 256 * 1024 * 1024
MAX_SECONDS = 1200


def campaign_queries():
    quarters = [("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")]
    queries = []
    for start, end in quarters:
        queries.append(("greater_london", "ecostress_v2", f"2023-{start}", f"2023-{end}"))
    for year in (2021, 2022):
        for start, end in quarters:
            for pilot in ("greater_london", "sioux_falls"):
                queries.append((pilot, "ecostress_v2", f"{year}-{start}", f"{year}-{end}"))
    for year in (2021, 2022):
        for pilot in ("greater_london", "sioux_falls"):
            queries.append((pilot, "aster_v4", f"{year}-01-01", f"{year}-12-31"))
    return queries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists() and any(output.iterdir()):
        parser.error("Campaign output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    start_clock = time.monotonic()
    catalog_path = root / "web/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    pilots = {p["properties"]["id"]: p for p in catalog["pilots"]["features"]}
    sources = [{"path": path, "present": (root / path).is_file(),
                "sha256": sha256_file(root / path) if (root / path).is_file() else None} for path in LEGACY_PATHS]
    common = {"protocol": PROTOCOL, "catalog_sha256": sha256_file(catalog_path),
              "module_sha256": sha256_file(Path(__file__).with_name("night_inventory.py")),
              "campaign_module_sha256": sha256_file(__file__),
              "protocol_document_sha256": sha256_file(root / "reports/night_replacement/OPTION_B_PROTOCOL.md"),
              "metadata_only": True, "protected_pixel_downloads_permitted": False,
              "credentials_used": False, "protected_source_hashes": sources,
              "calendar_candidates_per_stratum": 2,
              "split_rules": {"fit": "2021–2022", "development": "2023 January–June", "calibration": "2023 July–December",
                              "reserved_legacy_test": "2024; already inspected, never used for tuning here",
                              "reserved_blind_test": "2025; metadata only until final evaluation"}}
    queries = campaign_queries()
    plan = {**common, "created_utc": datetime.now(timezone.utc).isoformat(),
            "limits": {"max_requests": MAX_REQUESTS, "max_metadata_bytes": MAX_BYTES, "max_seconds": MAX_SECONDS},
            "ordered_queries": queries,
            "reuse_source": "runs/option_b_inventory_2023_night_20260909; use complete Sioux2023 V2 and both ASTER2023 queries only"}
    write_json(output / "campaign_plan.json", plan)
    # Leave a full receive-chunk margin, since a request may cross the byte limit.
    client = MetadataClient(MAX_REQUESTS, MAX_BYTES - 65536)
    complete_shards, partial_shards, errors = [], [], []
    verified = {}

    def deadline(signum, frame):
        raise TimeoutError("Campaign 20-minute limit reached")

    previous_handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(MAX_SECONDS)
    try:
        for key in ("ecostress_v2", "aster_v4"):
            verified[key] = client.verify_product(key)
        for pilot_id, key, first, last in queries:
            if client.requests >= MAX_REQUESTS or client.bytes >= MAX_BYTES - 65536:
                errors.append("Global metadata budget reached")
                break
            shard = output / f"{pilot_id}_{key}_{first}_{last}"
            shard.mkdir()
            block_plan = {**common, "created_utc": datetime.now(timezone.utc).isoformat(),
                          "pilots": [pilot_id], "products": {key: PRODUCTS[key]}, "period": [first, last],
                          "day_night_flags": ["NIGHT"], "blocks": pilot_blocks(pilots[pilot_id])}
            write_json(shard / "plan.json", block_plan)
            before_requests, before_bytes, before_audit = client.requests, client.bytes, len(client.audit)
            rows, query = inventory_query(client, pilots[pilot_id], key, "NIGHT", first, last, 100, 40)
            write_json(shard / "granules.json", rows)
            write_json(shard / "inventory.json", {"protocol": PROTOCOL,
                       "status": "complete_metadata_only" if query["complete"] else "partial_metadata_only",
                       "collection_verification": {key: verified[key]}, "collection_verification_errors": {},
                       "queries": [query], "http_audit": client.audit[before_audit:],
                       "metadata_bytes_read": client.bytes - before_bytes,
                       "network_requests": client.requests - before_requests, "protected_pixels_downloaded": 0})
            write_json(shard / "calendar_candidates.json", {"protocol": PROTOCOL, "inventory_complete": query["complete"],
                       "frozen_for_label_acquisition": False, "acquisitions": calendar_candidates(rows)})
            (complete_shards if query["complete"] else partial_shards).append(str(shard))
            write_json(output / "progress.json", {"complete_shards": complete_shards, "partial_shards": partial_shards,
                       "metadata_bytes_read": client.bytes, "network_requests": client.requests,
                       "elapsed_seconds": round(time.monotonic() - start_clock, 2)})
            print(json.dumps({"shard": str(shard), "period": [first, last], "complete": query["complete"],
                              "loaded": len(rows), "orbits_or_timestamps": query["unique_acquisition_groups"],
                              "utc_dates": query["unique_utc_dates"], "quarantined": query["review_required_records"],
                              "total_metadata_mib": round(client.bytes / 1024 / 1024, 2),
                              "total_requests": client.requests}), flush=True)
    except (TimeoutError, KeyboardInterrupt, ValueError) as error:
        errors.append(str(error))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        report = {"protocol": PROTOCOL, "complete_shards": complete_shards, "partial_shards": partial_shards,
                  "expected_shards": len(queries), "errors": errors, "collection_verification": verified,
                  "network_requests": client.requests, "metadata_bytes_read": client.bytes,
                  "elapsed_seconds": round(time.monotonic() - start_clock, 2),
                  "protected_pixels_downloaded": 0, "credentials_used": False}
        report["status"] = "complete_metadata_only" if len(complete_shards) == len(queries) and not errors else "partial_metadata_only"
        write_json(output / "campaign.json", report)
        print(json.dumps(report), flush=True)
    return 0 if report["status"] == "complete_metadata_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
