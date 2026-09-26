"""Bounded, anonymous CMR metadata inventory; never opens thermal data assets.

Run on the pilot server with ``python -m lst_pilot.night_inventory --help``.
This research preflight does not import a fitted model or inspect label values.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlencode, urlsplit

import requests
from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import transform, unary_union

CMR = "https://cmr.earthdata.nasa.gov/search"
PROTOCOL = "option-b-2026-09-09-v1"
SEED = "lst-option-b-v1"
PRODUCTS = {
    "ecostress_v2": {"short_name": "ECO_L2T_LSTE", "version": "002", "concept_id": "C2076090826-LPCLOUD", "family": "ECOSTRESS"},
    "ecostress_v3": {"short_name": "ECO_L2T_LSTE", "version": "003", "concept_id": "C3998139651-LPCLOUD", "family": "ECOSTRESS"},
    "aster_v4": {"short_name": "AST_08", "version": "004", "concept_id": "C3306885674-LPCLOUD", "family": "ASTER"},
}
ECO_NAME = re.compile(r"^ECOv(?P<version>\d{3})_L2T_LSTE_(?P<orbit>\d+)_(?P<scene>\d+)_(?P<tile>\d{2}[C-X][A-Z]{2})_(?P<time>\d{8}T\d{6})_")
ASTER_NAME = re.compile(r"^AST_08_(?P<version>\d{3})(?P<time>\d{14})_")
LEGACY_PATHS = (
    "runs/pilot_v1_model/model.joblib", "runs/pilot_v1_model/metrics.json",
    "runs/pilot_v1_model/heldout_predictions.parquet",
    "runs/pilot_v1_assembly/model_input.parquet",
    "runs/pilot_v1_satellite/satellite_manifest.json",
    "reports/night_replacement/access_audit.json",
)


class PreflightError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc(value):
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise PreflightError("CMR timestamp lacks a timezone")
    return timestamp.astimezone(timezone.utc)


def split_for_date(value):
    day = value if isinstance(value, date) else date.fromisoformat(value[:10])
    if date(2021, 1, 1) <= day <= date(2022, 12, 31):
        return "fit"
    if date(2023, 1, 1) <= day <= date(2023, 6, 30):
        return "development"
    if date(2023, 7, 1) <= day <= date(2023, 12, 31):
        return "calibration"
    if day.year == 2024:
        return "reserved_legacy_2024_test"
    if day.year == 2025:
        return "reserved_blind_2025_test"
    return "outside_frozen_protocol"


def pilot_blocks(pilot):
    """Fixed 10 km blocks; deterministic 1-in-5 reserved pattern and 1 km buffer."""
    props = pilot["properties"]
    left, bottom, right, top = props["extent_m"]
    result = []
    for row, y in enumerate(range(int(bottom), int(top), 10_000)):
        for column, x in enumerate(range(int(left), int(right), 10_000)):
            result.append({"id": f"{props['id']}_r{row:02d}_c{column:02d}",
                           "bounds_m": [x, y, min(x + 10_000, right), min(y + 10_000, top)],
                           "epsg": props["epsg"], "spatial_holdout": (row + 2 * column) % 5 == 0,
                           "holdout_buffer_m": 1000})
    return result


def name_identity(title, product, timestamp):
    """Different tiles/scenes on one ECOSTRESS orbit are one overpass group."""
    parsed = ECO_NAME.match(title) if product["family"] == "ECOSTRESS" else ASTER_NAME.match(title)
    if not parsed:
        raise PreflightError("Unrecognized product naming convention")
    fields = parsed.groupdict()
    if fields["version"] != product["version"]:
        raise PreflightError("Granule name version differs from pinned collection")
    fmt = "%Y%m%dT%H%M%S" if product["family"] == "ECOSTRESS" else "%m%d%Y%H%M%S"
    named_time = datetime.strptime(fields["time"], fmt).replace(tzinfo=timezone.utc)
    if abs((named_time - timestamp).total_seconds()) >= 2:
        raise PreflightError("Granule name time differs from CMR start time")
    fields["acquisition_group"] = ("ECOSTRESS:orbit:" + fields["orbit"] if product["family"] == "ECOSTRESS"
                                    else "ASTER:time:" + named_time.isoformat())
    fields["orbit_available"] = product["family"] == "ECOSTRESS"
    return fields


def footprint_geometry(entry):
    """CMR JSON polygon coordinate order is latitude, longitude."""
    polygons = []
    for group in entry.get("polygons", []):
        for ring in group:
            values = [float(value) for value in ring.split()]
            if len(values) < 8 or len(values) % 2:
                raise PreflightError("Malformed CMR polygon ring")
            points = [(values[i + 1], values[i]) for i in range(0, len(values), 2)]
            if any(not (-180 <= lon <= 180 and -90 <= lat <= 90) for lon, lat in points):
                raise PreflightError("Out-of-range CMR polygon coordinates")
            if max(p[0] for p in points) - min(p[0] for p in points) > 180:
                raise PreflightError("Antimeridian/spanning CMR footprint needs separate review")
            polygon = Polygon(points)
            if not polygon.is_valid or polygon.is_empty:
                raise PreflightError("Invalid CMR polygon topology")
            polygons.append(polygon)
    for extent in entry.get("boxes", []):
        south, west, north, east = map(float, extent.split())
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise PreflightError("Invalid or antimeridian CMR box")
        polygons.append(box(west, south, east, north))
    if not polygons:
        raise PreflightError("No polygon/box in CMR summary; geometry remains unverified")
    return unary_union(polygons)


def footprint_qa(entry, pilot, identity):
    failures = []
    geometry = None
    try:
        geometry = footprint_geometry(entry)
        pilot_geometry = shape(pilot["geometry"])
        if not geometry.intersects(pilot_geometry):
            failures.append("CMR footprint does not intersect the actual pilot polygon")
        project = Transformer.from_crs(4326, pilot["properties"]["epsg"], always_xy=True).transform
        projected = transform(project, geometry)
        if not math.isfinite(projected.area) or projected.area > 100_000 * 1_000_000:
            failures.append("Footprint is nonfinite or implausibly large for one thermal granule")
        if "tile" in identity:
            tile = identity["tile"]
            zone = int(tile[:2])
            bands = "CDEFGHJKLMNPQRSTUVWX"
            if not 1 <= zone <= 60 or tile[2] not in bands:
                failures.append("Invalid MGRS zone/band")
            else:
                # A broad geographic screen, not a reconstruction of the MGRS square.
                west = (zone - 1) * 6 - 180
                south = -80 + bands.index(tile[2]) * 8
                zone_band = box(west - 3, south - 2, west + 9, min(south + 10, 86))
                if not zone_band.intersects(pilot_geometry):
                    failures.append("Named MGRS zone/band is inconsistent with pilot geography")
    except (ValueError, TypeError, OverflowError) as error:
        failures.append(str(error))
    return {"status": "review_required" if failures else "metadata_consistent_pixels_unverified",
            "issues": failures,
            "cmr_geometry": mapping(geometry) if geometry is not None else None,
            "actual_cog_bounds_checked": False, "cloud_free_pixels_checked": False}


def normalize(entry, pilot, key):
    product = PRODUCTS[key]
    title = entry.get("title", "")
    timestamp = utc(entry["time_start"])
    issues = []
    try:
        identity = name_identity(title, product, timestamp)
    except (ValueError, TypeError) as error:
        identity = {"acquisition_group": "unverified:" + entry.get("id", title), "orbit_available": False}
        issues.append(str(error))
    qa = footprint_qa(entry, pilot, identity)
    qa["issues"] += issues
    if issues:
        qa["status"] = "review_required"
    local_solar_hour = (timestamp.hour + timestamp.minute / 60 + pilot["properties"]["center"][0] / 15) % 24
    return {"pilot_id": pilot["properties"]["id"], "product": key, "collection": product,
            "granule_concept_id": entry.get("id"), "granule_title": title,
            "time_start": timestamp.isoformat(), "utc_date": timestamp.date().isoformat(),
            "day_night_flag": entry.get("day_night_flag", "UNKNOWN"),
            "catalog_cloud_cover": entry.get("cloud_cover"),
            "local_solar_hour_approx": round(local_solar_hour, 3),
            "local_solar_hour_bin": int(local_solar_hour // 6),
            "temporal_split": split_for_date(timestamp.date()), "identity": identity, "footprint_qa": qa}


def summarize(records):
    good = [record for record in records if record["footprint_qa"]["status"] == "metadata_consistent_pixels_unverified"]
    return {"catalog_records_loaded": len(records), "unique_granule_ids": len({r["granule_concept_id"] for r in records}),
            "metadata_consistent_records": len(good), "review_required_records": len(records) - len(good),
            "unique_acquisition_groups": len({r["identity"]["acquisition_group"] for r in good}),
            "unique_utc_dates": len({r["utc_date"] for r in good}),
            "calendar_records_by_month": dict(sorted(Counter(r["utc_date"][:7] for r in good).items())),
            "records_by_temporal_split": dict(Counter(r["temporal_split"] for r in good)),
            "usable_thermal_acquisitions": None,
            "interpretation": "Counts are metadata candidates; actual coverage/cloud/thermal QA is unverified. ASTER acquisition groups use time, because orbit is absent from this summary."}


def calendar_candidates(records, per_stratum=2):
    """Select whole overpasses by calendar/time bin, not thermal error or cloudiness."""
    grouped = defaultdict(list)
    for record in records:
        if record["footprint_qa"]["status"] == "metadata_consistent_pixels_unverified":
            grouped[(record["pilot_id"], record["identity"]["acquisition_group"])].append(record)
    strata = defaultdict(list)
    for (pilot, acquisition), group in grouped.items():
        first = min(group, key=lambda r: r["time_start"])
        splits = sorted({r["temporal_split"] for r in group})
        # Never let an orbit crossing a split boundary pull a reserved record into fit.
        split = splits[0] if len(splits) == 1 else "cross_split_boundary_reserved"
        item = {"pilot_id": pilot, "acquisition_group": acquisition, "utc_dates": sorted({r["utc_date"] for r in group}),
                "temporal_split": split, "products_present": sorted({r["product"] for r in group}),
                "granule_concept_ids": sorted({r["granule_concept_id"] for r in group}),
                "metadata_day_night_flags": sorted({r["day_night_flag"] for r in group}),
                "rank": hashlib.sha256(f"{SEED}:{pilot}:{acquisition}".encode()).hexdigest()}
        stratum = (pilot, first["collection"]["family"], first["utc_date"][:7], first["local_solar_hour_bin"], first["day_night_flag"])
        item["stratum"] = list(stratum)
        strata[stratum].append(item)
    selected = []
    for stratum in sorted(strata):
        for rank, item in enumerate(sorted(strata[stratum], key=lambda v: v["rank"]), 1):
            item["rank_within_stratum"] = rank
            item["initial_candidate"] = rank <= per_stratum
            selected.append(item)
    return selected


class MetadataClient:
    """Only public CMR JSON search endpoints; no netrc, cookies or bearer headers."""
    def __init__(self, max_requests=40, max_bytes=32 * 1024 * 1024):
        self.max_requests, self.max_bytes = max_requests, max_bytes
        self.requests = self.bytes = 0
        self.audit = []
        self.session = requests.Session()
        self.session.trust_env = False  # requests otherwise consults ~/.netrc.
        self.session.headers.update({"User-Agent": "lst-pilot-metadata-preflight/1"})

    def get(self, endpoint, params):
        if endpoint not in {"collections.json", "granules.json"}:
            raise PreflightError("Only public CMR search metadata endpoints are permitted")
        if self.requests >= self.max_requests or self.bytes >= self.max_bytes:
            raise PreflightError("Metadata request/byte budget reached; inventory is partial")
        url = CMR + "/" + endpoint
        self.requests += 1
        self.session.cookies.clear()
        with self.session.get(url, params=params, timeout=(10, 45), stream=True, allow_redirects=False) as response:
            if response.status_code != 200:
                raise PreflightError(f"CMR HTTP {response.status_code}; no redirect or authentication attempted")
            data = bytearray()
            for chunk in response.iter_content(64 * 1024):
                self.bytes += len(chunk)
                if self.bytes > self.max_bytes or len(data) + len(chunk) > 8 * 1024 * 1024:
                    raise PreflightError("Metadata byte budget reached; inventory is partial")
                data.extend(chunk)
            self.audit.append({"url": url, "params": params, "http_status": 200, "bytes": len(data),
                               "body_sha256": hashlib.sha256(data).hexdigest(), "cmr_hits": response.headers.get("CMR-Hits")})
            return json.loads(data), int(response.headers.get("CMR-Hits", 0))

    def verify_product(self, key):
        expected = PRODUCTS[key]
        body, _ = self.get("collections.json", {"concept_id": expected["concept_id"], "page_size": 1})
        entries = body.get("feed", {}).get("entry", [])
        if len(entries) != 1:
            raise PreflightError(f"Pinned collection {key} was not found")
        entry = entries[0]
        if (entry.get("id") != expected["concept_id"] or entry.get("short_name") != expected["short_name"]
                or entry.get("version_id") != expected["version"]):
            raise PreflightError(f"Pinned collection identity changed for {key}")
        return {**expected, "verified": True, "title": entry.get("title"), "time_start": entry.get("time_start")}


def inventory_query(client, pilot, product, flag, start, end, page_size, max_pages):
    bounds = shape(pilot["geometry"]).bounds
    params = {"collection_concept_id": PRODUCTS[product]["concept_id"],
              "bounding_box": ",".join(f"{v:.8f}" for v in bounds), "day_night_flag": flag,
              "temporal": f"{start}T00:00:00Z,{end}T23:59:59Z", "page_size": page_size,
              "sort_key[]": ["start_date", "producer_granule_id"]}
    records, seen, hits, errors = [], set(), None, []
    complete = False
    for page in range(1, max_pages + 1):
        try:
            body, current_hits = client.get("granules.json", {**params, "page_num": page})
            if hits is not None and hits != current_hits:
                errors.append("CMR hit count changed during pagination; repeat inventory before freezing labels")
            hits = current_hits
            entries = body.get("feed", {}).get("entry", [])
            for entry in entries:
                if entry.get("id") in seen:
                    errors.append("Repeated granule ID during pagination; verify completeness")
                    continue
                seen.add(entry.get("id"))
                try:
                    records.append(normalize(entry, pilot, product))
                except (ValueError, KeyError, TypeError) as error:
                    errors.append(f"Rejected malformed granule {entry.get('id')}: {error}")
            if len(seen) >= hits:
                complete = not errors
                break
            if not entries:
                errors.append("CMR pagination ended before reported hits")
                break
        except (PreflightError, requests.RequestException, json.JSONDecodeError) as error:
            errors.append(str(error))
            break
    if not complete and not errors:
        errors.append("Page cap reached; unique acquisition/date counts are lower bounds")
    return records, {"pilot_id": pilot["properties"]["id"], "product": product, "day_night_flag": flag,
                     "params": params, "cmr_hits": hits, "complete": complete, "errors": errors,
                     **summarize(records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--catalog", type=Path, default=Path("web/catalog.json"))
    parser.add_argument("--pilots", nargs="+", default=["greater_london", "sioux_falls"])
    parser.add_argument("--products", nargs="+", choices=PRODUCTS, default=list(PRODUCTS))
    parser.add_argument("--day-night", nargs="+", choices=["NIGHT", "DAY"], default=["NIGHT"])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2023, 12, 31))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=40)
    parser.add_argument("--max-metadata-mib", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages-per-query", type=int, default=15)
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true", help="Write frozen query/block plan without any network requests")
    args = parser.parse_args(argv)
    if not (date(2021, 1, 1) <= args.start <= args.end <= date(2025, 12, 31)):
        parser.error("Inventory is restricted to the frozen 2021–2025 protocol")
    if not (1 <= args.max_requests <= 100 and 1 <= args.max_metadata_mib <= 64
            and 1 <= args.page_size <= 100 and 1 <= args.max_pages_per_query <= 40 and 1 <= args.per_stratum <= 10):
        parser.error("Invalid resource bound; maximum100 requests,64 MiB,100 records/page,40 pages/query")
    root = args.root.resolve()
    catalog_path = args.catalog if args.catalog.is_absolute() else root / args.catalog
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be new or empty; frozen preflights are not overwritten")
    catalog = json.loads(catalog_path.read_text())
    available = {p["properties"]["id"]: p for p in catalog["pilots"]["features"]}
    if any(key not in available for key in args.pilots):
        parser.error("Unknown pilot ID")
    pilots = [available[key] for key in dict.fromkeys(args.pilots)]
    output.mkdir(parents=True, exist_ok=True)
    protected = [{"path": path, "present": (root / path).is_file(),
                  "sha256": sha256_file(root / path) if (root / path).is_file() else None}
                 for path in LEGACY_PATHS]
    plan = {"protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
            "catalog_sha256": sha256_file(catalog_path), "module_sha256": sha256_file(__file__),
            "metadata_only": True, "protected_pixel_downloads_permitted": False,
            "credentials_used": False, "period": [str(args.start), str(args.end)],
            "pilots": args.pilots, "products": {key: PRODUCTS[key] for key in args.products},
            "day_night_flags": args.day_night, "calendar_candidates_per_stratum": args.per_stratum,
            "limits": {"max_requests": args.max_requests, "max_metadata_bytes": args.max_metadata_mib * 1024 * 1024,
                       "page_size": args.page_size, "max_pages_per_query": args.max_pages_per_query},
            "protected_source_hashes": protected, "blocks": [b for p in pilots for b in pilot_blocks(p)],
            "split_rules": {"fit": "2021–2022", "development": "2023 January–June", "calibration": "2023 July–December",
                            "reserved_legacy_test": "2024; already inspected, never used for tuning here",
                            "reserved_blind_test": "2025; metadata only until final evaluation"}}
    write_json(output / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"status": "plan_only", "output": str(output), "network_requests": 0}))
        return 0
    client = MetadataClient(args.max_requests, args.max_metadata_mib * 1024 * 1024)
    verified, verification_errors, queries, records = {}, {}, [], []
    for key in args.products:
        try:
            verified[key] = client.verify_product(key)
        except (PreflightError, requests.RequestException, ValueError) as error:
            verification_errors[key] = str(error)
    for pilot in pilots:
        for key in verified:
            for flag in args.day_night:
                rows, audit = inventory_query(client, pilot, key, flag, args.start, args.end, args.page_size, args.max_pages_per_query)
                records.extend(rows)
                queries.append(audit)
                print(json.dumps({k: audit[k] for k in ("pilot_id", "product", "day_night_flag", "cmr_hits", "complete", "unique_acquisition_groups", "unique_utc_dates")}), flush=True)
    complete = not verification_errors and all(query["complete"] for query in queries)
    write_json(output / "granules.json", records)
    write_json(output / "calendar_candidates.json", {"protocol": PROTOCOL, "inventory_complete": complete,
               "frozen_for_label_acquisition": False,
               "note": "Freeze only after complete inventory and coverage review; metadata candidates do not authorize label access.",
               "acquisitions": calendar_candidates(records, args.per_stratum)})
    report = {"protocol": PROTOCOL, "status": "complete_metadata_only" if complete else "partial_metadata_only",
              "collection_verification": verified, "collection_verification_errors": verification_errors,
              "queries": queries, "http_audit": client.audit, "metadata_bytes_read": client.bytes,
              "network_requests": client.requests, "protected_pixels_downloaded": 0,
              "warnings": ["No cloud-free pixel coverage or actual COG footprint has been established.",
                           "Partial inventories yield lower bounds and cannot prove absence or freeze final samples.",
                           "ASTER has timestamp groups, not verified orbit groups; split conservatively by whole UTC dates.",
                           "The two-pilot experiment cannot establish global 100 m, hourly or all-weather accuracy."]}
    write_json(output / "inventory.json", report)
    print(json.dumps({"status": report["status"], "output": str(output), "metadata_bytes_read": client.bytes, "network_requests": client.requests}))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
