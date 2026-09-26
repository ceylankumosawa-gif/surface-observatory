"""More independent Landsat fitting dates; frozen metadata, bounded window reads."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import fcntl
import json
import logging
from pathlib import Path
import time

import pandas as pd

from . import option_b_day as day, satellite
from .ecostress import digest
from .option_b_cohort import spatial_flags

VERSION = "more-days-20260909-v1"
MONTH_ORDER = (1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12)


def save(path, value):
    satellite._save_json(Path(path), value)


def fitting_date(item):
    stamp = pd.Timestamp(item["properties"]["datetime"])
    if stamp.tzinfo is None:
        raise ValueError("Catalog timestamp must have a timezone.")
    stamp = stamp.tz_convert("UTC")
    if stamp.year not in (2021, 2022):
        raise ValueError("Only 2021–2022 fitting dates; no thermal source was opened.")
    return stamp


def catalog(region, output):
    """Enumerate metadata, with a hard page bound and explicit truncation flag."""
    path = Path(output) / "catalog" / f'{region["id"]}.json'
    if path.exists():
        return json.loads(path.read_text())
    items = {}
    request = {"collections": [satellite.COLLECTION], "bbox": satellite.region_bbox(region),
               "datetime": "2021-01-01T00:00:00Z/2022-12-31T23:59:59Z", "limit": 500,
               "query": {"platform": {"in": ["landsat-8", "landsat-9"]},
                         "landsat:collection_category": {"eq": "T1"},
                         "landsat:correction": {"eq": "L2SP"}}}
    payload = request.copy()
    url = satellite.STAC_URL + "/search"
    method = "POST"
    next_link = None
    with satellite._session() as session:
        for page in range(4):
            response = (session.post(url, json=payload, timeout=(15, 90)) if method == "POST"
                        else session.get(url, timeout=(15, 90)))
            response.raise_for_status()
            result = response.json()
            for item in result.get("features", []):
                fitting_date(item)
                items[item["id"]] = item
            next_link = next((x for x in result.get("links", []) if x.get("rel") == "next"), None)
            if not next_link:
                break
            url = next_link["href"]
            if not url.startswith(satellite.STAC_URL + "/"):
                raise ValueError("Unexpected catalog pagination host.")
            method = next_link.get("method", "GET").upper()
            if method not in ("GET", "POST"):
                raise ValueError("Unexpected pagination method.")
            payload = ({**payload, **next_link.get("body", {})} if next_link.get("merge")
                       else next_link.get("body", {}))
    record = {"region_id": region["id"], "request": request, "pages": page + 1,
              "truncated": next_link is not None, "items": list(items.values())}
    save(path, record)
    return record


def queues(items, region, old_dates, old_scene_ids):
    groups = {(year, month): [] for month in MONTH_ORDER for year in (2021, 2022)}
    rejection = {}
    for item in items:
        stamp = fitting_date(item)
        reason = None
        if stamp.strftime("%Y-%m-%d") in old_dates:
            reason = "date_already_in_original_cohort"
        elif item["id"] in old_scene_ids:
            reason = "scene_already_attempted"
        elif item["properties"].get("platform") not in ("landsat-8", "landsat-9"):
            reason = "other_platform"
        elif not set((*satellite.REQUIRED_ASSETS, "qa")).issubset(item.get("assets", {})):
            reason = "missing_source_assets"
        elif item["properties"].get("eo:cloud_cover", 100) > 80:
            reason = "scene_cloud_above_80_percent"
        if reason:
            rejection[reason] = rejection.get(reason, 0) + 1
            continue
        item = dict(item)
        item["_pilot_roi_coverage_fraction"] = satellite.scene_coverage_fraction(item, region)
        if item["_pilot_roi_coverage_fraction"] < .15:
            rejection["footprint_below_15_percent"] = rejection.get("footprint_below_15_percent", 0) + 1
            continue
        groups[stamp.year, stamp.month].append(item)
    result = []
    for (year, month), values in groups.items():
        values.sort(key=lambda i: (-i["_pilot_roi_coverage_fraction"],
                                  i["properties"].get("eo:cloud_cover", 100), i["id"]))
        result.append({"year": year, "month": month, "available_candidates": len(values), "items": values[:4]})
    return result, rejection


def freeze(root, old_cohort, old_audit, output, protocol):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        raise FileExistsError("A frozen plan already exists; run it or choose a new directory.")
    areas = json.loads((Path(root) / "pilot/areas_resolved.json").read_text())["areas"]
    areas = [a for a in areas if a["id"] != "cabauw"]
    # Read identities and timestamps only from the previous table.
    prior = pd.read_parquet(old_cohort, columns=["region_id", "datetime_utc"])
    stamps = pd.to_datetime(prior.datetime_utc, utc=True)
    if not stamps.dt.year.isin((2021, 2022, 2023)).all():
        raise ValueError("Unexpected years in prior metadata.")
    prior["day"] = stamps.dt.strftime("%Y-%m-%d")
    attempted = json.loads(Path(old_audit).read_text())["records"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        catalogs = list(pool.map(lambda area: catalog(area, output), areas))
    groups = []
    for area, inventory in zip(areas, catalogs):
        old_dates = set(prior.loc[prior.region_id.eq(area["id"]), "day"])
        old_ids = {x["scene_id"] for x in attempted if x["region_id"] == area["id"]}
        ranked, rejected = queues(inventory["items"], area, old_dates, old_ids)
        groups.append({"region": area, "target_dates": 24 if area["id"] in ("greater_london", "sioux_falls") else 16,
                       "old_dates": sorted(old_dates), "strata": ranked, "catalog_scenes": len(inventory["items"]),
                       "catalog_truncated": inventory["truncated"], "rejections": rejected,
                       "catalog_sha256": digest(output / "catalog" / f'{area["id"]}.json')})
    groups.sort(key=lambda g: (g["region"]["id"] not in ("greater_london", "sioux_falls"), g["region"]["id"]))
    plan = {"version": VERSION, "source_sha256": digest(__file__), "sampler_sha256": digest(day.__file__),
            "old_cohort_sha256": digest(old_cohort), "old_audit_sha256": digest(old_audit),
            "protocol_sha256": digest(protocol), "old_cohort": str(Path(old_cohort).resolve()),
            "per_window_samples": 60, "max_attempts_per_pilot": 96, "minimum_safe_cells_for_quota": 40,
            "selection": "Quarter-interleaved calendar months, 2021/2022 alternation; footprint then cloud metadata; no residual/LST ranking.",
            "groups": groups}
    save(output / "plan.json", plan)
    summary = [{"pilot": g["region"]["id"], "catalog_scenes": g["catalog_scenes"],
                "catalog_truncated": g["catalog_truncated"], "target_dates": g["target_dates"],
                "queued_scenes": sum(len(s["items"]) for s in g["strata"]), "rejections": g["rejections"]} for g in groups]
    save(output / "inventory_summary.json", summary)
    print(json.dumps(summary), flush=True)


def acquire_pilot(spec):
    plan, group, output, runtime_seconds = spec
    logging.basicConfig(level=logging.ERROR)
    satellite.configure_safe_logging()
    region = group["region"]
    output = Path(output) / region["id"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signature = satellite._hash(plan)
        logpath = output / "audit.json"
        log = json.loads(logpath.read_text()) if logpath.exists() else {"plan_signature": signature, "records": []}
        if log["plan_signature"] != signature:
            raise ValueError("Acquisition plan changed.")
        if (output / "completion.json").exists():
            return json.loads((output / "completion.json").read_text())
        done = {r["scene_id"] for r in log["records"]}
        used_dates = {r["date"] for r in log["records"] if r.get("rows", 0) > 0}
        completed_months = {(r["year"], r["month"]) for r in log["records"] if r.get("safe_rows", 0) >= 40}
        successes = sum(r.get("safe_rows", 0) >= 40 for r in log["records"])
        start = time.monotonic()
        # One useful date per calendar month; whole months spread across quarters.
        for group_month in group["strata"]:
            month = (group_month["year"], group_month["month"])
            if month in completed_months:
                continue
            for item in group_month["items"]:
                stamp = fitting_date(item)
                date = stamp.strftime("%Y-%m-%d")
                if item["id"] in done or date in used_dates:
                    continue
                if successes >= group["target_dates"] or len(log["records"]) >= 96 or time.monotonic() - start >= runtime_seconds:
                    break
                path = output / "scenes" / f'{item["id"]}.parquet'
                record = {"scene_id": item["id"], "date": date, "year": stamp.year, "month": stamp.month,
                          "region_id": region["id"], "datetime_utc": stamp.isoformat(), "path": str(path)}
                try:
                    frame, windows = day.sample_scene(item, region)
                    if len(frame):
                        frame = spatial_flags(frame, {region["id"]: region})
                    safe = int((~frame.spatial_holdout & ~frame.in_holdout_buffer).sum()) if len(frame) else 0
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp.parquet")
                    frame.to_parquet(tmp, index=False)
                    tmp.replace(path)
                    record.update(status="complete", rows=len(frame), safe_rows=safe, windows=windows, sha256=digest(path))
                except Exception as error:
                    record.update(status="failed", rows=0, safe_rows=0, error=satellite._safe_error(error))
                log["records"].append(record)
                save(logpath, log)
                done.add(item["id"])
                if record["rows"]:
                    used_dates.add(date)
                print(json.dumps({k: record[k] for k in ("region_id", "scene_id", "status", "rows", "safe_rows")}), flush=True)
                if record["safe_rows"] >= 40:
                    successes += 1
                    completed_months.add(month)
                    break
            if successes >= group["target_dates"] or len(log["records"]) >= 96 or time.monotonic() - start >= runtime_seconds:
                break
        parts = []
        for r in log["records"]:
            if r.get("rows", 0):
                if digest(r["path"]) != r["sha256"]:
                    raise ValueError("Completed sample hash changed.")
                parts.append(pd.read_parquet(r["path"]))
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(frame) and (frame.sample_id.duplicated().any() or pd.to_datetime(frame.datetime_utc, utc=True).dt.year.gt(2022).any()):
            raise ValueError("Invalid new observation identities or dates.")
        temp = output / "samples.tmp.parquet"
        frame.to_parquet(temp, index=False)
        temp.replace(output / "samples.parquet")
        report = {"region_id": region["id"], "plan_signature": signature, "attempts": len(log["records"]),
                  "useful_dates": successes, "positive_dates": len(used_dates), "rows": len(frame),
                  "elapsed_seconds": time.monotonic() - start,
                  "stopped_by_time_cap": time.monotonic() - start >= runtime_seconds,
                  "target_dates": group["target_dates"], "path": str(output / "samples.parquet"),
                  "sha256": digest(output / "samples.parquet"), "research_only": True}
        save(output / "completion.json", report)
        print(json.dumps(report), flush=True)
        return report


def run(output, workers=2, runtime_seconds=1800):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if digest(__file__) != plan["source_sha256"] or digest(day.__file__) != plan["sampler_sha256"]:
        raise ValueError("Frozen acquisition code changed.")
    if not 1 <= workers <= 2 or not 60 <= runtime_seconds <= 3600:
        raise ValueError("Require <=2 workers and <=1 hour per pilot.")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(acquire_pilot, [(plan, g, output, runtime_seconds) for g in plan["groups"]]))
    save(output / "completion.json", {"pilots": results, "total_rows": sum(r["rows"] for r in results),
                                      "total_positive_dates": sum(r["positive_dates"] for r in results)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "run"))
    p.add_argument("--root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--old-cohort")
    p.add_argument("--old-audit")
    p.add_argument("--protocol")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--runtime-seconds", type=int, default=1800)
    a = p.parse_args()
    if a.mode == "plan":
        freeze(a.root, a.old_cohort, a.old_audit, a.output, a.protocol)
    else:
        run(a.output, a.workers, a.runtime_seconds)


if __name__ == "__main__":
    main()
