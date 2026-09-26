"""Anonymous, bounded native-LST metadata inventory and engineering plan freeze."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import pvlib
import requests
from .satellite import region_bbox

PRODUCTS = {
    "MOD21": {"collection_id": "C2565791036-LPCLOUD", "cmr_version": "061", "version": "061", "sensor": "Terra MODIS", "native_m": 1000, "minutes": 5, "suffix": ".hdf"},
    "VNP21": {"collection_id": "C2545314550-LPCLOUD", "cmr_version": "002", "version": "002", "sensor": "Suomi-NPP VIIRS", "native_m": 750, "minutes": 6, "suffix": ".nc"},
    "MOD03": {"collection_id": "C1379767668-LAADS", "cmr_version": "6.1", "version": "061", "sensor": "Terra MODIS geolocation", "native_m": 1000, "minutes": 5, "suffix": ".hdf"},
}
NAME = re.compile(r"^(MOD21|VNP21|MOD03)\.A(\d{4})(\d{3})\.(\d{4})\.(\d{3})\.(\d{13})(?:\.(hdf|nc))?$")
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n"); temp.replace(path)


def utc(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None: raise ValueError("Explicit UTC acquisition time required.")
    return t.tz_convert("UTC")


def identity(name):
    m = NAME.fullmatch(name)
    if not m: raise ValueError("Unexpected native product name.")
    product, year, day, hm, version, produced, suffix = m.groups()
    if version != PRODUCTS[product]["version"]: raise ValueError("Product version changed.")
    stamp = pd.Timestamp(datetime.strptime(year+day+hm, "%Y%j%H%M"), tz="UTC")
    if stamp.strftime("%Y%j") != year+day: raise ValueError("Invalid year/day in source identity.")
    return {"product": product, "version": version, "start": stamp.isoformat(),
            "stem": name.rsplit(".", 1)[0] if suffix else name,
            "acquisition_key": f"{product}:A{year}{day}.{hm}", "production_tag": produced}


def normalize(item):
    u, meta = item["umm"], item["meta"]
    names = [u["GranuleUR"], *[x["Identifier"] for x in u.get("DataGranule", {}).get("Identifiers", []) if x.get("IdentifierType") == "ProducerGranuleId"]]
    names = list(dict.fromkeys(x for x in names if NAME.fullmatch(x)))
    if not names: raise ValueError("CMR lacks a recognized producer granule identity.")
    ident = identity(names[0]); p = PRODUCTS[ident["product"]]
    c = u["CollectionReference"]
    if c["ShortName"] != ident["product"] or c["Version"] != p["cmr_version"] or meta["collection-concept-id"] != p["collection_id"]:
        raise ValueError("Pinned CMR collection identity mismatch.")
    interval = u["TemporalExtent"]["RangeDateTime"]
    start, end = utc(interval["BeginningDateTime"]), utc(interval["EndingDateTime"])
    if abs((start-utc(ident["start"])).total_seconds()) > 1 or not 0 < (end-start).total_seconds() <= p["minutes"]*60+1:
        raise ValueError("Granule temporal interval differs from its identity.")
    if start.year not in (2021, 2022, 2023) or end.year > 2023:
        raise ValueError("Only 2021–2023 metadata permitted.")
    links = [x["URL"] for x in u.get("RelatedUrls", []) if x.get("Type") == "GET DATA" and urlsplit(x["URL"]).path.endswith(ident["stem"]+p["suffix"])]
    if len(links) != 1: raise ValueError("Exactly one original native data URL required.")
    asset = urlsplit(links[0])
    allowed = (asset.hostname == "data.lpdaac.earthdatacloud.nasa.gov" and asset.path.startswith(f'/lp-prod-protected/{ident["product"]}.{p["version"]}/'))
    allowed |= ident["product"] == "MOD03" and asset.hostname == "ladsweb.modaps.eosdis.nasa.gov" and asset.path.startswith("/archive/allData/61/MOD03/")
    allowed |= ident["product"] == "MOD03" and asset.hostname == "data.laadsdaac.earthdatacloud.nasa.gov" and asset.path.startswith("/prod-lads/MOD03/")
    if not allowed or asset.scheme != "https" or asset.query or asset.username or asset.password:
        raise ValueError("Unexpected asset origin.")
    sizes = u.get("DataGranule", {}).get("ArchiveAndDistributionInformation", [])
    factors = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    estimate = sum(int(x["SizeInBytes"]) if x.get("SizeInBytes") else int(np.ceil(float(x["Size"])*factors[x["SizeUnit"]])) for x in sizes)
    if not estimate: raise ValueError("CMR archive size is required before acquisition.")
    return {**ident, "granule_id": meta["concept-id"], "cmr_revision": int(meta["revision-id"]),
            "collection_id": p["collection_id"], "granule_start_utc": start.isoformat(), "granule_end_utc": end.isoformat(),
            "cmr_day_night": u.get("DataGranule", {}).get("DayNightFlag", "Unknown"),
            "estimated_source_bytes": estimate, "asset_url": links[0], "cmr_metadata": item,
            "time_method": "Exact granule interval; no invented per-pixel timestamps", "native_nominal_resolution_m": p["native_m"]}


def fetch(params, path):
    path = Path(path)
    if path.exists(): return json.loads(path.read_text())
    with requests.Session() as s:
        s.trust_env = False
        response = s.get(CMR, params={**params, "page_size": 100}, timeout=(10, 45), allow_redirects=False)
        response.raise_for_status(); data = response.json()
        hits = int(response.headers.get("CMR-Hits", len(data.get("items", []))))
    if hits > 100 or hits != len(data.get("items", [])): raise ValueError("Metadata query exceeded its complete one-page bound.")
    out = {"request": params, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "hits": hits, **data}
    save(path, out); return out


def solar_phase(start, end, area):
    west, south, east, north = region_bbox(area)
    values = pvlib.solarposition.get_solarposition(pd.DatetimeIndex([utc(start), utc(end)]), (south+north)/2, (west+east)/2).elevation.to_numpy()
    return "day" if np.min(values) >= 10 else "night" if np.max(values) <= -6 else "twilight"


def inventory(areas_path, output):
    output = Path(output); areas = json.loads(Path(areas_path).read_text())["areas"]
    tasks = [(a, product, f"{year}-{month:02d}-15") for a in areas for product in ("MOD21", "VNP21")
             for year in (2021, 2022, 2023) for month in (1, 4, 7, 10)]
    def query(spec):
        area, product, day = spec; source = output/"public_metadata"/area["id"]/f"{product}_{day}.json"
        metadata = fetch({"collection_concept_id": PRODUCTS[product]["collection_id"], "temporal": day+"T00:00:00Z,"+day+"T23:59:59Z", "bounding_box": ",".join(map(str, region_bbox(area)))}, source)
        records = []
        for item in metadata["items"]:
            rec = normalize(item)
            rec.update(region_id=area["id"], query_date=day, source_metadata_path=str(source.resolve()),
                       source_metadata_sha256=sha(source), phase_at_pilot=solar_phase(rec["granule_start_utc"], rec["granule_end_utc"], area),
                       cabauw_reference_only=area["id"] == "cabauw")
            records.append(rec)
        print(json.dumps({"pilot": area["id"], "product": product, "date": day, "granules": len(records)}), flush=True)
        return records
    with ThreadPoolExecutor(max_workers=4) as pool: parts = list(pool.map(query, tasks))
    records = [r for part in parts for r in part]
    save(output/"inventory.json", {"source_sha256": sha(__file__), "areas_sha256": sha(areas_path), "products": PRODUCTS,
         "metadata_queries": len(tasks), "metadata_only": True, "thermal_assets_opened": False, "years": [2021, 2022, 2023], "records": records})
    return records


def companion(record, output):
    t = record["granule_start_utc"]
    source = Path(output)/"public_metadata/MOD03"/(record["acquisition_key"].replace(":", "_")+".json")
    data = fetch({"collection_concept_id": PRODUCTS["MOD03"]["collection_id"], "temporal": t+","+t}, source)
    matches = [normalize(i) for i in data["items"]]
    matches = [r for r in matches if r["granule_start_utc"] == record["granule_start_utc"] and r["granule_end_utc"] == record["granule_end_utc"]]
    if not matches: raise ValueError("No exact acquisition/version MOD03 companion.")
    matches.sort(key=lambda r: (r["production_tag"], r["cmr_revision"]), reverse=True)
    return {**matches[0], "source_metadata_path": str(source.resolve()), "source_metadata_sha256": sha(source)}


def freeze_plan(output, areas_path):
    output = Path(output); path = output/"engineering_plan.json"
    if path.exists(): raise FileExistsError("Engineering plan already frozen.")
    source = output/"inventory.json"; data = json.loads(source.read_text()); selected = []
    for region in ("greater_london", "sioux_falls"):
        for day in ("2021-07-15", "2022-01-15"):
            for product in ("MOD21", "VNP21"):
                for phase in ("day", "night"):
                    choices = [r for r in data["records"] if r["region_id"] == region and r["query_date"] == day and r["product"] == product and r["phase_at_pilot"] == phase and utc(r["granule_start_utc"]).strftime("%Y-%m-%d") == day]
                    # Metadata-only deterministic selection: closest to local noon/midnight, never LST or native QA.
                    if not choices: raise ValueError(f"No metadata candidate for {region}/{day}/{product}/{phase}.")
                    def rank(r):
                        area = next(a for a in json.loads(Path(areas_path).read_text())["areas"] if a["id"] == region)
                        west, _, east, _ = region_bbox(area); stamp = utc(r["granule_start_utc"])
                        hour = (stamp.hour+stamp.minute/60+(west+east)/30)%24; target = 12 if phase == "day" else 0
                        delta = abs((hour-target+12)%24-12)
                        return delta, r["granule_start_utc"], -int(r["production_tag"]), -r["cmr_revision"]
                    record = dict(min(choices, key=rank)); record["requested_phase"] = phase
                    if product == "MOD21": record["geolocation_companion"] = companion(record, output)
                    selected.append(record)
    total = sum(r["estimated_source_bytes"]+r.get("geolocation_companion", {}).get("estimated_source_bytes", 0) for r in selected)
    if total > 1536*1024**2: raise ValueError("Metadata estimate leaves insufficient headroom under the 2 GiB engineering cap.")
    plan = {"version": "coarse-native-engineering-v1", "source_sha256": sha(__file__), "inventory_sha256": sha(source),
            "areas_sha256": sha(areas_path), "max_total_protected_bytes": 2*1024**3, "max_download_workers": 2,
            "max_file_bytes": {"MOD21": 32*1024**2, "MOD03": 64*1024**2, "VNP21": 192*1024**2},
            "max_granules_per_instrument": 8, "estimated_protected_bytes": total, "thermal_years": [2021, 2022],
            "purpose": "Native-area context and aggregate calibration engineering only; never independent 100 m labels", "records": selected}
    save(path, plan); print(json.dumps({"plan": str(path), "sha256": sha(path), "granules": len(selected), "estimated_bytes": total}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("mode", choices=["inventory", "plan"])
    p.add_argument("--areas", default="pilot/areas_resolved.json"); p.add_argument("--output", required=True)
    a = p.parse_args(); inventory(a.areas, a.output) if a.mode == "inventory" else freeze_plan(a.output, a.areas)
