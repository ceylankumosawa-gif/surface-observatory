"""Bounded recent METAR reports plus explicit same-provider background residual.

This is an operational observation adapter, not the historical GHCNh QC parser.
It verifies temperature, station and observation time against the raw METAR;
publisher QC bits are retained without claiming undocumented flag meanings.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pyproj import Geod

from lst_pilot.stations import station_inventory, nearby_stations
from lst_pilot.assemble import EXCLUDED_AIR_STATION_IDS, region_bbox
from . import weather_access as access

METAR_URL = "https://aviationweather.gov/api/data/metar"
STATION_OK = "verified_metar_raw_temperature_and_gfs_background"
GEOD = Geod(ellps="WGS84")


def verified_metar(report, target, allowed_ids):
    """Return an independently checked parsed report or an explicit reason."""
    try:
        station = report["icaoId"]
        raw = report["rawOb"]
        stamp = pd.Timestamp(int(report["obsTime"]), unit="s", tz="UTC")
        target = access._utc(target)
        if station not in allowed_ids or not re.search(r"(?:^|\s)"+re.escape(station)+r"\s", raw):
            return None, "station_identity_mismatch"
        time_match = re.search(r"\b(\d{2})(\d{2})(\d{2})Z\b", raw)
        if not time_match or list(map(int,time_match.groups())) != [stamp.day,stamp.hour,stamp.minute]:
            return None, "raw_observation_time_mismatch"
        if not pd.Timedelta(0) <= target-stamp <= pd.Timedelta(minutes=90):
            return None, "not_timely_backward_report"
        # Prefer the optional signed tenths-degree temperature remark.
        precise = re.search(r"\bT([01])(\d{3})(?:[01]\d{3})?\b", raw)
        coarse = re.search(r"(?:^|\s)(M?\d{2})/(?:M?\d{2}|//)?(?:\s|$)", raw)
        if precise:
            decoded = (-1 if precise[1]=="1" else 1)*int(precise[2])/10
            precision = "0.1C_T_remark"
        elif coarse:
            decoded = -float(coarse[1][1:]) if coarse[1].startswith("M") else float(coarse[1])
            precision = "1C_report_body"
        else:
            return None, "raw_temperature_unavailable"
        reported = float(report["temp"])
        if not np.isfinite(reported) or not -100 <= reported <= 65 or abs(reported-decoded)>1e-8:
            return None, "raw_temperature_mismatch_or_out_of_range"
        lat, lon = float(report["lat"]), float(report["lon"])
        if not np.isfinite([lat,lon]).all() or not -90<=lat<=90 or not -180<=lon<=180:
            return None, "invalid_report_coordinates"
        return {"station_id":station, "observation_time_utc":stamp, "air_temperature_c":reported,
                "latitude":lat,"longitude":lon,"elevation_m":float(report.get("elev",np.nan)),
                "raw_metar":raw, "precision":precision, "publisher_qc_field":report.get("qcField"),
                "receipt_time_utc":report.get("receiptTime")}, "verified"
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "invalid_report_schema"


def attach_recent_stations(frame, areas, cache, plan, *, stations_per_region=2, max_distance_km=100,
                           max_age_minutes=90, http=None):
    """Use at most two actual timely reports, selected from fixed tile candidates.

    Missing reports leave air_temperature_c NaN. Never invent a station value or
    borrow another day/year. Station background uses the same GFS delivery path,
    at floor(observation time), matching the historical backward-hour convention.
    """
    access._check_frame(frame,plan)
    if plan["source"] != "experimental_gfs":
        raise ValueError("Recent METAR/GFS adapter requires the explicit operational plan")
    if stations_per_region != 2 or max_distance_km != 100 or max_age_minutes != 90:
        raise ValueError("Current reviewed station limits are fixed at2/100km/90minutes")
    if frame.region_id.nunique() != 1:
        raise ValueError("One canonical region per bounded station request")
    target = access._utc(plan["valid_time_utc"])
    if access._utc(plan["availability"]["checked_utc"])-target > pd.Timedelta(days=7):
        raise ValueError("Recent station query exceeds the7-day product bound")
    cache = Path(cache)
    inventory = station_inventory(cache/"stations")
    inventory_path = cache/"stations"/"ghcnh-station-list.csv"
    area_map = {a["id"]:a for a in areas}
    area = area_map[str(frame.region_id.iloc[0])]
    candidates = nearby_stations(inventory,tuple(region_bbox(area)),max_distance_km=100,limit=16)
    candidates = candidates.loc[~candidates.station_id.isin(EXCLUDED_AIR_STATION_IDS) & candidates.icao.astype(str).str.fullmatch(r"[A-Z0-9]{4}")].copy()
    candidates = candidates.sort_values(["distance_to_center_km","station_id"],kind="stable").drop_duplicates("icao")
    output = frame.copy().reset_index(drop=True)
    output["air_temperature_c"] = np.nan
    output["air_temperature_source"] = "missing_verified_recent_station"
    output["station_id"] = ""
    output["verified_station_report_status"] = "missing_timely_verified_METAR"
    output["verified_station_observation_datetime_utc"] = pd.Series(pd.NaT,index=output.index,dtype="datetime64[ns, UTC]")
    for column in ["station_distance_km","station_age_minutes","station_elevation_m","station_air_correction_c", "observed_station_air_temperature_c", "station_background_air_temperature_c"]:
        output[column] = np.nan
    receipt = {"version":"global_recent_station_v1", "source":METAR_URL, "is_station_observation":True,
               "inventory":{"path":str(inventory_path.resolve()),"sha256":access._sha(inventory_path.read_bytes())},
               "candidate_icao_ids":candidates.icao.tolist(), "max_used_stations":2, "max_report_age_minutes":90,
               "max_distance_km":100, "station_residual_limit_c":20, "reports":[], "rejections":[],
               "quality_scope":"Raw report temperature/time/identity verified; publisher qcField retained but not decoded as GHCNh QC",
               "missing_station_policy":"air input missing; downstream must mask prediction"}
    if candidates.empty:
        receipt["status_counts"] = output.verified_station_report_status.value_counts().to_dict()
        return output,receipt
    http = http or access.BoundedHTTP(max_requests=3,max_bytes=1_000_000)
    params = {"ids":",".join(candidates.icao), "format":"json", "hours":2,
              "date":target.strftime("%Y-%m-%dT%H:%M:%SZ")}
    key = access._sha(json.dumps({"request":params,"checked_utc":plan["availability"]["checked_utc"]},sort_keys=True).encode())
    path = cache/"weather_access"/"metar"/(key+".json")
    if not path.exists():
        payload = http.get(METAR_URL,params=params,limit=400_000,allow_empty=True)
        reports = json.loads(payload) if payload else []
        if not isinstance(reports,list) or len(reports)>=400:
            raise ValueError("METAR result is invalid or may have hit the400-record truncation limit")
        record = {"request":params,"retrieved_utc":pd.Timestamp.now(tz="UTC").isoformat(), "source":METAR_URL,
                  "reports":reports,"raw_response_sha256":access._sha(payload)}
        access._atomic(path,(json.dumps(record,sort_keys=True)+"\n").encode())
    record = json.loads(path.read_text())
    if record["request"]!=params or record["source"]!=METAR_URL:
        raise ValueError("METAR cache request/source mismatch")
    receipt["response"] = {"path":str(path.resolve()),"sha256":access._sha(path.read_bytes())}
    parsed = []
    for raw in record["reports"]:
        verified,reason = verified_metar(raw,target,set(candidates.icao))
        if verified:
            parsed.append(verified)
        else:
            receipt["rejections"].append({"icao":raw.get("icaoId"),"obsTime":raw.get("obsTime"),"reason":reason})
    best_distance = np.full(len(output),np.inf)
    used_candidates = 0
    for _,candidate in candidates.iterrows():
        matching = [r for r in parsed if r["station_id"]==candidate.icao]
        if not matching:
            continue
        # Corrected same-time reports use latest publisher receipt, independently of temperature.
        report = max(matching,key=lambda r:(r["observation_time_utc"],r.get("receipt_time_utc") or ""))
        _,_,inventory_distance = GEOD.inv(float(candidate.longitude),float(candidate.latitude),report["longitude"],report["latitude"])
        if inventory_distance > 5000:
            receipt["rejections"].append({"icao":candidate.icao,"reason":"inventory_report_coordinates_differ_over5km"})
            continue
        background_plan = dict(plan, requested_time_utc=report["observation_time_utc"].floor("h").isoformat(),
                               valid_time_utc=report["observation_time_utc"].floor("h").isoformat())
        point = pd.DataFrame({"datetime_utc":[report["observation_time_utc"].floor("h")],
                              "latitude":[report["latitude"]],"longitude":[report["longitude"]]})
        background,binding = access.prepare_background(point,cache,background_plan,max_requests=1,http=http)
        base = float(background.background_air_temperature_c.iloc[0])
        residual = report["air_temperature_c"]-base
        if not np.isfinite(residual) or abs(residual)>20:
            receipt["rejections"].append({"icao":candidate.icao,"reason":"invalid_or_over20C_station_residual"})
            continue
        _,_,distance = GEOD.inv(output.longitude.to_numpy(),output.latitude.to_numpy(),
                               np.full(len(output),report["longitude"]),np.full(len(output),report["latitude"]))
        distance = np.asarray(distance)/1000
        eligible = distance<=100
        if not eligible.any():
            continue
        use = eligible & (distance<best_distance)
        output.loc[use,"air_temperature_c"] = output.loc[use,"background_air_temperature_c"]+residual
        output.loc[use,"air_temperature_source"] = "observed_METAR_residual_plus_GFS_spatial_background"
        output.loc[use,"station_id"] = str(candidate.icao)
        output.loc[use,"station_distance_km"] = distance[use]
        output.loc[use,"station_age_minutes"] = (target-report["observation_time_utc"]).total_seconds()/60
        output.loc[use,"station_elevation_m"] = report["elevation_m"]
        output.loc[use,"station_air_correction_c"] = residual
        output.loc[use,"observed_station_air_temperature_c"] = report["air_temperature_c"]
        output.loc[use,"station_background_air_temperature_c"] = base
        output.loc[use,"verified_station_report_status"] = STATION_OK
        output.loc[use,"verified_station_observation_datetime_utc"] = report["observation_time_utc"]
        best_distance[use] = distance[use]
        receipt["reports"].append({**report,"observation_time_utc":report["observation_time_utc"].isoformat(),
                                    "background":binding,"background_air_temperature_c":base,"residual_c":residual})
        used_candidates += 1
        if used_candidates >= 2:
            break
    receipt["http"] = http.receipts
    receipt["status_counts"] = output.verified_station_report_status.value_counts().to_dict()
    return output,receipt
