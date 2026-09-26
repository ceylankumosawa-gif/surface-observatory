"""Narrow, documented legacy-ISD fallback for Darwin International only.

This stage leaves GHCNh parsing and its unknown source401/QC0 rejection intact.
It reads the four separately audited 2021–2024 airport files, accepts only ISD
temperature QC1, and uses the existing backward station/ERA5 residual method.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from pyproj import Geod

from .weather import enrich_weather

VERSION = "darwin-legacy-isd-v1"
REGION_ID = "darwin_howard_springs"
STATION_ID = "94120099999"
AIR_SOURCE = "observed_legacy_ISD_station_residual_plus_ERA5_spatial_background"
MANUAL_URL = "https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf"
ERA5_ONLY_SOURCE = "ERA5_only_no_timely_station"


def file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_isd_csv(path):
    """Read only this airport; TMP is tenths °C and QC1 means all checks passed."""
    path = Path(path)
    required = {"STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "TMP", "SOURCE"}
    raw = pd.read_csv(path, dtype=str, keep_default_na=False,
                      usecols=lambda name: name in required | {"NAME", "REPORT_TYPE"})
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing legacy ISD columns: {sorted(required-set(raw.columns))}")
    if raw.empty or not raw.STATION.str.strip().eq(STATION_ID).all():
        raise ValueError(f"This adapter accepts only nonempty Darwin airport {STATION_ID} files.")
    temp = raw.TMP.str.strip().str.extract(r"^([+-]\d{4}),([A-Za-z0-9])$")
    numeric = pd.to_numeric(temp[0], errors="coerce")
    temperature = numeric.mask(numeric.eq(9999)) / 10.0
    timestamp = pd.to_datetime(raw.DATE, utc=True, errors="coerce")
    years = timestamp.dropna().dt.year.unique()
    if len(years) != 1:
        raise ValueError("The documented fallback expects one year per raw airport CSV.")
    year = int(years[0])
    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{STATION_ID}.csv"
    digest = file_digest(path)
    sidecar = path.with_suffix(".provenance.json")
    retrieval = json.loads(sidecar.read_text()) if sidecar.exists() else None
    if retrieval and (retrieval.get("sha256") != digest or retrieval.get("url") != url):
        raise ValueError(f"Raw airport file does not match its documented URL/hash provenance: {path}")
    result = pd.DataFrame({
        "isd_observation_datetime_utc": timestamp,
        "isd_temperature_c": temperature, "isd_quality_code": temp[1],
        "isd_station_latitude": pd.to_numeric(raw.LATITUDE, errors="coerce"),
        "isd_station_longitude": pd.to_numeric(raw.LONGITUDE, errors="coerce"),
        "isd_station_elevation_m": pd.to_numeric(raw.ELEVATION, errors="coerce"),
        "isd_source_code": raw.SOURCE.str.strip(),
        "isd_report_type": raw.get("REPORT_TYPE", pd.Series("", index=raw.index)).str.strip(),
        "isd_raw_url": url, "isd_raw_sha256": digest,
    })
    usable = (result.isd_quality_code.eq("1") & timestamp.notna() & temperature.between(-93.2, 61.8)
              & result.isd_station_latitude.between(-90, 90) & result.isd_station_longitude.between(-180, 180))
    result.loc[~result.isd_station_elevation_m.between(-500, 9000), "isd_station_elevation_m"] = np.nan
    provenance = {"path": str(path.resolve()), "url": url, "sha256": digest,
                  "year": year, "raw_rows": len(raw), "usable_qc1_rows": int(usable.sum()),
                  "temperature_quality_counts": temp[1].fillna("malformed").value_counts().to_dict(),
                  "retrieval": retrieval, "manual_url": MANUAL_URL,
                  "temperature_decode": "Signed TMP integer /10 °C; +9999 missing; only QC1 accepted."}
    return result.loc[usable].sort_values("isd_observation_datetime_utc"), provenance


def load_isd_archive(raw_dir, years):
    frames, provenance, missing = [], [], []
    for year in sorted(set(map(int, years))):
        path = Path(raw_dir) / f"{STATION_ID}_{year}.csv"
        if not path.exists():
            missing.append(year)
            continue
        frame, metadata = parse_isd_csv(path)
        frames.append(frame)
        provenance.append(metadata)
    if not frames:
        raise ValueError(f"No audited Darwin airport files exist for the requested years {list(years)} in {raw_dir}.")
    observations = pd.concat(frames, ignore_index=True).sort_values("isd_observation_datetime_utc", kind="stable")
    duplicates = int(observations.isd_observation_datetime_utc.duplicated().sum())
    observations = observations.drop_duplicates("isd_observation_datetime_utc", keep="last")
    return observations, {"files": provenance, "missing_years": missing,
                          "duplicate_timestamps_resolved": duplicates,
                          "duplicate_policy": "Stable file/row order; last QC1 record at a duplicate timestamp."}


def apply_darwin_fallback(frame, cache_dir, raw_dir, max_age_minutes=90, max_distance_km=100):
    required = {"region_id", "datetime_utc", "latitude", "longitude", "background_air_temperature_c",
                "air_temperature_c", "air_temperature_source", "station_id", "station_distance_km", "station_age_minutes"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing assembled columns: {sorted(required-set(frame.columns))}")
    if max_age_minutes <= 0 or max_distance_km <= 0:
        raise ValueError("Station time/distance limits must be positive.")
    original_index = frame.index
    result = frame.reset_index(drop=True).copy()
    target = result.region_id.eq(REGION_ID)
    audit = {"version": VERSION, "module_sha256": file_digest(__file__), "region_id": REGION_ID,
             "station_id": STATION_ID, "input_rows": len(result), "target_region_rows": int(target.sum()),
             "changed_rows": 0, "other_region_rows_untouched": int((~target).sum()),
             "max_observation_age_minutes": max_age_minutes, "max_station_distance_km": max_distance_km,
             "max_absolute_station_residual_c": 20,
             "quality_policy": "Legacy ISD temperature QC1 only; GHCNh source401/QC0 remains rejected.",
             "selection_policy": "Replace explicit ERA5-only rows, or valid observed-station corrections when this airport is closer. Preserve closer/equal valid stations.",
             "method": "Pixel ERA5 air + observed airport air - ERA5 air at airport observation time; retrospective inputs, not a real-time-availability claim.",
             "manual_url": MANUAL_URL}
    if not target.any():
        audit["status"] = "no_darwin_rows"
        result.index = original_index
        return result, audit
    work = result.loc[target].copy()
    work["_row_position"] = work.index
    work["datetime_utc"] = pd.to_datetime(work.datetime_utc, utc=True, errors="raise")
    years = set(work.datetime_utc.dt.year) | set((work.datetime_utc-pd.Timedelta(minutes=max_age_minutes)).dt.year)
    observations, archive_audit = load_isd_archive(raw_dir, years)
    audit["raw_archive"] = archive_audit
    queries = work[["datetime_utc"]].drop_duplicates().sort_values("datetime_utc")
    matched = pd.merge_asof(queries, observations, left_on="datetime_utc", right_on="isd_observation_datetime_utc",
                            direction="backward", tolerance=pd.Timedelta(minutes=max_age_minutes))
    work = work.merge(matched, on="datetime_utc", how="left", validate="many_to_one", sort=False)
    work["isd_age_minutes"] = (work.datetime_utc-work.isd_observation_datetime_utc).dt.total_seconds()/60
    _, _, distances = Geod(ellps="WGS84").inv(work.longitude.to_numpy(), work.latitude.to_numpy(),
                                              work.isd_station_longitude.to_numpy(), work.isd_station_latitude.to_numpy())
    work["isd_distance_km"] = np.asarray(distances)/1000
    era5_only = work.air_temperature_source.eq(ERA5_ONLY_SOURCE)
    observed_valid = (work.air_temperature_source.astype(str).str.startswith("observed_")
                      & work.station_id.notna() & work.station_id.astype(str).str.strip().ne("")
                      & np.isfinite(work.air_temperature_c)
                      & work.station_distance_km.between(0, max_distance_km)
                      & work.station_age_minutes.between(0, max_age_minutes))
    prefer_airport = era5_only | (observed_valid & work.isd_distance_km.lt(work.station_distance_km))
    timely = work.isd_age_minutes.between(0, max_age_minutes)
    nearby = work.isd_distance_km.le(max_distance_km)
    candidate = prefer_airport & timely & nearby & np.isfinite(work.background_air_temperature_c)
    audit["diagnostic_counts"] = {"explicit_era5_only_rows": int(era5_only.sum()),
                                  "missing_or_stale_airport_observation_rows": int((~timely).sum()),
                                  "outside_airport_distance_rows": int((timely & ~nearby).sum()),
                                  "closer_or_equal_existing_station_rows": int((observed_valid & ~prefer_airport).sum()),
                                  "candidate_rows_before_weather_residual_check": int(candidate.sum())}
    if not candidate.any():
        audit["status"] = "no_eligible_airport_replacements"
        result.index = original_index
        return result, audit
    selected = work.loc[candidate].copy()
    query = selected[["isd_observation_datetime_utc", "isd_station_latitude", "isd_station_longitude"]].drop_duplicates().rename(
        columns={"isd_observation_datetime_utc": "datetime_utc", "isd_station_latitude": "latitude", "isd_station_longitude": "longitude"})
    background, weather_audit = enrich_weather(query, cache_dir, max_requests=50)
    keys = ["isd_observation_datetime_utc", "isd_station_latitude", "isd_station_longitude"]
    background = background[["datetime_utc", "latitude", "longitude", "background_air_temperature_c"]].rename(
        columns={"datetime_utc": keys[0], "latitude": keys[1], "longitude": keys[2], "background_air_temperature_c": "isd_station_background_air_temperature_c"})
    selected = selected.merge(background, on=keys, how="left", validate="many_to_one", sort=False)
    selected["isd_station_air_correction_c"] = selected.isd_temperature_c-selected.isd_station_background_air_temperature_c
    accepted = selected.isd_station_air_correction_c.abs().le(20)
    audit["diagnostic_counts"]["missing_or_outlier_weather_residual_rows"] = int((~accepted).sum())
    selected = selected.loc[accepted].copy()
    selected["isd_corrected_air_temperature_c"] = selected.background_air_temperature_c+selected.isd_station_air_correction_c
    selected["isd_station_id"] = STATION_ID
    selected["isd_air_source"] = AIR_SOURCE
    selected["isd_dataset"] = "NOAA Global Hourly / legacy ISD"
    updates = {"air_temperature_c": "isd_corrected_air_temperature_c", "air_temperature_source": "isd_air_source",
               "station_id": "isd_station_id", "station_distance_km": "isd_distance_km", "station_age_minutes": "isd_age_minutes",
               "station_elevation_m": "isd_station_elevation_m", "station_air_correction_c": "isd_station_air_correction_c",
               "observed_station_air_temperature_c": "isd_temperature_c",
               "station_background_air_temperature_c": "isd_station_background_air_temperature_c",
               "station_observation_datetime_utc": "isd_observation_datetime_utc",
               "station_observation_dataset": "isd_dataset", "station_observation_source_code": "isd_source_code",
               "station_temperature_quality_code": "isd_quality_code", "station_observation_report_type": "isd_report_type",
               "station_raw_url": "isd_raw_url", "station_raw_sha256": "isd_raw_sha256"}
    positions = selected["_row_position"].to_numpy(dtype=int)
    for destination, source in updates.items():
        values = selected[source]
        if destination not in result:
            if pd.api.types.is_datetime64_any_dtype(values.dtype):
                result[destination] = pd.Series(pd.NaT, index=result.index, dtype=values.dtype)
            elif pd.api.types.is_numeric_dtype(values.dtype):
                result[destination] = np.nan
            else:
                result[destination] = pd.Series(pd.NA, index=result.index, dtype="string")
        result.loc[positions, destination] = values.to_numpy()
    audit.update(status="applied" if len(selected) else "all_candidates_failed_residual_check",
                 changed_rows=len(selected), changed_from_era5_only=int(selected.air_temperature_source.eq(ERA5_ONLY_SOURCE).sum()),
                 changed_from_farther_observed_station=int((~selected.air_temperature_source.eq(ERA5_ONLY_SOURCE)).sum()),
                 matched_airport_observation_times=int(selected.isd_observation_datetime_utc.nunique()),
                 weather_requests=weather_audit,
                 target_region_air_sources=result.loc[target, "air_temperature_source"].value_counts().to_dict())
    # Check all pre-existing fields outside Darwin remain exactly unchanged.
    pd.testing.assert_frame_equal(result.loc[~target, frame.columns].reset_index(drop=True),
                                  frame.reset_index(drop=True).loc[~target].reset_index(drop=True), check_dtype=False)
    result.index = original_index
    return result, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--raw-dir", type=Path, default=Path("runs/darwin_qc_audit/legacy_isd"))
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files.")
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError("Output/audit already exists; choose a new output path.")
    data, audit = apply_darwin_fallback(pd.read_parquet(args.input), args.cache, args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + "." + uuid.uuid4().hex + ".partial.parquet")
    try:
        data.to_parquet(temporary, index=False)
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    audit.update(input_path=str(args.input.resolve()), input_sha256=file_digest(args.input),
                 output_path=str(args.output.resolve()), output_sha256=file_digest(args.output))
    audit_path.write_text(json.dumps(audit, indent=2, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "audit": str(audit_path), "status": audit["status"],
                      "changed_rows": audit["changed_rows"], "target_region_air_sources": audit.get("target_region_air_sources", {})}, indent=2), flush=True)


if __name__ == "__main__":
    main()
