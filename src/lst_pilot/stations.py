"""Small NOAA GHCNh station adapter. Downloads are explicit and cache provenance.

GHCNh v1.1.0 (2026-03-10) PSV values are already in physical units. They
must not receive legacy ISD tenths scaling. Observations are UTC and can be
sub-hourly. Precipitation is a nominal hourly/running total: never sum all
sub-hourly reports. This module retains reports and flags, not gap-filled data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network"
INVENTORY_URL = BASE_URL + "/hourly/doc/ghcnh-station-list.csv"
DOCUMENTATION_URL = BASE_URL + "/hourly/doc/ghcnh_DOCUMENTATION.pdf"
VARIABLES = {
    "temperature": "air_temperature_c",
    "dew_point_temperature": "dew_point_c",
    "wind_speed": "wind_speed_m_s",
    "wind_direction": "wind_direction_deg",
    "station_level_pressure": "station_pressure_hpa",
    "sea_level_pressure": "sea_level_pressure_hpa",
    "relative_humidity": "relative_humidity_percent",
    "precipitation": "precipitation_mm",
    "snow_depth": "snow_depth_mm",
}
ATTRIBUTES = ("Measurement_Code", "Quality_Code", "Report_Type", "Source_Code", "Source_Station_ID")
HARMONIZED_QC = {"temperature", "dew_point_temperature", "wind_speed", "wind_direction", "station_level_pressure", "sea_level_pressure"}
ISD_QC_SOURCES = {"313", "314", "315", "322", "335", "343", "344", "346"}
OTHER_QC_SOURCES = {"220", "221", "222", "223", "347", "348"}


def _fetch(url: str, destination: Path, max_bytes: int, refresh: bool = False,
           documentation_url: str = DOCUMENTATION_URL) -> Path:
    """Atomically cache an HTTPS object and URL/time/hash provenance sidecar."""
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with urlopen(Request(url, headers={"User-Agent": "lst-pilot/0.1"}), timeout=90) as response:
            declared = int(response.headers.get("Content-Length", "0"))
            if declared > max_bytes:
                raise ValueError(f"Object is {declared} bytes; bound is {max_bytes}: {url}")
            sha = hashlib.sha256()
            size = 0
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
                temp_path = Path(tmp.name)
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError(f"Download exceeded {max_bytes} bytes: {url}")
                    tmp.write(block)
                    sha.update(block)
            if declared and size != declared:
                raise IOError(f"Truncated response: expected {declared}, received {size}")
            metadata = {"url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                        "bytes": size, "sha256": sha.hexdigest(), "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                        "documentation_url": documentation_url}
        os.replace(temp_path, destination)
        destination.with_suffix(destination.suffix + ".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
    return destination


def station_inventory(cache_dir: str | Path, refresh: bool = False) -> pd.DataFrame:
    """Fetch only the ~2.2 MB geographic inventory; it contains no date coverage."""
    path = _fetch(INVENTORY_URL, Path(cache_dir) / "ghcnh-station-list.csv", 10_000_000, refresh)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).rename(columns={
        "GHCN_ID": "station_id", "LATITUDE": "latitude", "LONGITUDE": "longitude",
        "ELEVATION": "elevation_m", "NAME": "name", "ICAO": "icao", "ISO_CODE": "country_iso"})
    required = {"station_id", "latitude", "longitude", "name"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unexpected NOAA station inventory schema: {list(frame.columns)}")
    for column in ("latitude", "longitude", "elevation_m"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.loc[frame.elevation_m.isin([-999, -999.9, -9999]), "elevation_m"] = float("nan")
    frame.attrs.update(source_url=INVENTORY_URL, coverage_note="Inventory is geographic; presence does not imply observations in any chosen year.")
    return frame


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, a)))


def nearby_stations(inventory: pd.DataFrame, bbox: tuple[float, float, float, float],
                    max_distance_km: float = 100, limit: int | None = 12) -> pd.DataFrame:
    """Rank candidates by center distance, within an approximate buffer of bbox.

    bbox is west,south,east,north in WGS84. Antimeridian-crossing boxes must be
    split by the caller. This geographic screen does not establish data coverage.
    """
    west, south, east, north = map(float, bbox)
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        raise ValueError("Expected ordered WGS84 bbox west,south,east,north")
    if max_distance_km < 0 or (limit is not None and limit < 1):
        raise ValueError("Distance must be nonnegative and limit positive")
    center_lat, center_lon = (south + north) / 2, (west + east) / 2
    result = inventory.dropna(subset=["latitude", "longitude"]).copy()
    result["distance_to_bbox_km"] = [
        _distance_km(lat, lon, min(max(lat, south), north), min(max(lon, west), east))
        for lat, lon in zip(result.latitude, result.longitude)]
    result = result[result.distance_to_bbox_km <= max_distance_km].copy()
    result["distance_to_center_km"] = [
        _distance_km(lat, lon, center_lat, center_lon) for lat, lon in zip(result.latitude, result.longitude)]
    result = result.sort_values(["distance_to_center_km", "station_id"])
    return result.head(limit) if limit is not None else result


def station_year_url(station_id: str, year: int) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", station_id):
        raise ValueError("GHCNh station identifiers must contain 11 alphanumeric/hyphen/underscore characters")
    if not 1700 <= int(year) <= datetime.now(timezone.utc).year:
        raise ValueError("Year is outside supported historical bounds")
    return f"{BASE_URL}/hourly/access/by-year/{int(year)}/psv/GHCNh_{station_id}_{int(year)}.psv"


def station_year_available(station_id: str, year: int) -> dict:
    """Small HEAD request, no station observations downloaded."""
    url = station_year_url(station_id, year)
    try:
        with urlopen(Request(url, method="HEAD"), timeout=30) as response:
            return {"station_id": station_id, "year": int(year), "available": True,
                    "bytes": int(response.headers.get("Content-Length", "0")), "url": url}
    except HTTPError as exc:
        if exc.code == 404:
            return {"station_id": station_id, "year": int(year), "available": False, "url": url}
        raise


def download_station_year(station_id: str, year: int, cache_dir: str | Path,
                          refresh: bool = False, max_bytes: int = 250_000_000) -> Path:
    """Explicit bounded station-year download (~several to tens of MB/site-year)."""
    url = station_year_url(station_id, year)
    return _fetch(url, Path(cache_dir) / str(year) / url.rsplit("/", 1)[-1], max_bytes, refresh)


def quality_usable(variable: str, quality: str, source: str) -> bool:
    """Conservative source-aware QC; unknown nonblank codes are rejected.

    For the six integrated-QC variables, blank means no harmonized failure flag.
    Legacy pass codes vary by source: source 223 code 0 is NOT checked, while
    source 313 code 0 passes gross limits. Accepted gross-check-only readings
    retain their raw codes and can be tightened downstream. Edited/suspect codes
    are excluded. Blank on other fields is retained as unflagged, not certified.
    """
    quality, source = str(quality).strip(), str(source).strip()
    if quality == "":
        return True
    if source in ISD_QC_SOURCES:
        return quality in {"0", "1", "4", "5", "9"}
    if source in OTHER_QC_SOURCES:
        return quality == "1"
    if source == "345" and variable in {"relative_humidity", "wind_speed"}:
        return quality == "0"
    # Source 382 any nonblank flag denotes a problem or non-hourly accumulation.
    return False


def parse_station_file(path: str | Path, keep_flags: bool = True) -> pd.DataFrame:
    """Parse physical units with QC masking; preserve sub-hourly timestamps.

    No nearest-hour rounding, interpolation, accumulation, or pressure-type
    substitution occurs. *_raw retains values before QC; *_usable indicates
    acceptance, and metadata is kept by default for auditing. Trace rain is
    explicitly flagged, not conflated with confidently dry observations.
    """
    metadata_columns = {"STATION", "Station_name", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION"}
    selected = metadata_columns | set(VARIABLES)
    for variable in VARIABLES:
        selected.update(f"{variable}_{attribute}" for attribute in ATTRIBUTES)
    raw = pd.read_csv(path, sep="|", dtype=str, keep_default_na=False,
                      usecols=lambda name: name in selected)
    if not {"STATION", "DATE", "temperature"}.issubset(raw.columns):
        raise ValueError("Not a supported GHCNh PSV: STATION, DATE, temperature required")
    result = pd.DataFrame({"station_id": raw.STATION,
                           "timestamp_utc": pd.to_datetime(raw.DATE, utc=True, errors="coerce")})
    for old, new in {"Station_name": "station_name", "LATITUDE": "latitude", "LONGITUDE": "longitude", "ELEVATION": "elevation_m"}.items():
        if old in raw:
            result[new] = raw[old] if old == "Station_name" else pd.to_numeric(raw[old], errors="coerce")
    empty = pd.Series("", index=raw.index, dtype=str)
    for variable, output in VARIABLES.items():
        values = pd.to_numeric(raw.get(variable, empty), errors="coerce")
        values = values.mask(values.isin([-9999, -999.9, 9999, 99999]))
        quality = raw.get(variable + "_Quality_Code", empty)
        source = raw.get(variable + "_Source_Code", empty)
        measure = raw.get(variable + "_Measurement_Code", empty)
        accepted = pd.Series([quality_usable(variable, q, s) for q, s in zip(quality, source)], index=raw.index)
        if variable in {"precipitation", "snow_depth", "wind_speed"}:
            accepted &= values.ge(0)
        if variable == "precipitation":
            # Incomplete, estimated, failed, and multi-hour accumulation markers
            # must not masquerade as ordinary measured hourly rain.
            known_measure_sources = source.isin(ISD_QC_SOURCES | OTHER_QC_SOURCES)
            accepted &= ~(known_measure_sources & measure.isin(["1", "3", "4", "5", "6", "7", "8", "E", "I", "J"]))
            result["precipitation_trace"] = measure.isin(["T", "2"])
            accepted &= ~(source.eq("382") & ~measure.isin(["", "T", "1"]))
        if variable == "relative_humidity":
            accepted &= values.between(0, 100)
        if variable == "wind_direction":
            accepted &= values.between(0, 360)
        result[output] = values.where(accepted)
        if keep_flags:
            result[output + "_raw"] = values
            result[output + "_usable"] = accepted & values.notna()
            for attribute in ATTRIBUTES:
                result[f"{output}_{attribute.lower()}"] = raw.get(variable + "_" + attribute, empty)
    result = result.dropna(subset=["timestamp_utc"]).sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    result.attrs.update(documentation_url=DOCUMENTATION_URL,
                        precipitation_warning="Sub-hourly precipitation may be running totals. Never sum reports indiscriminately.",
                        qc_policy="Reject known failures/unknown nonblank codes; preserve unflagged and source-specific passed checks.",
                        source_file=str(Path(path).resolve()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/stations")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    nearby = sub.add_parser("nearby")
    nearby.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    nearby.add_argument("--distance-km", type=float, default=100)
    nearby.add_argument("--limit", type=int, default=12)
    nearby.add_argument("--year", type=int, help="Probe file existence for candidates without downloading observations")
    download = sub.add_parser("download")
    download.add_argument("station_id")
    download.add_argument("year", type=int)
    parse = sub.add_parser("parse")
    parse.add_argument("path")
    parse.add_argument("--output", required=True, help=".csv or .parquet")
    args = parser.parse_args()
    if args.command in {"inventory", "nearby"}:
        frame = station_inventory(args.cache)
        if args.command == "nearby":
            frame = nearby_stations(frame, tuple(args.bbox), args.distance_km, args.limit)
            if args.year:
                available = [station_year_available(station, args.year) for station in frame.station_id]
                frame = frame.merge(pd.DataFrame(available), on="station_id", how="left") if available else frame
        print(frame.to_csv(index=False), end="")
    elif args.command == "download":
        print(download_station_year(args.station_id, args.year, args.cache))
    else:
        frame = parse_station_file(args.path)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".parquet":
            frame.to_parquet(output, index=False)
        else:
            frame.to_csv(output, index=False)
        output.with_suffix(output.suffix + ".schema.json").write_text(json.dumps(frame.attrs, indent=2) + "\n")
        print(json.dumps({"output": str(output), "rows": len(frame), "usable_air_temperature": int(frame.air_temperature_c.notna().sum())}))


if __name__ == "__main__":
    main()
