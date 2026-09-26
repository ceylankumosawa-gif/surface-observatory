"""Optional bounded ERA5 longwave and snow context from public Google ARCO.

Only two single-level variables are downloaded: no pressure-level fields and
no skin temperature. Each array chunk is one global 0.25-degree hour, ~4.15 MB
decoded. Samples share cached chunks. Failures yield NaN plus explicit statuses.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

STORE = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SOURCE_URL = "gs://" + STORE
DOC_URL = "https://github.com/google-research/arco-era5"
TIME_DOC_URL = "https://confluence.ecmwf.int/pages/viewpage.action?pageId=216495456"
VARIABLES = {
    "mean_surface_downward_long_wave_radiation_flux": "era5_longwave_down_w_m2",
    "snow_depth": "era5_snow_water_equivalent_m",
}
EXPECTED_UNITS = {
    "mean_surface_downward_long_wave_radiation_flux": "W m**-2",
    "snow_depth": "m of water equivalent",
}
DECODED_SURFACE_BYTES = 721 * 1440 * 4
MAX_OBJECT_BYTES = DECODED_SURFACE_BYTES + 65536
MAX_METADATA_BYTES = 1_000_000
EPOCH = pd.Timestamp("1900-01-01", tz="UTC")


def _publish_public_cache(destination: Path, payload: bytes, cache: Path) -> None:
    """Atomically share public ARCO bytes with the cache's existing Unix group.

    NamedTemporaryFile otherwise publishes mode 0600, blocking the web reader
    after a research process refreshes metadata. This policy applies only to
    this public-data cache, never to Earthdata credentials or general files.
    """
    cache.mkdir(parents=True, exist_ok=True)
    root = cache.resolve()
    if not destination.resolve().is_relative_to(root):
        raise ValueError("Public ARCO cache destination must remain inside its cache.")
    group = root.stat().st_gid
    missing = []
    parent = destination.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        os.chown(directory, -1, group)
        directory.chmod(0o2775)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
            os.fchown(tmp.fileno(), -1, group)
            os.fchmod(tmp.fileno(), 0o664)
            tmp.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cached_object(fs, key: str, cache: Path, limit: int, manifest: list) -> bytes:
    if Path(key).is_absolute() or ".." in Path(key).parts:
        raise ValueError("Public ARCO object keys must be relative to their cache.")
    destination = cache / key
    # Coverage metadata advances as new ERA5 hours are published. Data chunks
    # remain immutable; refresh this small index daily for arbitrary-date use.
    expired = key == ".zmetadata" and destination.exists() and time.time()-destination.stat().st_mtime > 86400
    if destination.exists() and not expired:
        size = destination.stat().st_size
        if size > limit:
            raise ValueError(f"Cached object exceeds byte limit: {key}")
        payload = destination.read_bytes()
        manifest.append({"key": key, "bytes": size, "cached": True,
                         "sha256": hashlib.sha256(payload).hexdigest()})
        return payload
    info = fs.info(STORE + "/" + key)
    size = int(info["size"])
    if size > limit:
        raise ValueError(f"Object exceeds {limit} byte bound ({size} bytes): {key}")
    # Explicit single-object GET; never invoke a directory or whole-array fetch.
    payload = fs.cat_file(STORE + "/" + key)
    if len(payload) != size or len(payload) > limit:
        raise IOError(f"Object size differs from metadata: {key}")
    _publish_public_cache(destination, payload, cache)
    record = {"key": key, "bytes": size, "cached": False,
              "sha256": hashlib.sha256(payload).hexdigest(),
              "retrieved_utc": datetime.now(timezone.utc).isoformat(),
              "source_url": SOURCE_URL + "/" + key,
              "generation": str(info.get("generation", "")), "etag": info.get("etag")}
    _publish_public_cache(destination.with_name(destination.name + ".provenance.json"),
                          (json.dumps(record, indent=2) + "\n").encode(), cache)
    manifest.append(record)
    return payload


def _decode(payload: bytes, description: dict) -> np.ndarray:
    from numcodecs import get_codec
    if description.get("filters"):
        raise ValueError("Unexpected Zarr filters; refusing to guess their decoding")
    decoded = get_codec(description["compressor"]).decode(payload) if description.get("compressor") else payload
    shape = description["chunks"]
    dtype = np.dtype(description["dtype"])
    expected = int(np.prod(shape)) * dtype.itemsize
    if len(decoded) != expected:
        raise ValueError(f"Decoded chunk size is {len(decoded)} not expected {expected}")
    return np.frombuffer(decoded, dtype=dtype).reshape(shape, order=description.get("order", "C"))


def _validate_metadata(metadata: dict) -> None:
    for variable in VARIABLES:
        arr = metadata[variable + "/.zarray"]
        attrs = metadata[variable + "/.zattrs"]
        if arr["chunks"] != [1, 721, 1440] or arr["shape"][1:] != [721, 1440]:
            raise ValueError(f"Unexpected surface shape/chunking for {variable}")
        if np.dtype(arr["dtype"]).itemsize != 4 or np.dtype(arr["dtype"]).kind != "f" or arr.get("zarr_format") != 2:
            raise ValueError(f"Unexpected dtype or storage format for {variable}")
        if attrs.get("_ARRAY_DIMENSIONS") != ["time", "latitude", "longitude"]:
            raise ValueError(f"Unexpected dimension order for {variable}")
        if attrs.get("units") != EXPECTED_UNITS[variable]:
            raise ValueError(f"Unexpected units for {variable}: {attrs.get('units')}")
    if metadata["time/.zattrs"].get("units") != "hours since 1900-01-01 00:00:00":
        raise ValueError("Unknown ARCO time indexing")


def _grid_indices(latitude, longitude) -> tuple[np.ndarray, np.ndarray]:
    """Nearest 0.25-degree grid point; longitude wraps across Greenwich/dateline."""
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    if np.any(~np.isfinite(lat) | ~np.isfinite(lon) | (lat < -90) | (lat > 90)):
        raise ValueError("Invalid sample coordinates")
    row = np.floor((90 - lat) * 4 + 0.5).astype(int).clip(0, 720)
    col = np.floor((lon % 360) * 4 + 0.5).astype(int) % 1440
    return row, col


def add_radiation(frame: pd.DataFrame, cache: str | Path, max_hours: int = 128,
                  timestamp_col: str | None = None, allow_provisional: bool = False) -> pd.DataFrame:
    """Append nearest-grid ERA5 context at the last completed UTC hour.

    Required columns: latitude, longitude and datetime_utc or timestamp_utc.
    Longwave is the preceding-hour mean ending at floor(sample time); snow is
    instantaneous at that hour and is METRES WATER EQUIVALENT, not snow height.
    If max_hours is exceeded, no remote reads occur and all rows receive a
    budget-exceeded status. Individual access failures remain explicit NaNs.
    Cache is a directory for this adapter; it stores compressed global chunks.
    """
    if max_hours < 1:
        raise ValueError("max_hours must be positive")
    timestamp_col = timestamp_col or next((name for name in ["datetime_utc", "timestamp_utc"] if name in frame), None)
    if timestamp_col is None or not {"latitude", "longitude"}.issubset(frame.columns):
        raise ValueError("Expected latitude, longitude and datetime_utc/timestamp_utc")
    result = frame.copy()
    timestamps = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce").dt.floor("h")
    result["radiation_era5_time_utc"] = timestamps
    for output in VARIABLES.values():
        result[output] = np.nan
        result[output + "_status"] = "not_requested"
    result["radiation_era5_latitude"] = np.nan
    result["radiation_era5_longitude"] = np.nan
    latitudes = pd.to_numeric(frame.latitude, errors="coerce")
    longitudes = pd.to_numeric(frame.longitude, errors="coerce")
    valid = timestamps.notna() & latitudes.between(-90, 90) & longitudes.between(-180, 360)
    times = sorted(timestamps[valid].unique())
    report = {"source_url": SOURCE_URL, "documentation_url": DOC_URL, "temporal_documentation_url": TIME_DOC_URL,
              "resolution_degrees": 0.25, "sampling": "nearest grid point; last completed hour, no spatial or temporal interpolation",
              "longwave_time_support": "mean over one hour ending at radiation_era5_time_utc",
              "snow_units": "m of water equivalent; not physical snow height",
              "is_observation": False, "variables": VARIABLES, "requested_hours": len(times),
              "max_hours": max_hours, "maximum_decoded_data_bytes": len(times) * len(VARIABLES) * DECODED_SURFACE_BYTES,
              "errors": [], "objects": []}
    for output in VARIABLES.values():
        result.loc[~valid, output + "_status"] = "invalid_time_or_coordinate"
    if len(times) > max_hours:
        for output in VARIABLES.values():
            result.loc[valid, output + "_status"] = "hour_budget_exceeded"
        report["errors"].append(f"{len(times)} hours exceeds cap {max_hours}; no remote reads made")
        result.attrs["radiation_context"] = report
        return result
    if not times:
        result.attrs["radiation_context"] = report
        return result
    cache = Path(cache) / "arco-era5-v3"
    try:
        import fsspec
        fs = fsspec.filesystem("gcs", token="anon", timeout=30)
        metadata = json.loads(_cached_object(fs, ".zmetadata", cache, MAX_METADATA_BYTES, report["objects"]))["metadata"]
        _validate_metadata(metadata)
        report["dataset_attributes"] = metadata[".zattrs"]
        # Verify small coordinate arrays rather than trusting hard-coded indexing.
        for name, expected in [("latitude", np.linspace(90, -90, 721)), ("longitude", np.arange(1440) / 4)]:
            values = _decode(_cached_object(fs, name + "/0", cache, 65536, report["objects"]), metadata[name + "/.zarray"])
            if not np.allclose(values, expected):
                raise ValueError(f"Unexpected {name} grid")
        start = pd.Timestamp(metadata[".zattrs"]["valid_time_start"], tz="UTC")
        final_stop = pd.Timestamp(metadata[".zattrs"]["valid_time_stop"], tz="UTC") + pd.Timedelta(hours=23)
        stop_key = "valid_time_stop_era5t" if allow_provisional and metadata[".zattrs"].get("valid_time_stop_era5t") else "valid_time_stop"
        stop = pd.Timestamp(metadata[".zattrs"][stop_key], tz="UTC") + pd.Timedelta(hours=23)
        report["provisional_era5t_allowed"] = bool(allow_provisional)
    except Exception as exc:
        for output in VARIABLES.values():
            result.loc[valid, output + "_status"] = "metadata_unavailable"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        result.attrs["radiation_context"] = report
        return result
    verified_time_chunks = {}
    for timestamp in times:
        selected = valid & timestamps.eq(timestamp)
        if timestamp < start or timestamp > stop:
            for output in VARIABLES.values():
                result.loc[selected, output + "_status"] = "outside_final_era5_coverage"
            continue
        hour_index = int((timestamp - EPOCH) / pd.Timedelta(hours=1))
        try:
            time_description = metadata["time/.zarray"]
            time_chunk_size = time_description["chunks"][0]
            time_chunk_number = hour_index // time_chunk_size
            if time_chunk_number not in verified_time_chunks:
                time_values = _decode(_cached_object(fs, f"time/{time_chunk_number}", cache,
                                                    MAX_METADATA_BYTES, report["objects"]), time_description)
                verified_time_chunks[time_chunk_number] = time_values
            if verified_time_chunks[time_chunk_number][hour_index % time_chunk_size] != hour_index:
                raise ValueError("ARCO time coordinate does not match expected hour-index mapping")
        except Exception as exc:
            for output in VARIABLES.values():
                result.loc[selected, output + "_status"] = "time_index_unavailable"
            report["errors"].append(f"time {hour_index}: {type(exc).__name__}: {exc}")
            continue
        rows, cols = _grid_indices(latitudes[selected], longitudes[selected])
        result.loc[selected, "radiation_era5_latitude"] = 90 - rows / 4
        result.loc[selected, "radiation_era5_longitude"] = ((cols / 4 + 180) % 360) - 180
        for variable, output in VARIABLES.items():
            key = f"{variable}/{hour_index}.0.0"
            try:
                payload = _cached_object(fs, key, cache, MAX_OBJECT_BYTES, report["objects"])
                chunk = _decode(payload, metadata[variable + "/.zarray"])
                values = chunk[0, rows, cols].astype(float)
                usable = np.isfinite(values) & (values >= 0 if variable == "snow_depth" else values > 0)
                result.loc[selected, output] = np.where(usable, values, np.nan)
                status = "coarse_provisional_ERA5T" if timestamp > final_stop else "coarse_reanalysis"
                result.loc[selected, output + "_status"] = np.where(usable, status, "missing_or_invalid_value")
            except Exception as exc:
                result.loc[selected, output + "_status"] = "source_unavailable"
                report["errors"].append(f"{key}: {type(exc).__name__}: {exc}")
    report["downloaded_bytes"] = sum(x["bytes"] for x in report["objects"] if not x["cached"])
    report["cached_bytes_used"] = sum(x["bytes"] for x in report["objects"] if x["cached"])
    result.attrs["radiation_context"] = report
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input sample Parquet")
    parser.add_argument("output", help="Separate output Parquet; never overwrites input")
    parser.add_argument("--cache", default="data/radiation")
    parser.add_argument("--max-hours", type=int, default=128)
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError("Write optional radiation augmentation to a separate output")
    result = add_radiation(pd.read_parquet(source), args.cache, args.max_hours)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    report = result.attrs["radiation_context"]
    output.with_suffix(".radiation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": len(result), "hours": report["requested_hours"], "downloaded_bytes": report.get("downloaded_bytes", 0),
                      "errors": report["errors"], "coverage": {v: int(result[v].notna().sum()) for v in VARIABLES.values()}}))


if __name__ == "__main__":
    main()
