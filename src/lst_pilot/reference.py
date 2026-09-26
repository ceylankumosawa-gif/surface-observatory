"""NOAA SURFRAD minute radiation reference data, including nights/cloudy periods.

The derived broadband radiometric temperature is emissivity-dependent and has
the downward-looking radiometer footprint. It is not a validated 100 m pixel
truth. SURFRAD air temperature is measured at 10 m, unlike nominal 2 m GHCNh.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stations import _fetch

BASE_URL = "https://gml.noaa.gov/aftp/data/radiation/surfrad"
DOCUMENTATION_URL = BASE_URL + "/Boulder_CO/README_SURFRAD.txt"
SIGMA = 5.670374419e-8  # W m^-2 K^-4, Stefan-Boltzmann constant
SITES = {
    "tbl": {"directory": "Boulder_CO", "name": "Table Mountain", "latitude": 40.12498,
            "longitude": -105.23680, "elevation_m": 1689,
            "metadata_url": "https://gml.noaa.gov/grad/surfrad/tablemt.html",
            "footprint_note": "Site coordinates identify facility; upwelling radiometer tower is approximately 100 yards north of the instrument deck. Audit exact tower footprint before pixel validation."},
    "sxf": {"directory": "Sioux_Falls_SD", "name": "Sioux Falls", "latitude": 43.73403,
            "longitude": -96.62328, "elevation_m": 473,
            "metadata_url": "https://gml.noaa.gov/grad/surfrad/siouxfalls.html",
            "footprint_note": "Site coordinates identify facility; audit downward-looking tower radiometer footprint before pixel validation."},
}
PAIRS = ["shortwave_down_w_m2", "shortwave_up_w_m2", "direct_normal_w_m2", "diffuse_w_m2",
         "longwave_down_w_m2", "lw_down_case_k", "lw_down_dome_k", "longwave_up_w_m2",
         "lw_up_case_k", "lw_up_dome_k", "uvb_mw_m2", "par_w_m2", "net_shortwave_w_m2",
         "net_longwave_w_m2", "net_radiation_w_m2", "air_temperature_10m_c",
         "relative_humidity_percent", "wind_speed_m_s", "wind_direction_deg", "station_pressure_hpa"]
COLUMNS = ["year", "day_of_year", "month", "day", "hour", "minute", "decimal_hour", "solar_zenith_deg"]
for _variable in PAIRS:
    COLUMNS.extend([_variable, _variable + "_qc"])


def daily_url(site: str, day: str | date) -> str:
    if site not in SITES:
        raise ValueError(f"Supported pilot SURFRAD sites: {', '.join(SITES)}")
    if isinstance(day, str):
        day = date.fromisoformat(day)
    if isinstance(day, datetime):
        day = day.date()
    if not 2009 <= day.year <= date.today().year:
        raise ValueError("Pilot supports the one-minute era (2009 onward)")
    filename = f"{site}{day:%y}{day.timetuple().tm_yday:03d}.dat"
    return f"{BASE_URL}/{SITES[site]['directory']}/{day.year}/{filename}"


def download_day(site: str, day: str | date, cache_dir: str | Path, refresh: bool = False) -> Path:
    """Explicit download of one station/day; usually less than 0.4 MB."""
    url = daily_url(site, day)
    filename = url.rsplit("/", 1)[-1]
    year = url.split("/")[-2]
    return _fetch(url, Path(cache_dir) / site / year / filename, 2_000_000,
                  refresh, documentation_url=DOCUMENTATION_URL)


def radiometric_temperature_c(longwave_up, longwave_down, emissivity: float = 0.97):
    """T=[(LW_up - (1-epsilon)*LW_down)/(epsilon*sigma)]^(1/4).

    A scalar broadband emissivity is an explicit assumption, not a measurement.
    Nonphysical or missing flux inputs yield NaN. Unit inputs are W/m².
    """
    if not 0 < emissivity <= 1:
        raise ValueError("Emissivity must be in (0,1]")
    up, down = np.asarray(longwave_up, dtype=float), np.asarray(longwave_down, dtype=float)
    thermal = (up - (1 - emissivity) * down) / (emissivity * SIGMA)
    thermal = np.where((thermal > 0) & (up > 0) & (down >= 0), thermal, np.nan)
    return np.power(thermal, 0.25) - 273.15


def parse_day(path: str | Path, emissivity: float = 0.97, keep_raw: bool = True) -> pd.DataFrame:
    """Keep NOAA minute END times, mask QC failures, retain emissivity sensitivity."""
    path = Path(path)
    with path.open() as stream:
        station_name = stream.readline().strip()
        position = stream.readline().split()
    if len(position) < 3:
        raise ValueError("Missing SURFRAD station coordinate header")
    frame = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"Unexpected SURFRAD column count: {frame.shape[1]}, expected {len(COLUMNS)}")
    frame.columns = COLUMNS
    frame["timestamp_utc"] = pd.to_datetime(frame[["year", "month", "day", "hour", "minute"]], utc=True, errors="coerce")
    for column in PAIRS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if keep_raw:
            frame[column + "_raw"] = values
        frame[column] = values.where(frame[column + "_qc"].eq(0) & values.ne(-9999.9))
    up, down = frame.longwave_up_w_m2, frame.longwave_down_w_m2
    frame["radiometric_temperature_c"] = radiometric_temperature_c(up, down, emissivity)
    for alternate in (0.96, 0.98):
        frame[f"radiometric_temperature_emissivity_{str(alternate).replace('.', 'p')}_c"] = radiometric_temperature_c(up, down, alternate)
    frame["assumed_emissivity"] = emissivity
    frame["is_daylight"] = frame.solar_zenith_deg.lt(90)
    # NOAA recommends DNI*cos(SZA)+diffuse over a single global pyranometer.
    cosine = np.cos(np.deg2rad(frame.solar_zenith_deg)).clip(lower=0)
    frame["shortwave_components_w_m2"] = frame.direct_normal_w_m2.clip(lower=0) * cosine + frame.diffuse_w_m2.clip(lower=0)
    frame["station_id"] = path.name[:3]
    frame.attrs.update(
        source_file=str(path.resolve()), station_name=station_name,
        header_latitude=float(position[0]), header_longitude=float(position[1]), header_elevation_m=float(position[2]),
        timestamp_convention="UTC end of preceding one-minute averaging period (since 2009)",
        air_temperature_height_m=10,
        documentation_url=DOCUMENTATION_URL,
        emissivity_assumption=emissivity,
        temperature_target="Emissivity-dependent broadband radiometric temperature over instrument footprint; not 100 m satellite truth",
        site_metadata=SITES.get(path.name[:3], {}))
    return frame.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", choices=SITES)
    parser.add_argument("day", help="UTC YYYY-MM-DD")
    parser.add_argument("--cache", default="data/reference")
    parser.add_argument("--output", help="Optional parsed .csv/.parquet output")
    parser.add_argument("--emissivity", type=float, default=0.97)
    args = parser.parse_args()
    path = download_day(args.site, args.day, args.cache)
    frame = parse_day(path, args.emissivity)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".parquet":
            frame.to_parquet(output, index=False)
        else:
            frame.to_csv(output, index=False)
        output.with_suffix(output.suffix + ".schema.json").write_text(json.dumps(frame.attrs, indent=2) + "\n")
    print(json.dumps({"path": str(path), "rows": len(frame),
                      "valid_radiometric_temperature": int(frame.radiometric_temperature_c.notna().sum()),
                      "valid_night_reference": int((frame.radiometric_temperature_c.notna() & ~frame.is_daylight).sum())}))


if __name__ == "__main__":
    main()
