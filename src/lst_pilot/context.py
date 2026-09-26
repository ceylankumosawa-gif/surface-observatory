"""Static climate sampling and deterministic solar/time features."""
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import rasterio

CLIMATE_URL = "https://data.naturalcapitalalliance.stanford.edu/download/global/koppen_geiger_climatezones/koppen_geiger_climatezones_1991_2020_1km.tif"
CLIMATE_CODES = ["unknown", "Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk", "Csa", "Csb", "Csc", "Cwa", "Cwb", "Cwc", "Cfa", "Cfb", "Cfc", "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd", "Dfa", "Dfb", "Dfc", "Dfd", "ET", "EF"]


def climate_label(value):
    """Nodata, masked and malformed raster cells remain explicitly unknown."""
    if np.ma.is_masked(value):
        return "unknown"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(number) or not number.is_integer():
        return "unknown"
    code = int(number)
    return CLIMATE_CODES[code] if 0 <= code < len(CLIMATE_CODES) else "unknown"


def add_context(frame, climate_raster=CLIMATE_URL):
    data = frame.copy()
    data["datetime_utc"] = pd.to_datetime(data.datetime_utc, utc=True)
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=2, GDAL_HTTP_RETRY_DELAY=2, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(str(climate_raster)) as source:
            from rasterio.warp import transform
            xs, ys = transform("EPSG:4326", source.crs, data.longitude.to_list(), data.latitude.to_list())
            labels = []
            for x, y, sampled in zip(xs, ys, source.sample(zip(xs, ys), indexes=1, masked=True)):
                inside = source.bounds.left <= x < source.bounds.right and source.bounds.bottom < y <= source.bounds.top
                labels.append(climate_label(sampled[0]) if inside else "unknown")
    data["climate_class"] = labels
    data["climate_sampling_status"] = np.where(data["climate_class"].eq("unknown"), "unknown_or_nodata", "classified")
    data["climate_source"] = "Beck et al. 2023, 1991-2020; NatCap COG conversion"
    # pvlib's numpy SPA accepts vector latitude/longitude paired with timestamps.
    position = pvlib.solarposition.spa_python(pd.DatetimeIndex(data.datetime_utc),
                                            data.latitude.to_numpy(), data.longitude.to_numpy(),
                                            how="numpy")
    data["solar_elevation_deg"] = position["elevation"].to_numpy()
    azimuth = np.deg2rad(position["azimuth"].to_numpy())
    data["solar_azimuth_sin"], data["solar_azimuth_cos"] = np.sin(azimuth), np.cos(azimuth)
    hour = data.datetime_utc.dt.hour + data.datetime_utc.dt.minute / 60 + data.longitude / 15
    data["hour_sin"], data["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    day = data.datetime_utc.dt.dayofyear
    data["day_of_year_sin"], data["day_of_year_cos"] = np.sin(2*np.pi*day/365.2425), np.cos(2*np.pi*day/365.2425)
    return data
