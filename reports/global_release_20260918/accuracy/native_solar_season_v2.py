"""Prospective seasonal coordinates; no data loading, model fitting or I/O.

Daily mean top-of-atmosphere horizontal irradiance uses pvlib's Spencer
declination/distance approximations and fixed solar constant1366.1W/m². A
continuous365.2425-day phase anchored at2021-01-01 avoids duplicating the
orbital phase of leap-year December31 and January1. This is a fixed analytic
seasonal coordinate, not an astronomical ephemeris. Its centered derivative
uses real adjacent UTC calendar dates. Neither
field is measured surface radiation or a closed surface-energy budget.

Call cells() on the SAME100m cells/area weights as existing solar geometry;
do not replace footprint integration with a centroid lookup.
"""
import numpy as np
import pandas as pd
import pvlib
from pyproj import Transformer
from native_contrast_transform_v1 import INPUT_FEATURES

FIELDS = ('toa_daily_mean_horizontal_w_m2', 'toa_daily_change_w_m2_per_day')
OUTPUT_FEATURES = INPUT_FEATURES[:-2] + FIELDS
SOLAR_CONSTANT = 1366.1
PHASE_EPOCH = pd.Timestamp('2021-01-01T00:00:00Z')
PHASE_PERIOD_DAYS = 365.2425


def utc_date(timestamp):
    t = pd.Timestamp(timestamp)
    if pd.isna(t) or t.tz is None:
        raise ValueError('A finite timezone-aware timestamp is required')
    return t.tz_convert('UTC').normalize()


def spencer_phase(timestamp):
    """Continuous fractional Spencer day; its trigonometric period is365."""
    elapsed_days = (utc_date(timestamp) - PHASE_EPOCH).days
    return 1. + elapsed_days * 365. / PHASE_PERIOD_DAYS


def daily_mean(latitude_deg, timestamp):
    """W/m² averaged over24h, with exact geometric polar-day/night limits."""
    source = np.asarray(latitude_deg)
    if source.dtype.kind not in 'ifu':
        raise TypeError('Real numeric latitudes are required')
    lat = source.astype(float)
    if np.any(np.isinf(lat)) or np.any(np.abs(lat[np.isfinite(lat)]) > 90):
        raise ValueError('Latitude must lie in[-90,90] or be missing')
    date = utc_date(timestamp)
    phase = spencer_phase(date)
    delta = float(pvlib.solarposition.declination_spencer71(phase))
    extra = float(pvlib.irradiance.get_extra_radiation(
        phase, solar_constant=SOLAR_CONSTANT, method='spencer'))
    phi = np.deg2rad(lat)
    a, b = np.sin(phi) * np.sin(delta), np.cos(phi) * np.cos(delta)
    # a+b*cos(hour_angle) is cosine of geometric solar zenith.
    # Classify polar limits before division, including latitudes exactly±90°.
    daylight, night = a - b >= 0, a + b <= 0
    ordinary = ~(daylight | night) & np.isfinite(lat)
    sunset = np.full(lat.shape, np.nan, dtype=float)
    sunset[daylight], sunset[night] = np.pi, 0.
    sunset[ordinary] = np.arccos(np.clip(-a[ordinary] / b[ordinary], -1., 1.))
    return np.maximum(0., extra / np.pi * (sunset * a + b * np.sin(sunset)))


def seasonal_values(latitude_deg, timestamp):
    date = utc_date(timestamp)
    today = daily_mean(latitude_deg, date)
    change = (daily_mean(latitude_deg, date + pd.Timedelta(days=1))
              - daily_mean(latitude_deg, date - pd.Timedelta(days=1))) / 2.
    return np.stack([today, change], axis=-1)


def cells(meta, columns, timestamp):
    """Ordered per-cell values for an existing static-grid sparse operator."""
    cols = np.asarray(columns)
    if cols.ndim != 1 or cols.dtype.kind not in 'iu':
        raise TypeError('One-dimensional integer cell indices are required')
    width, height = int(meta['width']), int(meta['height'])
    if width <= 0 or height <= 0 or np.any(cols < 0) or np.any(cols >= width * height):
        raise ValueError('Cell indices must belong to the supplied grid')
    yy, xx = np.divmod(cols, width)
    _, lat = Transformer.from_crs(meta['epsg'], 4326, always_xy=True).transform(
        meta['west'] + (xx + .5) * 100, meta['south'] + (yy + .5) * 100)
    return seasonal_values(lat, timestamp)


def replace_calendar(frame, integrated_values):
    """Replace only the last two slots; preserve29 fields and row identities.

    Caller supplies complete-area weighted values and preserves original
    admission masks. Geometry does not make previously missing rows eligible.
    """
    if tuple(frame.columns) != INPUT_FEATURES:
        raise ValueError('Exactly the original31 features in order are required')
    values = np.asarray(integrated_values)
    if values.shape != (len(frame), 2) or values.dtype.kind not in 'ifu':
        raise ValueError('Provide two real-valued integrated fields per row')
    if not np.isfinite(values).all():
        raise ValueError('Seasonal geometry support must be complete; do not alter admission')
    result = frame.iloc[:, :-2].copy(deep=True)
    for j, field in enumerate(FIELDS):
        result[field] = values[:, j]
    return result
