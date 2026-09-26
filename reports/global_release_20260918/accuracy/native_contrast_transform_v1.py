"""Pure prospective 31-column representation; no I/O, model or fitting entry point.

Inputs must already follow the frozen native feature source/time/area contract.
Land skin/soil/air are the same instantaneous Land support. Both air lags are
ERA5; their separation is 23 hours. Shortwave fields retain their saved floor
hour and trailing-three-hour means. These contrasts are not measured heat
capacity, an energy budget, or an instantaneous shortwave heating rate.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

INPUT_FEATURES = (
    'era5_land_air_temperature_c', 'era5_land_skin_temperature_c',
    'era5_land_soil_temperature_0_7cm_c',
    'era5_land_soil_moisture_0_7cm_m3_m3', 'era5_land_snow_water_equivalent_m',
    'dewpoint_c', 'wind_speed_m_s', 'wind_direction_sin', 'wind_direction_cos',
    'cloud_cover_fraction', 'shortwave_down_w_m2', 'era5_longwave_down_w_m2',
    'precipitation_mm_h', 'rain_mm_24h', 'rain_mm_72h',
    'air_temperature_lag1_c', 'air_temperature_lag24_c', 'shortwave_down_mean3_w_m2',
    'elevation', 'slope', 'worldcover_tree_class_fraction',
    'worldcover_grass_class_fraction', 'worldcover_crop_class_fraction',
    'worldcover_built_class_fraction', 'worldcover_bare_class_fraction',
    'worldcover_water_fraction', 'solar_elevation_deg', 'hour_sin', 'hour_cos',
    'day_of_year_sin', 'day_of_year_cos',
)
# Original slot -> (new name, subtracted retained input, divisor, output unit).
REPLACEMENTS = {
    'era5_land_skin_temperature_c': (
        'era5_land_skin_minus_air_c', 'era5_land_air_temperature_c', 1., 'degC'),
    'era5_land_soil_temperature_0_7cm_c': (
        'era5_land_soil_0_7cm_minus_air_c', 'era5_land_air_temperature_c', 1., 'degC'),
    'air_temperature_lag1_c': (
        'era5_air_antecedent_23h_tendency_c_per_hour', 'air_temperature_lag24_c', 23., 'degC/hour'),
    'shortwave_down_w_m2': (
        'shortwave_hour_minus_mean3_w_m2', 'shortwave_down_mean3_w_m2', 1., 'W/m2'),
}
OUTPUT_FEATURES = tuple(REPLACEMENTS[name][0] if name in REPLACEMENTS else name
                        for name in INPUT_FEATURES)


def transform(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exactly 31 ordered columns without changing rows or the input.

    The 27 retained columns (including the air reconstruction baseline) preserve
    their values and dtypes. Derived values require both finite operands; missing
    or infinite operands yield NaN, never zero or a filled value. No support mask
    is created or changed here. Caller must preserve the original admission.
    """
    if tuple(frame.columns) != INPUT_FEATURES:
        raise ValueError('Exactly the frozen 31 input columns in their original order are required')
    for name in INPUT_FEATURES:
        dtype = frame[name].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise TypeError('Real numeric feature required: ' + name)
    result = frame.copy(deep=True)
    for original, (renamed, reference, divisor, _) in REPLACEMENTS.items():
        left = frame[original].to_numpy(dtype=float, na_value=np.nan)
        right = frame[reference].to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(left) & np.isfinite(right)
        values = np.full(len(frame), np.nan, dtype=float)
        with np.errstate(over='ignore', invalid='ignore'):
            values[finite] = (left[finite] - right[finite]) / divisor
        if not np.isfinite(values[finite]).all():
            raise ValueError('Contrast overflow from finite inputs: ' + original)
        result[original] = values
    result.columns = OUTPUT_FEATURES
    return result
