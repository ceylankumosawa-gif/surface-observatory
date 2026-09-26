import numpy as np
import pandas as pd
import pytest

from lst_global.patch import patch_area, support_reasons, restore_rows
from lst_global.model import NUMERIC_FEATURES
from lst_global.receipts import STATION_OK


def test_patch_is_aligned_inside_one_registry_tile_and_never_requires_labels():
    area, frame = patch_area(-1.25, 51.75, 32)
    assert area['epsg'] == 32630
    assert frame.shape[0] == 1024
    assert frame.sample_id.is_unique and frame.zone_owned.all()
    assert all(x % 100 == 0 for x in area['extent_m'])
    assert 'lst_c' not in frame and 'label_product' not in frame


@pytest.mark.parametrize('lon,lat,cells', [(180, 0, 32), (0, 88, 32), (0, -88, 32), (0, 0, 513)])
def test_unsupported_source_queries_fail_before_fetch(lon, lat, cells):
    with pytest.raises(ValueError):
        patch_area(lon, lat, cells)


def test_support_reasons_separate_mask_weather_water_and_climate_gaps():
    f = pd.DataFrame({name: np.zeros(7) for name in NUMERIC_FEATURES})
    f['climate_class'] = 'Cfb'
    f['zone_owned'] = True
    f['worldcover_classified_fraction'] = 1.
    f['worldcover_land_fraction'] = 1.
    f['solar_elevation_deg'] = 20.
    f['station_id'] = 'test-station'
    f['station_age_minutes'] = 30.
    f['station_distance_km'] = 10.
    f['observed_station_air_temperature_c'] = 20.
    f['air_temperature_source'] = 'observed_station_residual_plus_ERA5_spatial_background'
    f['verified_station_report_status'] = STATION_OK
    f.loc[1, 'zone_owned'] = False
    f.loc[2, 'worldcover_classified_fraction'] = np.nan
    f.loc[3, 'worldcover_land_fraction'] = .7
    f.loc[4, 'water_fraction'] = .1
    f.loc[5, 'era5_longwave_down_w_m2'] = np.nan
    f.loc[6, 'climate_class'] = 'EF'
    np.testing.assert_array_equal(support_reasons(f), np.arange(7))
    f.loc[0, 'solar_elevation_deg'] = 5.
    assert support_reasons(f)[0] == 7
    f.loc[0, 'solar_elevation_deg'] = 20.
    f.loc[0, 'station_id'] = ''
    assert support_reasons(f)[0] == 8


def test_source_merge_reorders_by_pixel_id_but_rejects_coordinate_mutation():
    _, before = patch_area(-1.25, 51.75, 2)
    before['datetime_utc'] = pd.Timestamp('2023-06-21T00:00Z')
    shuffled = before.iloc[::-1].copy()
    assert restore_rows(before, shuffled).sample_id.equals(before.sample_id)
    shuffled.loc[0, 'latitude'] += .01
    with pytest.raises(AssertionError):
        restore_rows(before, shuffled)
