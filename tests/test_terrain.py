import numpy as np
from lst_pilot.terrain import terrain_descriptors


def test_horn_gradient_and_downhill_aspect_on_known_plane():
    # Rises 10 m eastward each 100 m; downhill is west.
    z = np.array([[90, 100, 110], [90, 100, 110], [90, 100, 110]], dtype=float)
    result = terrain_descriptors(z)
    assert np.isclose(result["elevation_m"], 100)
    assert np.isclose(result["slope_deg"], np.degrees(np.arctan(.1)))
    assert np.isclose(result["aspect_sin"], -1)
    assert np.isclose(result["aspect_cos"], 0)


def test_flat_surface_has_no_invented_aspect_and_nodata_is_retained():
    result = terrain_descriptors(np.full((3, 3), 100.))
    assert result["slope_deg"] == result["aspect_sin"] == result["aspect_cos"] == 0
    z = np.full((3, 3), 100.)
    z[0, 0] = np.nan
    assert np.isnan(terrain_descriptors(z)["slope_deg"])
    assert terrain_descriptors(z)["elevation_m"] == 100
