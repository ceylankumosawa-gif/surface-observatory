"""Climate nodata must never become a fabricated valid climate class."""

import numpy as np
import pytest

from lst_pilot.context import climate_label


@pytest.mark.parametrize("value", [np.ma.masked, np.nan, np.inf, None, -1, 255, 1.5, "invalid"])
def test_bad_climate_cells_are_unknown(value):
    assert climate_label(value) == "unknown"


def test_published_integer_climate_codes_are_mapped():
    assert climate_label(1) == "Af"
    assert climate_label(15) == "Cfb"
    assert climate_label(30) == "EF"
