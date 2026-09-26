import pandas as pd
import pytest

from lst_pilot.multisensor_combine import distinct_rows


def row(sample="one", acquisition="scene", grid_col=1):
    return pd.DataFrame({"sample_id": [sample], "region_id": ["pilot"],
                         "acquisition_id": [acquisition], "grid_row": [1], "grid_col": [grid_col]})


@pytest.mark.parametrize("other", [row(), row("renamed"), row("one", "other_scene", 2)])
def test_reject_same_identity_or_renamed_physical_observation(other):
    with pytest.raises(ValueError, match="duplicate"):
        distinct_rows([row(), other])


def test_retain_different_sensors_and_cells_without_resampling():
    parts = [row("b"), row("a", "other_sensor"), row("c", grid_col=2)]
    result = distinct_rows(parts)
    assert result.sample_id.tolist() == ["a", "b", "c"]
    assert len(result) == sum(map(len, parts))
