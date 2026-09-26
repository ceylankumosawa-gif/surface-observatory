import pandas as pd
import pytest
from lst_pilot.eligibility import select_eligible


def test_known_terrestrial_snow_retained_unknown_mangrove_quarantined():
    frame = pd.DataFrame({"climate_class": ["ET", "unknown"], "water_fraction": [0., 0.],
                          "snow_fraction": [1., 0.], "surface": ["tundra", "mangrove"]})
    kept, quarantine = select_eligible(frame)
    assert kept.surface.tolist() == ["tundra"]
    assert quarantine.surface.tolist() == ["mangrove"]
    assert quarantine.quarantine_reason.iloc[0] == "unresolved_climate_or_coastal_support"


def test_recognised_climate_does_not_override_water_screen():
    frame = pd.DataFrame({"climate_class": ["Cfb"]*3, "water_fraction": [0., .2, float("nan")]})
    kept, quarantine = select_eligible(frame)
    assert len(kept) == 1 and len(quarantine) == 2


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError, match="Invalid water"):
        select_eligible(pd.DataFrame({"climate_class": ["Cfb"], "water_fraction": [-.1]}))
