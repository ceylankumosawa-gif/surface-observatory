import pandas as pd
import pytest
from lst_pilot.coarse_finalize import TARGET_COLUMNS, check_targets, split_targets, count_support


def samples():
    return pd.DataFrame([
        ['fit', 'greater_london', '2022-01-01T00:00:00Z', 51., 0., 32630],
        ['eval', 'greater_london', '2023-01-01T00:00:00Z', 51., 0., 32630],
    ], columns=TARGET_COLUMNS)


def test_split_preserves_all_ids_and_rejects_heldout_year_before_values():
    frame = samples(); fitting, evaluation = split_targets(frame)
    assert list(fitting.sample_id) == ['fit'] and list(evaluation.sample_id) == ['eval']
    with pytest.raises(ValueError, match='year boundary'):
        check_targets(frame, fitting=True)
    frame.loc[1, 'datetime_utc'] = '2024-01-01T00:00:00Z'
    with pytest.raises(ValueError, match='year boundary'):
        split_targets(frame)


def test_does_not_accept_labels_or_duplicate_identity_in_metadata():
    frame = samples(); frame['lst_c'] = 20.
    with pytest.raises(ValueError, match='six metadata'):
        split_targets(frame)
    frame = samples(); frame.loc[1, 'sample_id'] = 'fit'
    with pytest.raises(ValueError, match='unique sample'):
        split_targets(frame)


def test_coverage_counts_independent_dates_not_only_repeated_cells():
    frame = samples()
    joined = pd.DataFrame({'sample_id': frame.sample_id, 'coarse_context_eligible': [True, True],
                           'coarse_product': ['VNP21', 'VNP21'], 'coarse_native_id': ['same', 'same']})
    audit = count_support(frame, joined)
    assert audit['matched_pilot_dates'] == 2 and audit['unique_native_cells'] == 1
    joined = joined.iloc[::-1]
    with pytest.raises(ValueError, match='identity/order'):
        count_support(frame, joined)
