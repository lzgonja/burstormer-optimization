import csv

import numpy as np
import pytest

from reference_selection import (reconstruct_selected_references, reorder_burst, select_available_references,
                                 select_reference_frame, write_experiment_csv)


def burst():
    return np.arange(14)[:, None, None, None] * np.ones((14, 4, 8, 8))


def test_default_reference_is_zero_and_unchanged():
    original = burst()
    assert select_reference_frame(original) == 0
    np.testing.assert_array_equal(reorder_burst(original), original)


def test_all_available_methods_are_selected_automatically():
    selections = select_available_references(burst())
    assert list(selections) == ['frame0', 'gbrs']
    assert selections == {'frame0': 0, 'gbrs': 0}


def test_duplicate_selections_reuse_reconstruction():
    calls = []

    def model(value):
        calls.append(value[:, 0, 0, 0].tolist())
        return value

    outputs = reconstruct_selected_references(
        burst(), model, {'frame0': 0, 'gbrs': 7, 'another_method': 7})
    assert set(outputs) == {0, 7}
    assert len(calls) == 2
    assert [7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13] in calls


def test_frame0_corrected_gain_is_zero():
    shift_control = pytest.importorskip('shift_control')
    assert shift_control.evaluate_reference_pair({}, 0, None, None, None, None) == 0.0


def test_selected_frame_moves_first_and_preserves_capture_order():
    reordered = reorder_burst(burst(), 7)
    assert reordered[:, 0, 0, 0].tolist() == [7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
    assert reordered.shape[0] == 14


@pytest.mark.parametrize('index', [-1, 14, 1.5, True])
def test_invalid_reference_is_rejected(index):
    with pytest.raises((TypeError, ValueError)):
        reorder_burst(burst(), index)


def test_experiment_csv_schema_and_metadata(tmp_path):
    path = tmp_path / 'per_burst.csv'
    write_experiment_csv(path, [{'burst_id': 'burst_001', 'selection_method': 'gbrs',
                                 'selected_reference_index': 7, 'corrected_psnr_gain': 0.5}])
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['burst_id', 'selection_method', 'selected_reference_index',
                                     'corrected_psnr_gain']
        assert next(reader) == {'burst_id': 'burst_001', 'selection_method': 'gbrs',
                                'selected_reference_index': '7', 'corrected_psnr_gain': '0.5'}
