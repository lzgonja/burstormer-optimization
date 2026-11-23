import csv

import numpy as np
import pytest

from reference_selection import (EVALUATION_METHODS, reconstruct_selected_references,
                                 reconstruction_filename, reorder_burst, select_available_references,
                                 select_reference_frame, write_experiment_csv)


def burst():
    return np.arange(14)[:, None, None, None] * np.ones((14, 4, 8, 8))


def test_default_reference_is_zero_and_unchanged():
    original = burst()
    assert select_reference_frame(original) == 0
    np.testing.assert_array_equal(reorder_burst(original), original)


def test_all_available_methods_are_selected_automatically(monkeypatch):
    from reference_selection import algorithm1, algorithm2

    calls = []
    monkeypatch.setattr(algorithm1, 'select_reference_frame',
                        lambda value: calls.append('gbrs') or 5)
    monkeypatch.setattr(algorithm2, 'select_reference_frame',
                        lambda value: calls.append('algorithm2') or 7)

    selections = select_available_references(burst())
    assert list(selections) == ['frame0', 'gbrs', 'algorithm2']
    assert calls == ['gbrs', 'algorithm2']
    assert selections['frame0'] == 0
    assert 0 <= selections['gbrs'] < 14
    assert 0 <= selections['algorithm2'] < 14
    assert selections['algorithm2'] == 7
    assert EVALUATION_METHODS == ('frame0', 'gbrs', 'algorithm2', 'algorithm3')


def test_duplicate_selections_reuse_reconstruction():
    calls = []

    def model(value):
        calls.append(value[:, 0, 0, 0].tolist())
        return value

    outputs = reconstruct_selected_references(
        burst(), model, {'frame0': 0, 'gbrs': 7, 'algorithm2': 7})
    assert set(outputs) == {0, 7}
    assert len(calls) == 2
    assert [7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13] in calls


def test_frame0_corrected_gain_is_zero():
    shift_control = pytest.importorskip('shift_control')
    assert shift_control.evaluate_reference_pair({}, 0, None, None, None, None) == 0.0


def test_algorithm3_uses_cached_frame0_and_one_additional_inference():
    torch = pytest.importorskip('torch')
    from reference_selection.algorithm3 import FINAL_ALPHA, build_synthetic_reference_burst

    original = torch.arange(14, dtype=torch.float32).view(1, 14, 1, 1, 1).expand(1, 14, 4, 8, 8) / 14
    cached_frame0 = torch.ones(1, 3, 64, 64)
    modified = build_synthetic_reference_burst(original, cached_frame0, FINAL_ALPHA)
    calls = []

    def model(value):
        calls.append(value)
        return cached_frame0

    output = model(modified)
    assert output is cached_frame0
    assert len(calls) == 1
    assert not torch.equal(modified[:, 0], original[:, 0])
    assert torch.equal(modified[:, 1:], original[:, 1:])


def test_same_reference_pair_gain_uses_one_shared_region():
    torch = pytest.importorskip('torch')
    shift_control = pytest.importorskip('shift_control')
    baseline, candidate = object(), object()

    class Metrics:
        def align_patches(self, output, gt, burst):
            value = 30.0 if output is baseline else 32.5
            return [{'valid': torch.tensor([True, True]),
                     'in_view': torch.tensor([True, False]), 'psnr': value}]

        def score(self, aligned, regions):
            assert regions[0].tolist() == [True, False]
            return aligned[0]['psnr'], 0.0, 0.0

    gain = shift_control.evaluate_same_reference_pair(baseline, candidate, None, None, Metrics())
    assert gain == pytest.approx(2.5)


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
    rows = [
        {'burst_id': 'burst_001', 'selection_method': method,
         'experiment_type': 'iterative_reference_refinement' if method == 'algorithm3'
         else 'adaptive_reference_selection',
         'selected_reference_index': '' if method == 'algorithm3' else 0,
         'psnr_gain_vs_frame0': 0.5 if method == 'algorithm3' else 0.0,
         'alpha': 0.25 if method == 'algorithm3' else '',
         'evaluation_method': 'same_reference_pair' if method == 'algorithm3'
         else 'v2_reverse_shift_control'}
        for method in EVALUATION_METHODS
    ]
    write_experiment_csv(path, rows)
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['burst_id', 'selection_method', 'experiment_type', 'selected_reference_index',
                                     'psnr_gain_vs_frame0', 'alpha', 'evaluation_method']
        saved = list(reader)
    assert [row['selection_method'] for row in saved] == list(EVALUATION_METHODS)
    assert {row['experiment_type'] for row in saved[:-1]} == {'adaptive_reference_selection'}
    assert saved[-1] == {'burst_id': 'burst_001', 'selection_method': 'algorithm3',
                         'experiment_type': 'iterative_reference_refinement',
                         'selected_reference_index': '', 'psnr_gain_vs_frame0': '0.5',
                         'alpha': '0.25', 'evaluation_method': 'same_reference_pair'}
    assert reconstruction_filename('burst_001', 'algorithm3') == 'burst_001_algorithm3.png'
