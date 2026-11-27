"""Score reference-frame selectors against shift-control labels, without running the network.

`shift_control.py --refs 14` writes, for every burst, the shift-corrected PSNR, SSIM and LPIPS gain of each of the
14 frames as reference over frame 0 (per_burst.csv). A selector only picks a frame from the input burst, so its
gain on a burst is that frame's labelled gain. This scores every selector on exactly the same labels, in seconds.

    python score_ref_selectors.py --labels ../docs/logs/shift_control_all_metrics_2026-11-24/test_40_bursts/per_burst.csv

Positive gains mean the selected frame is better than frame 0 (for LPIPS: lower). 95% bootstrap over bursts.
"""

import argparse
import csv
from typing import Callable, Dict

import numpy as np

from datasets.burstsr_dataset import BurstSRDataset
from reference_selection import algorithm1, algorithm2

METRICS = {'PSNR (dB)': ('gain_corrected', 1), 'SSIM (x1e-3)': ('ssim_gain_corrected', 1000),
           'LPIPS (x1e-3)': ('lpips_gain_corrected', 1000)}

SELECTORS: Dict[str, Callable] = {
    'frame 0': lambda burst: 0,
    'gbrs (algorithm 1)': algorithm1.select_reference_frame,
    'algorithm 2, v1': algorithm2.select_reference_frame_v1,
    'algorithm 2': algorithm2.select_reference_frame,
}


def load_labels(path: str):
    """Returns (burst names, {metric column: array [bursts, 14]})."""
    rows = [r for r in csv.DictReader(open(path)) if r['ref'].isdigit()]  # skip the probe rows
    bursts = sorted({r['burst'] for r in rows})
    labels = {col: np.full((len(bursts), 14), np.nan) for col, _ in METRICS.values()}
    for r in rows:
        for col in labels:
            labels[col][bursts.index(r['burst']), int(r['ref'])] = float(r[col])
    return bursts, labels


def bootstrap(values: np.ndarray, n: int = 4000, seed: int = 0):
    means = values[np.random.default_rng(seed).integers(0, len(values), (n, len(values)))].mean(1)
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--labels', required=True, help='per_burst.csv from shift_control.py --refs 14')
    parser.add_argument('--data_root', default='./burstsr_dataset')
    args = parser.parse_args()

    bursts, labels = load_labels(args.labels)
    if np.isnan(next(iter(labels.values()))).any():
        raise ValueError('The labels must cover all 14 references for every burst (shift_control.py --refs 14)')
    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=14, crop_sz=80, random_flip=False,
                             ref_id=0, shuffle_frames=False)
    index = {name: i for i, name in enumerate(dataset.burst_list)}
    frames = [dataset[index[b]][0] for b in bursts]
    rows = np.arange(len(bursts))

    print(f'{len(bursts)} bursts from {args.labels}')
    print(f'{"selector":22}' + ''.join(f'{m:>30}' for m in METRICS) + '  picks by frame')
    for name, select in SELECTORS.items():
        picks = np.array([select(f) for f in frames])
        cells = []
        for col, scale in METRICS.values():
            mean, lo, hi = bootstrap(labels[col][rows, picks] * scale)
            cells.append(f'{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        print(f'{name:22}' + ''.join(f'{c:>30}' for c in cells) + f'  {np.bincount(picks, minlength=14).tolist()}')
    for k in (3, 5, 7):
        cells = [f'{labels[col][:, k].mean() * scale:+.3f}' for col, scale in METRICS.values()]
        print(f'{f"fixed frame {k}":22}' + ''.join(f'{c:>30}' for c in cells))
    cells = [f'{np.max(labels[col], 1).mean() * scale:+.3f}' for col, scale in METRICS.values()]
    print(f'{"oracle (ground truth)":22}' + ''.join(f'{c:>30}' for c in cells))


if __name__ == '__main__':
    main()
