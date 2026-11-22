"""Compare every reference frame with frame 0, from a reference_frame_sweep.py result (no model run, no GPU).

For each reference frame: mean PSNR / SSIM / LPIPS, the paired PSNR gain over frame 0 with a 95% bootstrap interval
over bursts, and how often it beats frame 0. Done twice: with the standard metric (psnr) and on the pixels every
reference covers (psnr_common). The oracle picks the best frame per burst using the ground truth (an upper bound).

Example:
    python summarize_reference_sweep.py --sweep_dir ./Results/Real/t01_sweep_scenes
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np


def bootstrap(diffs: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), (n, len(diffs)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sweep_dir', required=True, type=str)
    args = parser.parse_args()

    table = defaultdict(dict)
    with open(os.path.join(args.sweep_dir, 'per_burst.csv'), newline='') as f:
        for r in csv.DictReader(f):
            table[r['burst']][int(r['ref'])] = {k: float(r[k]) for k in ('psnr', 'ssim', 'lpips', 'psnr_common')}
    refs = sorted(next(iter(table.values())))
    if 0 not in refs:
        raise ValueError('The sweep must include reference frame 0')
    bursts = sorted(b for b, d in table.items() if sorted(d) == refs)

    def column(ref, key):
        return np.array([table[b][ref][key] for b in bursts])

    rows = []
    choices = {'oracle': {k: [max(refs, key=lambda r: table[b][r][k]) for b in bursts] for k in ('psnr', 'psnr_common')}}
    for ref in refs + ['oracle']:
        row = {'reference': ref}
        for k in ('psnr', 'psnr_common'):
            if ref == 'oracle':
                vals = np.array([table[b][c][k] for b, c in zip(bursts, choices['oracle'][k])])
            else:
                vals = column(ref, k)
            diff = vals - column(0, k)
            lo, hi = bootstrap(diff)
            row.update({k: vals.mean(), f'{k}_gain': diff.mean(), f'{k}_ci_low': lo, f'{k}_ci_high': hi,
                        f'{k}_beats_frame0': int((diff > 0).sum())})
        if ref != 'oracle':
            row.update({'ssim': column(ref, 'ssim').mean(), 'lpips': column(ref, 'lpips').mean()})
        rows.append(row)

    print(f'{len(bursts)} bursts. Gain = paired difference with frame 0, 95% bootstrap interval over bursts.')
    print(f'{"ref":>7}{"PSNR":>8}{"gain":>8}{"interval":>18}{"wins":>6}{"PSNR_common":>13}{"gain":>8}'
          f'{"interval":>18}{"wins":>6}{"SSIM":>8}{"LPIPS":>8}')
    for r in rows:
        ci = lambda k: '[%+.2f, %+.2f]' % (r[f'{k}_ci_low'], r[f'{k}_ci_high'])
        extra = f'{r["ssim"]:>8.4f}{r["lpips"]:>8.4f}' if 'ssim' in r else ''
        print(f'{str(r["reference"]):>7}{r["psnr"]:>8.2f}{r["psnr_gain"]:>+8.2f}{ci("psnr"):>18}'
              f'{r["psnr_beats_frame0"]:>6}{r["psnr_common"]:>13.2f}{r["psnr_common_gain"]:>+8.2f}'
              f'{ci("psnr_common"):>18}{r["psnr_common_beats_frame0"]:>6}{extra}')
    for k in ('psnr', 'psnr_common'):
        used = np.bincount(choices['oracle'][k], minlength=len(refs))
        print(f'Oracle ({k}) picks per frame: ' + ' '.join(f'{i}:{n}' for i, n in enumerate(used) if n))

    path = os.path.join(args.sweep_dir, 'vs_frame0.csv')
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != 'reference', k))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()
