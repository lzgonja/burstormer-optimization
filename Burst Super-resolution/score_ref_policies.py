"""Score reference-frame policies on clean bursts from a reference_frame_sweep.py result (no model run, no GPU).

The sweep stores the metrics of every burst with every reference frame. A policy only chooses a frame, so its score
is looked up from that table. Each policy's choice is computed from the input burst itself (loaded from the dataset).

Policies: frame0 (original), frame2, guarded (ref_selection.py), algorithm2 at several margins
(reference_selection/algorithm2.py), and the oracle (best frame per burst using the ground truth: an upper bound).

PSNR_common scores every reference on the same pixels, so frames other than 0 are not penalised for the ground truth
being framed on frame 0 (docs/PLAN.md, Track A). Intervals: 95% bootstrap over bursts, paired with frame 0.

Example:
    python score_ref_policies.py --sweep_dir ./Results/Real/alg2_sweep_dev
"""

import argparse
import csv
import os
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from datasets.burstsr_dataset import BurstSRDataset
from ref_selection import guarded_choice
from reference_selection import algorithm2

MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]


def load_sweep(path: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    by_burst = defaultdict(dict)
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            by_burst[r['burst']][int(r['ref'])] = {k: float(r[k]) for k in ('psnr', 'psnr_common', 'sharpness_smooth')}
    return by_burst


def bootstrap(diffs: np.ndarray, n: int = 10000, seed: int = 0) -> List[float]:
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), (n, len(diffs)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sweep_dir', required=True, type=str)
    parser.add_argument('--data_root', default='./burstsr_dataset', type=str)
    args = parser.parse_args()

    sweep = load_sweep(os.path.join(args.sweep_dir, 'per_burst.csv'))
    refs = sorted(next(iter(sweep.values())))
    if refs != list(range(algorithm2.WINDOW)):
        raise ValueError(f'The sweep must cover references 0-{algorithm2.WINDOW - 1}, found {refs}')
    bursts = sorted(b for b, d in sweep.items() if sorted(d) == refs)

    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=14, crop_sz=80, random_flip=False,
                             ref_id=0, shuffle_frames=False)
    index = {name: i for i, name in enumerate(dataset.burst_list)}
    costs = {b: algorithm2.frame_costs(dataset[index[b]][0]) for b in bursts}

    policies = {'frame0': lambda b: 0, 'frame2': lambda b: 2,
                'guarded': lambda b: guarded_choice({r: sweep[b][r]['sharpness_smooth'] for r in refs})}
    for m in MARGINS:
        policies[f'algorithm2 m={m:g}'] = lambda b, m=m: algorithm2.choose_from_costs(costs[b], margin=m)
    policies['algorithm2 best cost'] = lambda b: min(costs[b], key=costs[b].get)
    policies['oracle (common)'] = lambda b: max(refs, key=lambda r: sweep[b][r]['psnr_common'])

    out_rows = []
    print(f'{len(bursts)} bursts from {args.sweep_dir}. Gains are paired with frame 0, 95% bootstrap over bursts.')
    print(f'{"policy":<22}{"PSNR":>8}{"gain":>8}{"interval":>18}{"PSNR_common":>13}{"gain":>8}{"interval":>18}'
          f'{"switched":>10}  frames used')
    base = {k: np.array([sweep[b][0][k] for b in bursts]) for k in ('psnr', 'psnr_common')}
    for name, pick in policies.items():
        chosen = [pick(b) for b in bursts]
        row = {'policy': name, 'n_bursts': len(bursts), 'switched': float(np.mean([c != 0 for c in chosen]))}
        for k in ('psnr', 'psnr_common'):
            vals = np.array([sweep[b][c][k] for b, c in zip(bursts, chosen)])
            row[k] = float(vals.mean())
            row[f'{k}_gain'] = float((vals - base[k]).mean())
            row[f'{k}_ci'] = bootstrap(vals - base[k])
        out_rows.append(row)
        used = ' '.join(f'{r}:{c}' for r, c in sorted(Counter(chosen).items()))
        print(f'{name:<22}{row["psnr"]:>8.3f}{row["psnr_gain"]:>+8.3f}'
              f'{"[%+.3f, %+.3f]" % tuple(row["psnr_ci"]):>18}{row["psnr_common"]:>13.3f}{row["psnr_common_gain"]:>+8.3f}'
              f'{"[%+.3f, %+.3f]" % tuple(row["psnr_common_ci"]):>18}{100 * row["switched"]:>9.0f}%  {used}')

    path = os.path.join(args.sweep_dir, 'policies.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: (' '.join(f'{x:.4f}' for x in v) if isinstance(v, list) else v) for k, v in r.items()})
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()
