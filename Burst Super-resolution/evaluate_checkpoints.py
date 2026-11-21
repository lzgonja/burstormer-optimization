"""Score several Burstormer checkpoints and the bilinear baseline on the same BurstSR val bursts.

Every model sees identical input (frame 0 as reference, frames in capture order) and is scored with the standard
aligned PSNR / SSIM / LPIPS (reference_frame_sweep.FastAlignedMetrics). Differences are paired per burst, with a 95%
bootstrap interval resampled over scenes (crops of one scene are near-duplicates, so bursts are not independent).

Outputs in --out_dir:
    per_burst.csv  one row per (burst, model)
    summary.csv    mean metrics per model, and the paired PSNR difference to bilinear and to the first model

Example (whole val set):
    python evaluate_checkpoints.py \
        --weights downloaded=./Trained_models/Real/burstormer_real.pth \
                  trained=./Trained_models/Real/burstormer_burstsr_train.pth \
        --out_dir ./Results/Real/burstsr_train_eval
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from datasets.burstsr_dataset import BurstSRDataset
from pwcnet.pwcnet import PWCNet
from reference_frame_sweep import (NUM_FRAMES, FastAlignedMetrics, bilinear_reference, load_model, select_bursts,
                                   set_seed)

BILINEAR = 'bilinear'


def scene_bootstrap(diff_by_burst: Dict[str, float], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean paired difference and a 95% interval from resampling whole scenes (burst name = <scene>_<crop>)."""
    by_scene = defaultdict(list)
    for name, d in diff_by_burst.items():
        by_scene[name.split('_')[0]].append(d)
    scenes = [np.array(v) for v in by_scene.values()]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = [scenes[i] for i in rng.integers(0, len(scenes), len(scenes))]
        boots.append(np.concatenate(pick).mean())
    mean = float(np.mean(list(diff_by_burst.values())))
    return mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare Burstormer checkpoints and bilinear on BurstSR val')
    parser.add_argument('--weights', nargs='+', required=True, help='label=path pairs; the first is the comparison base')
    parser.add_argument('--data_root', default='./burstsr_dataset', type=str)
    parser.add_argument('--pwcnet', default='./pwcnet/pwcnet-network-default.pth', type=str)
    parser.add_argument('--out_dir', default='./Results/Real/checkpoint_eval', type=str)
    parser.add_argument('--bursts_file', default=None, type=str, help='e.g. ./splits/test.txt (default: whole val set)')
    parser.add_argument('--limit', default=None, type=int, help='Only N bursts, spread across scenes')
    args = parser.parse_args()

    set_seed(50)
    os.makedirs(args.out_dir, exist_ok=True)
    metrics = FastAlignedMetrics(PWCNet(load_pretrained=True, weights_path=args.pwcnet).cuda())
    models = {}
    for item in args.weights:
        label, path = item.split('=', 1)
        models[label] = load_model(path)

    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                             random_flip=False, ref_id=0, shuffle_frames=False)
    if args.limit is not None or args.bursts_file is not None:
        dataset = torch.utils.data.Subset(dataset, select_bursts(dataset.burst_list, args.bursts_file, args.limit))
    loader = DataLoader(dataset, batch_size=1, num_workers=2)

    labels_order = [BILINEAR] + list(models)
    scores: Dict[str, Dict[str, Tuple[float, float, float]]] = {m: {} for m in labels_order}
    start = time.time()
    with open(os.path.join(args.out_dir, 'per_burst.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['burst', 'model', 'psnr', 'ssim', 'lpips'])
        writer.writeheader()
        for i, (burst, gt, _, _, name) in enumerate(loader):
            burst, gt, name = burst.cuda(), gt.cuda(), name[0]
            with torch.no_grad():
                outputs = {BILINEAR: bilinear_reference(burst)}
                outputs.update({label: m(burst).clamp(0.0, 1.0) for label, m in models.items()})
                for label in labels_order:
                    psnr, ssim, lp = metrics(outputs[label], gt, burst)
                    scores[label][name] = (psnr, ssim, lp)
                    writer.writerow({'burst': name, 'model': label, 'psnr': psnr, 'ssim': ssim, 'lpips': lp})
            f.flush()
            if (i + 1) % 50 == 0 or i == 0:
                el = time.time() - start
                print(f'[{i + 1}/{len(dataset)}] {el / (i + 1):.2f}s/burst, '
                      f'ETA {el / (i + 1) * (len(dataset) - i - 1) / 60:.1f} min', flush=True)

    base = list(models)[0]
    rows: List[Dict] = []
    for label in labels_order:
        vals = np.array(list(scores[label].values()))
        row = {'model': label, 'n_bursts': len(vals), 'psnr': vals[:, 0].mean(), 'ssim': vals[:, 1].mean(),
               'lpips': vals[:, 2].mean()}
        for ref, key in ((BILINEAR, 'vs_bilinear'), (base, f'vs_{base}')):
            if label == ref:
                row[key] = ''
                continue
            d, lo, hi = scene_bootstrap({b: scores[label][b][0] - scores[ref][b][0] for b in scores[label]})
            row[key] = f'{d:+.2f} [{lo:+.2f}, {hi:+.2f}]'
        rows.append(row)

    with open(os.path.join(args.out_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_scenes = len({b.split('_')[0] for b in scores[BILINEAR]})
    print(f'\n{len(dataset)} bursts, {n_scenes} scenes. PSNR differences are paired, 95% interval over scenes.')
    print(f'{"model":<14}{"PSNR":>8}{"SSIM":>8}{"LPIPS":>8}  {"PSNR vs bilinear":<24}PSNR vs {base}')
    for r in rows:
        print(f'{r["model"]:<14}{r["psnr"]:>8.2f}{r["ssim"]:>8.4f}{r["lpips"]:>8.4f}  {r["vs_bilinear"]:<24}'
              f'{r[f"vs_{base}"]}')
    print(f'\nWrote {args.out_dir}/per_burst.csv and summary.csv')


if __name__ == '__main__':
    main()
