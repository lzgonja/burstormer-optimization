"""Stress test: degrade real BurstSR bursts by a controlled amount and see where Burstormer stops coping.

One degradation is swept over several levels. At each level the degraded burst goes to the model; the aligned metrics
still colour-match with the clean burst, so the metric itself is not degraded along with the input. Two no-network
baselines are scored on the same degraded input:
    bilinear    bilinear upsampling of the degraded reference frame
    burst_mean  bilinear upsampling of the plain average of all degraded frames (no alignment)

Degradations (--degradation) and what the level means:
    noise    shot-noise level; signal-dependent shot noise + read noise, same model as the synthetic pipeline
             (data_processing/camera_pipeline.py). Synthetic training covered 0.0001 - 0.012.
    frames   number of distinct frames (1-14); the rest are replaced by copies of the reference
    shift    maximum extra random shift of every non-reference frame, in packed raw pixels
    blur     Gaussian blur sigma applied to every frame, in packed raw pixels
    outlier  number of non-reference frames replaced by a badly misaligned copy (shifted by half the crop)

Two options turn this into a test of reference-frame selection:
    --degrade_frames 0        damage only the listed frames (capture order); noise and blur only. E.g. `0` damages
                              the default reference alone, `1 2 ... 13` everything except it.
    --ref_policy guarded      choose the reference from the (damaged) burst before running the model: frame0, frame2
                              or guarded (see ref_selection.py). Shows whether a selector avoids a damaged frame.

Outputs (in --out_dir): per_burst.csv, summary.csv, and with --save_images N a folder of images per burst and level.

Example:
    python robustness_sweep.py --weights ./Trained_models/Real/burstormer_real_calibrated.pth \
        --degradation noise --bursts_file ./splits/dev.txt --limit 12 --save_images 2
"""

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.burstsr_dataset import BurstSRDataset
from pwcnet.pwcnet import PWCNet
from ref_selection import GUARD_MARGIN, REF_POLICIES
from reference_frame_sweep import (NUM_FRAMES, FastAlignedMetrics, bilinear_reference, load_model, reorder,
                                   select_bursts, select_reference, set_seed)
from utils.metrics import AlignedPred
from utils.postprocessing_functions import process_burstsr_image_rgb

DEFAULT_LEVELS = {
    'noise': [0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    'frames': [14, 10, 8, 6, 4, 2, 1],
    'shift': [0, 1, 2, 4, 8, 16],
    'blur': [0, 0.5, 1, 1.5, 2, 3],
    'outlier': [0, 1, 2, 4, 8, 13],
}
CLEAN_LEVEL = {'noise': 0, 'frames': 14, 'shift': 0, 'blur': 0, 'outlier': 0}
METHODS = ['model', 'bilinear', 'burst_mean']


def read_noise_for(shot_noise: float) -> float:
    """Read-noise variance paired with a shot-noise level (the log-linear fit used in camera_pipeline.py)."""
    return math.exp(2.18 * math.log(shot_noise) + 1.20)


def degrade(burst: torch.Tensor, kind: str, level: float, gen: torch.Generator) -> torch.Tensor:
    """burst: [1, N, 4, H, W] packed RGGB in [0, 1]. Returns a degraded copy; the clean level returns the input."""
    if level == CLEAN_LEVEL[kind]:
        return burst
    n, h, w = burst.shape[1], burst.shape[-2], burst.shape[-1]

    if kind == 'noise':
        variance = burst.clamp(min=0.0) * level + read_noise_for(level)
        return burst + torch.randn(burst.shape, generator=gen, device=burst.device) * variance.sqrt()

    if kind == 'frames':
        out = burst.clone()
        out[:, int(level):] = burst[:, :1]
        return out

    if kind == 'shift':
        max_shift = int(level)
        padded = F.pad(burst[0], [max_shift] * 4, mode='replicate')
        out = burst.clone()
        for i in range(1, n):
            dy, dx = torch.randint(-max_shift, max_shift + 1, (2,), generator=gen, device=burst.device).tolist()
            out[0, i] = padded[i, :, max_shift + dy:max_shift + dy + h, max_shift + dx:max_shift + dx + w]
        return out

    if kind == 'blur':
        radius = max(1, math.ceil(3 * level))
        x = torch.arange(-radius, radius + 1, device=burst.device, dtype=burst.dtype)
        k1 = torch.exp(-x ** 2 / (2 * level ** 2))
        k1 = k1 / k1.sum()
        kernel = (k1[:, None] * k1[None, :]).view(1, 1, 2 * radius + 1, 2 * radius + 1)
        flat = F.pad(burst.view(-1, 1, h, w), [radius] * 4, mode='reflect')
        return F.conv2d(flat, kernel).view(burst.shape)

    if kind == 'outlier':
        out = burst.clone()
        chosen = (torch.randperm(n - 1, generator=gen, device=burst.device)[:int(level)] + 1).tolist()
        for i in chosen:
            out[0, i] = torch.roll(burst[0, i], shifts=(h // 2, w // 2), dims=(-2, -1))
        return out

    raise ValueError(f'Unknown degradation {kind}')


def crossing(levels: List[float], values: List[float], threshold: float, log_scale: bool) -> Optional[float]:
    """First level (interpolated) at which `values` drops to `threshold`. None if it never does."""
    for i, v in enumerate(values):
        if v <= threshold:
            if i == 0:
                return levels[0]
            l0, l1, v0 = levels[i - 1], levels[i], values[i - 1]
            t = (v0 - threshold) / (v0 - v + 1e-12)
            if log_scale and l0 > 0 and l1 > 0:
                return math.exp(math.log(l0) + t * (math.log(l1) - math.log(l0)))
            return l0 + t * (l1 - l0)
    return None


def to_srgb(im: torch.Tensor, meta: Dict, norm: float) -> np.ndarray:
    rgb = process_burstsr_image_rgb(im.cpu(), meta, external_norm_factor=norm)
    return cv2.cvtColor((rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description='Sweep a degradation of the input burst and score the output')
    parser.add_argument('--weights', default='./Trained_models/Real/burstormer_real_calibrated.pth', type=str)
    parser.add_argument('--data_root', default='./burstsr_dataset', type=str)
    parser.add_argument('--pwcnet', default='./pwcnet/pwcnet-network-default.pth', type=str)
    parser.add_argument('--out_dir', default=None, type=str, help='Default: ./Results/Real/robustness_<degradation>')
    parser.add_argument('--degradation', default='noise', choices=sorted(DEFAULT_LEVELS))
    parser.add_argument('--levels', default=None, type=float, nargs='+', help='Override the default levels')
    parser.add_argument('--bursts_file', default=None, type=str, help='Burst names to use, e.g. ./splits/dev.txt')
    parser.add_argument('--limit', default=None, type=int, help='Only N bursts, spread across scenes')
    parser.add_argument('--save_images', default=0, type=int, help='Save images at every level for the first N bursts')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--degrade_frames', default=None, type=int, nargs='+',
                        help='Capture-order indices of the frames to degrade (default: all). Noise and blur only')
    parser.add_argument('--ref_policy', default='frame0', choices=REF_POLICIES,
                        help='How the reference frame is chosen, from the degraded burst (default: frame 0)')
    parser.add_argument('--guard_margin', default=GUARD_MARGIN, type=float,
                        help='Guarded policy: how far the sharpness of frame 2 may be from its neighbours before it is replaced')
    args = parser.parse_args()
    if args.degrade_frames is not None and args.degradation not in ('noise', 'blur'):
        parser.error('--degrade_frames only applies to the noise and blur degradations')

    kind = args.degradation
    levels = args.levels or DEFAULT_LEVELS[kind]
    out_dir = args.out_dir or f'./Results/Real/robustness_{kind}'
    os.makedirs(out_dir, exist_ok=True)
    set_seed(50)

    alignment_net = PWCNet(load_pretrained=True, weights_path=args.pwcnet).cuda()
    metrics = FastAlignedMetrics(alignment_net)
    aligner = AlignedPred(alignment_net=alignment_net)
    model = load_model(args.weights)
    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                             random_flip=False, ref_id=0, shuffle_frames=False)
    indices = select_bursts(dataset.burst_list, args.bursts_file, args.limit)
    gen = torch.Generator(device='cuda')

    rows: List[Dict] = []
    fields = ['burst', 'degradation', 'level', 'method', 'ref', 'psnr', 'ssim', 'lpips', 'input_psnr']
    per_burst_path = os.path.join(out_dir, 'per_burst.csv')
    with open(per_burst_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n_done, idx in enumerate(indices):
            burst, gt, meta_burst, meta_gt, name = dataset[idx]
            burst, gt = burst[None].cuda(), gt[None].cuda()
            save = n_done < args.save_images
            if save:
                img_dir = os.path.join(out_dir, 'images', name)
                os.makedirs(img_dir, exist_ok=True)
                gt_norm = float((gt[0] * meta_gt['norm_factor']).max())
                in_norm = float((burst[0, 0] * meta_burst['norm_factor']).max())
                cv2.imwrite(os.path.join(img_dir, 'gt.jpg'), to_srgb(gt[0], meta_gt, gt_norm))
                display_gain = None

            for li, level in enumerate(levels):
                gen.manual_seed(args.seed * 1000003 + idx * 101 + li)
                bad = degrade(burst, kind, level, gen)
                if args.degrade_frames is not None and level != CLEAN_LEVEL[kind]:
                    mixed = burst.clone()
                    mixed[:, args.degrade_frames] = bad[:, args.degrade_frames]
                    bad = mixed
                # The reference is chosen from what the model would actually receive, then moved to the front.
                # `clean` is the undamaged burst in the same order, used only by the metric.
                ref = select_reference(bad, args.ref_policy, args.guard_margin)
                bad, clean = reorder(bad, ref), reorder(burst, ref)
                with torch.no_grad():
                    outputs = {'model': model(bad).clamp(0.0, 1.0),
                               'bilinear': bilinear_reference(bad),
                               'burst_mean': bilinear_reference(bad.mean(dim=1, keepdim=True))}
                # How far the degraded reference frame is from the clean one (raw domain)
                mse_in = F.mse_loss(bad[:, 0], clean[:, 0]).item()
                input_psnr = '' if mse_in == 0 else -10 * math.log10(mse_in)
                for method, out in outputs.items():
                    # Colour matching inside the metric uses the CLEAN burst
                    psnr, ssim, lp = metrics(out, gt, clean)
                    row = {'burst': name, 'degradation': kind, 'level': level, 'method': method, 'ref': ref,
                           'psnr': psnr, 'ssim': ssim, 'lpips': lp, 'input_psnr': input_psnr}
                    rows.append(row)
                    writer.writerow(row)

                if save:
                    with torch.no_grad():
                        pred_m, _, valid = aligner(outputs['model'], gt, clean)
                    if display_gain is None:
                        # Display only: one per-channel gain per burst, from the first level, so brightness matches
                        # the ground truth. Applied identically at every level; metrics are not affected.
                        v = valid[0].float()
                        display_gain = (gt[0] * v).sum((1, 2)) / ((pred_m[0] * v).sum((1, 2)) + 1e-8)
                    tag = f'{kind}_{li:02d}_{level:g}'
                    cv2.imwrite(os.path.join(img_dir, f'{tag}_model.jpg'),
                                to_srgb(pred_m[0] * display_gain.view(3, 1, 1), meta_gt, gt_norm))
                    raw = bad[0, 0]
                    rgb = torch.stack([raw[0], (raw[1] + raw[2]) / 2, raw[3]]).clamp(0.0, 1.0)
                    rgb = F.interpolate(rgb[None], scale_factor=8, mode='nearest')[0]
                    cv2.imwrite(os.path.join(img_dir, f'{tag}_input.jpg'), to_srgb(rgb, meta_burst, in_norm))
            f.flush()
            print(f'[{n_done + 1}/{len(indices)}] {name}', flush=True)

    # Summary: mean per level and method
    agg = defaultdict(list)
    for r in rows:
        agg[(r['level'], r['method'])].append(r)
    summary = []
    for level in levels:
        for method in METHODS:
            sel = agg[(level, method)]
            in_psnr = [s['input_psnr'] for s in sel if s['input_psnr'] != '']
            summary.append({'degradation': kind, 'level': level, 'method': method, 'n_bursts': len(sel),
                            'psnr': np.mean([s['psnr'] for s in sel]), 'ssim': np.mean([s['ssim'] for s in sel]),
                            'lpips': np.mean([s['lpips'] for s in sel]),
                            'input_psnr': np.mean(in_psnr) if in_psnr else '',
                            'refs_used': ' '.join(f'{r}:{c}' for r, c in
                                                  sorted(Counter(s['ref'] for s in sel).items()))})
    summary_path = os.path.join(out_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    mean = {(s['level'], s['method']): s for s in summary}
    which = 'all frames' if args.degrade_frames is None else f'frames {args.degrade_frames} only'
    print(f'\n{kind} on {which}, reference policy {args.ref_policy}: mean over {len(indices)} bursts')
    print(f'{"level":>8}{"input dB":>10} | {"model PSNR":>10}{"SSIM":>7}{"LPIPS":>7} | {"bilinear":>9}{"burst_mean":>11}'
          f' | {"model-bilinear":>14} | reference used (frame:bursts)')
    for level in levels:
        m, b, a = mean[(level, 'model')], mean[(level, 'bilinear')], mean[(level, 'burst_mean')]
        in_psnr = f'{m["input_psnr"]:.1f}' if m['input_psnr'] != '' else 'clean'
        print(f'{level:>8g}{in_psnr:>10} | {m["psnr"]:>10.2f}{m["ssim"]:>7.3f}{m["lpips"]:>7.3f} | {b["psnr"]:>9.2f}'
              f'{a["psnr"]:>11.2f} | {m["psnr"] - b["psnr"]:>+14.2f} | {m["refs_used"]}')

    # Breaking points along the sweep direction
    model_psnr = [mean[(lv, 'model')]['psnr'] for lv in levels]
    log_scale = kind == 'noise'
    drop3 = crossing(levels, model_psnr, model_psnr[0] - 3.0, log_scale)
    margin = [mean[(lv, 'model')]['psnr'] - mean[(lv, 'bilinear')]['psnr'] for lv in levels]
    no_gain = crossing(levels, margin, 0.0, log_scale)
    clean_bilinear = mean[(levels[0], 'bilinear')]['psnr']
    below_clean_bilinear = crossing(levels, model_psnr, clean_bilinear, log_scale)
    margin_mean = [mean[(lv, 'model')]['psnr'] - mean[(lv, 'burst_mean')]['psnr'] for lv in levels]
    below_mean = crossing(levels, margin_mean, 0.0, log_scale)
    fmt = lambda x: 'not reached' if x is None else f'{x:.3g}'
    print(f'\nBreaking points (level, interpolated):')
    print(f'  model PSNR 3 dB below its clean value:               {fmt(drop3)}')
    print(f'  model below bilinear upsampling of a CLEAN frame:    {fmt(below_clean_bilinear)}')
    print(f'  model no better than bilinear on the same input:     {fmt(no_gain)}')
    print(f'  model no better than plain averaging of the frames:  {fmt(below_mean)}')
    print(f'\nWrote {per_burst_path} and {summary_path}')


if __name__ == '__main__':
    main()
