"""Shift control: how much does the BurstSR metric penalise a reference frame just for its position?

The ground truth is registered to frame 0. An output made with frame k as reference is displaced from it, so the
metric must warp it further (bilinear resampling, larger optical flow). This script separates that penalty from real
reconstruction quality:

    1. Run the model with references 0..K-1 (outputs O_0 ... O_k).
    2. Measure the translation from O_0 to each O_k by phase correlation at output resolution.
    3. Build S_k = O_0 moved by that translation (mirror padding, then an exact Fourier shift: no blur, no ringing).
    4. Score S_k exactly as O_k is scored (burst reordered to reference k, same metric, same pixels).

S_k has the same content as O_0, so PSNR(O_0) - PSNR(S_k) is the metric's penalty for position alone. The corrected
gain of frame k is PSNR(O_k) - PSNR(O_0) + penalty_k. All scores use the pixels valid and in view for every real
output and every shifted copy. Two probes split the penalty: O_0 shifted by half an output pixel (resampling only)
and by a whole number of pixels (flow and framing only). For the probes, input frame 0 is moved by the same amount
(at raw scale), because the metric colour-matches with it and it must stay in register with the output.

--reverse_check estimates the corrected gain a second, independent way: move each real output O_k back to frame 0's
position and score it as reference 0, so frame k's own reconstruction is scored without its position. If the
penalty depends only on position, both estimates agree (docs/SHIFT_CONTROL_VALIDATION.md). With this option every
score also uses the pixels the moved-back outputs cover, so the common region is slightly smaller.

Example:
    python shift_control.py --weights ./Trained_models/Real/burstormer_burstsr_train.pth \
        --bursts_file ./splits/dev.txt --limit 40 --out_dir ./Results/Real/shift_control_dev
"""

import argparse
import csv
import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from datasets.burstsr_dataset import BurstSRDataset
from pwcnet.pwcnet import PWCNet
from reference_frame_sweep import NUM_FRAMES, FastAlignedMetrics, load_model, reorder, select_bursts, set_seed

PAD = 64  # minimum mirror padding (output pixels) around a shifted image; grown for larger shifts
PROBES = {'probe_half_px': (0.5, 0.5), 'probe_whole_px': (8.0, 8.0)}


def translation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sub-pixel (dy, dx) that moves image a onto image b, by phase correlation of the green channels. [1, 3, H, W]."""
    ga, gb = a[0, 1] - a[0, 1].mean(), b[0, 1] - b[0, 1].mean()
    h, w = ga.shape
    window = torch.hann_window(h, periodic=False, device=a.device)[:, None] * \
        torch.hann_window(w, periodic=False, device=a.device)[None, :]
    fa, fb = torch.fft.fft2(ga * window), torch.fft.fft2(gb * window)
    cross = fb * fa.conj()
    corr = torch.fft.ifft2(cross / (cross.abs() + 1e-12)).real  # peaks at the displacement of b relative to a
    peak = int(corr.argmax())
    py, px = peak // w, peak % w

    def refine(cm: float, c0: float, cp: float) -> float:
        d = cm - 2 * c0 + cp
        return 0.0 if abs(d) < 1e-12 else max(-0.5, min(0.5, 0.5 * (cm - cp) / d))

    dy = py + refine(corr[(py - 1) % h, px].item(), corr[py, px].item(), corr[(py + 1) % h, px].item())
    dx = px + refine(corr[py, (px - 1) % w].item(), corr[py, px].item(), corr[py, (px + 1) % w].item())
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return torch.tensor([dy, dx])


def shift_image(img: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """Move img [1, C, H, W] by (dy, dx) output pixels: mirror-pad, exact Fourier shift, crop back."""
    h, w = img.shape[-2:]
    pad = max(PAD, int(math.ceil(max(abs(dy), abs(dx)))) + 4)  # later frames (e.g. 13) can shift ~64 px
    if pad >= min(h, w):
        raise ValueError(f'Shift ({dy:.1f}, {dx:.1f}) is too large for a {h}x{w} image')
    padded = F.pad(img, [pad] * 4, mode='reflect')
    hp, wp = padded.shape[-2:]
    fy = torch.fft.fftfreq(hp, device=img.device)[:, None]
    fx = torch.fft.fftfreq(wp, device=img.device)[None, :]
    ramp = torch.exp(-2j * math.pi * (fy * dy + fx * dx))
    shifted = torch.fft.ifft2(torch.fft.fft2(padded) * ramp).real
    return shifted[..., pad:pad + h, pad:pad + w].clamp(0.0, 1.0)


def bootstrap(x: np.ndarray, n: int = 10000, seed: int = 0) -> List[float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description='Measure the metric penalty for the position of the reference frame')
    parser.add_argument('--weights', default='./Trained_models/Real/burstormer_burstsr_train.pth', type=str)
    parser.add_argument('--data_root', default='./burstsr_dataset', type=str)
    parser.add_argument('--pwcnet', default='./pwcnet/pwcnet-network-default.pth', type=str)
    parser.add_argument('--out_dir', default='./Results/Real/shift_control', type=str)
    parser.add_argument('--refs', default=5, type=int, help='References 0..refs-1')
    parser.add_argument('--bursts_file', default=None, type=str)
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument('--reverse_check', action='store_true',
                        help='Also move each O_k back to frame 0 and score it as reference 0 (second estimate)')
    args = parser.parse_args()

    set_seed(50)
    os.makedirs(args.out_dir, exist_ok=True)
    metrics = FastAlignedMetrics(PWCNet(load_pretrained=True, weights_path=args.pwcnet).cuda())
    model = load_model(args.weights)
    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                             random_flip=False, ref_id=0, shuffle_frames=False)
    indices = select_bursts(dataset.burst_list, args.bursts_file, args.limit)
    refs = list(range(args.refs))

    rows: List[Dict] = []
    path = os.path.join(args.out_dir, 'per_burst.csv')
    fields = ['burst', 'ref', 'shift_dy', 'shift_dx', 'psnr_real', 'psnr_shifted', 'penalty', 'gain_real',
              'gain_corrected', 'shifted_vs_real_psnr', 'common_frac'] + (['gain_reverse'] if args.reverse_check else [])
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n_done, idx in enumerate(indices):
            burst, gt, _, _, name = dataset[idx]
            burst, gt = burst[None].cuda(), gt[None].cuda()
            with torch.no_grad():
                outputs = {k: model(reorder(burst, k)).clamp(0.0, 1.0) for k in refs}
            shifts = {k: translation(outputs[0], outputs[k]) for k in refs}
            shifts[0] = torch.zeros(2)
            shifted = {k: shift_image(outputs[0], *shifts[k].tolist()) for k in refs}
            shifted[0] = outputs[0]

            # Align every image once; real and shifted outputs of frame k are both scored as reference k
            real_al = {k: metrics.align_patches(outputs[k], gt, reorder(burst, k)) for k in refs}
            shift_al = {k: metrics.align_patches(shifted[k], gt, reorder(burst, k)) for k in refs}
            probe_al = {}
            for p, (dy, dx) in PROBES.items():
                moved = burst.clone()
                moved[0, 0] = shift_image(burst[:, 0], dy / 8, dx / 8)[0]
                probe_al[p] = metrics.align_patches(shift_image(outputs[0], dy, dx), gt, moved)
            # Second estimate: frame k's own output moved back to frame 0's position, scored as reference 0
            back_al = {k: metrics.align_patches(shift_image(outputs[k], *(-shifts[k]).tolist()), gt, burst)
                       for k in refs[1:]} if args.reverse_check else {}
            all_al = list(real_al.values()) + list(shift_al.values()) + list(probe_al.values()) + list(back_al.values())
            n_patches = len(real_al[0])
            common = [torch.stack([al[p]['valid'] & al[p]['in_view'] for al in all_al]).all(dim=0)
                      for p in range(n_patches)]
            common_frac = float(np.mean([c.float().mean().item() for c in common]))

            def psnr(al):
                return metrics.score(al, regions=common)[0]

            real = {k: psnr(real_al[k]) for k in refs}
            shift = {k: psnr(shift_al[k]) for k in refs}
            for k in refs:
                # How close a pure translation of O_0 is to the real O_k (on the pixels both cover)
                margin = int(math.ceil(shifts[k].abs().max().item())) + 8
                crop = (Ellipsis, slice(margin, -margin), slice(margin, -margin))
                mse = F.mse_loss(shifted[k][crop], outputs[k][crop]).item()
                row = {'burst': name, 'ref': k, 'shift_dy': shifts[k][0].item(), 'shift_dx': shifts[k][1].item(),
                       'psnr_real': real[k], 'psnr_shifted': shift[k], 'penalty': real[0] - shift[k],
                       'gain_real': real[k] - real[0], 'gain_corrected': real[k] - real[0] + (real[0] - shift[k]),
                       'shifted_vs_real_psnr': -10 * math.log10(mse + 1e-12), 'common_frac': common_frac}
                if args.reverse_check:
                    row['gain_reverse'] = 0.0 if k == 0 else psnr(back_al[k]) - real[0]
                rows.append(row)
                writer.writerow(row)
            for p in PROBES:
                row = {'burst': name, 'ref': p, 'psnr_real': real[0], 'psnr_shifted': psnr(probe_al[p]),
                       'penalty': real[0] - psnr(probe_al[p]), 'common_frac': common_frac}
                rows.append(row)
                writer.writerow(row)
            f.flush()
            print(f'[{n_done + 1}/{len(indices)}] {name}', flush=True)

    # Summary
    by_ref = {k: [r for r in rows if r['ref'] == k] for k in refs + list(PROBES)}
    print(f'\n{len(indices)} bursts, PSNR on the region common to every image ({np.mean([r["common_frac"] for r in rows]):.0%} '
          f'of pixels). Penalty = PSNR(O_0) - PSNR(O_0 moved to frame k). 95% bootstrap over bursts.')
    print(f'{"ref":>14}{"shift px":>10}{"O_k~S_k dB":>12}{"penalty":>9}{"interval":>18}{"real gain":>11}{"interval":>18}'
          f'{"corrected":>11}{"interval":>18}')
    summary = []
    for k in refs:
        sel = by_ref[k]
        pen = np.array([r['penalty'] for r in sel])
        gr = np.array([r['gain_real'] for r in sel])
        gc = np.array([r['gain_corrected'] for r in sel])
        sh = np.mean([math.hypot(r['shift_dy'], r['shift_dx']) for r in sel])
        sim = np.mean([r['shifted_vs_real_psnr'] for r in sel if k != 0]) if k else float('nan')
        summary.append({'ref': k, 'mean_shift_px': sh, 'penalty': pen.mean(), 'penalty_ci': bootstrap(pen),
                        'gain_real': gr.mean(), 'gain_real_ci': bootstrap(gr), 'gain_corrected': gc.mean(),
                        'gain_corrected_ci': bootstrap(gc), 'shifted_vs_real_psnr': sim})
        s = summary[-1]
        print(f'{k:>14}{sh:>10.1f}{sim:>12.1f}{s["penalty"]:>+9.3f}{"[%+.3f, %+.3f]" % tuple(s["penalty_ci"]):>18}'
              f'{s["gain_real"]:>+11.3f}{"[%+.3f, %+.3f]" % tuple(s["gain_real_ci"]):>18}'
              f'{s["gain_corrected"]:>+11.3f}{"[%+.3f, %+.3f]" % tuple(s["gain_corrected_ci"]):>18}')
    for p in PROBES:
        pen = np.array([r['penalty'] for r in by_ref[p]])
        summary.append({'ref': p, 'penalty': pen.mean(), 'penalty_ci': bootstrap(pen)})
        print(f'{p:>14}{math.hypot(*PROBES[p]):>10.1f}{"":>12}{pen.mean():>+9.3f}{"[%+.3f, %+.3f]" % tuple(bootstrap(pen)):>18}')

    if args.reverse_check:
        print('\nReverse check: corrected gain estimated two ways (95% bootstrap over bursts)')
        print(f'{"ref":>6}{"O_0 moved to k":>16}{"O_k moved to 0":>16}{"difference":>12}{"interval":>18}{"per-burst r":>13}')
        for k in refs[1:]:
            sel = by_ref[k]
            a = np.array([r['gain_corrected'] for r in sel])
            b = np.array([r['gain_reverse'] for r in sel])
            r_ = float(np.corrcoef(a, b)[0, 1])
            summary.append({'ref': f'reverse_{k}', 'gain_corrected': a.mean(), 'gain_reverse': b.mean(),
                            'difference': (b - a).mean(), 'difference_ci': bootstrap(b - a), 'per_burst_r': r_})
            print(f'{k:>6}{a.mean():>+16.3f}{b.mean():>+16.3f}{(b - a).mean():>+12.3f}'
                  f'{"[%+.3f, %+.3f]" % tuple(bootstrap(b - a)):>18}{r_:>13.2f}')

    # Oracle per burst, before and after the correction
    bursts = sorted({r['burst'] for r in rows})
    per = {(r['burst'], r['ref']): r for r in rows if r['ref'] in refs}
    oracle_real = np.array([max(per[(b, k)]['gain_real'] for k in refs) for b in bursts])
    oracle_corr = np.array([max(per[(b, k)]['gain_corrected'] for k in refs) for b in bursts])
    print(f'\nOracle gain over frame 0: measured {oracle_real.mean():+.3f} {bootstrap(oracle_real)}, '
          f'corrected {oracle_corr.mean():+.3f} {bootstrap(oracle_corr)}')
    best = [max(refs, key=lambda k: per[(b, k)]['gain_corrected']) for b in bursts]
    print('Best frame after correction (frame:bursts): ' + ' '.join(f'{k}:{best.count(k)}' for k in refs))

    with open(os.path.join(args.out_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for s in summary for k in s}))
        writer.writeheader()
        for s in summary:
            writer.writerow({k: (' '.join(f'{x:.4f}' for x in v) if isinstance(v, list) else v) for k, v in s.items()})
        writer.writerow({'ref': 'oracle_measured', 'gain_real': oracle_real.mean()})
        writer.writerow({'ref': 'oracle_corrected', 'gain_corrected': oracle_corr.mean()})
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
