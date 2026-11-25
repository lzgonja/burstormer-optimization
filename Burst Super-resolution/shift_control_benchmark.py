"""Known-answer benchmark for the shift control: how accurately does each estimator recover the corrected gain?

On real bursts the true corrected gain of a reference frame is unknown, so the estimators can only be checked
against each other. This script builds frame-k outputs whose answer is known:

    1. Run the model with reference 0 (O_0) and with each reference k (O_k), as the control does.
    2. Measure the real geometry from O_0 to O_k: dense optical flow between the two outputs, kept where the flow
       both ways agrees within 1 px (elsewhere the homography fit), then strongly smoothed.
    3. Change the content of O_0 in a known way, C = D(O_0): nothing, Gaussian blur, or noise.
    4. Synthetic frame-k output: O_k* = C moved by the real geometry of frame k. Where frame 0's view ends, O_k*
       shows the real frame-k output, as a real output would (those pixels are never scored).
    5. Known answer: score(C as reference 0) - score(O_0 as reference 0). Both are at frame 0's position, so no
       position penalty is involved.
    6. Every estimator sees only O_0, O_k* and the burst, exactly as on a real output, and its estimate is
       compared with the known answer.

Estimators: for each geometry model (translation, affine, homography, dense),
    forward  move O_0 to frame k's position, score it as frame k, and add the penalty back (the original method);
    reverse  move O_k* back to frame 0's position and score it as frame 0;
    mean     the average of the two.
All of them, and the known answer, are scored on one region per burst and frame: the pixels every image covers.

Every estimate is checked with all three BurstSR metrics (PSNR, SSIM, LPIPS), signed as in shift_control.py: a
positive gain is an improvement, so LPIPS gains are reductions. PSNR keeps the original column names (errors in dB);
the SSIM and LPIPS columns carry an `ssim_` or `lpips_` prefix and are in those metrics' own units.

The warp used to build O_k* has a small loss of its own. It is measured by moving C there and back
(column roundtrip_loss) and must stay well below the errors being compared.

Example:
    python shift_control_benchmark.py --bursts_file five_bursts.txt --out_dir ./Results/Real/shift_benchmark
"""

import argparse
import csv
import math
import os
from typing import Dict, List

import numpy as np
import torch

from datasets.burstsr_dataset import BurstSRDataset
from pwcnet.pwcnet import PWCNet
from reference_frame_sweep import NUM_FRAMES, FastAlignedMetrics, load_model, reorder, select_bursts, set_seed
from shift_control import DECIMALS, LABELS, METRICS, SIGN, col
from shift_geometry import (MODELS, _gauss_blur, consistent_dense, coverage, estimate_geometry, invert,
                            metric_patch_masks, move_reference, warp_image)

CHANGES = {'none': None, 'blur0.5': ('blur', 0.5), 'blur1.0': ('blur', 1.0), 'noise0.01': ('noise', 0.01)}
BUCKETS = [(0, 8), (8, 24), (24, 60), (60, 1e9)]


def change(img: torch.Tensor, spec, seed: int) -> torch.Tensor:
    if spec is None:
        return img
    kind, amount = spec
    if kind == 'blur':
        return _gauss_blur(img, amount).clamp(0.0, 1.0)
    gen = torch.Generator(device=img.device).manual_seed(seed)
    return (img + amount * torch.randn(img.shape, generator=gen, device=img.device)).clamp(0.0, 1.0)


def bootstrap(x: np.ndarray, n: int = 10000, seed: int = 0) -> List[float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description='Known-answer benchmark for the shift-control estimators')
    parser.add_argument('--weights', default='./Trained_models/Real/burstormer_burstsr_train.pth', type=str)
    parser.add_argument('--data_root', default='./burstsr_dataset', type=str)
    parser.add_argument('--pwcnet', default='./pwcnet/pwcnet-network-default.pth', type=str)
    parser.add_argument('--out_dir', default='./Results/Real/shift_benchmark', type=str)
    parser.add_argument('--refs', default=14, type=int, help='Frames 1..refs-1 are tested')
    parser.add_argument('--bursts_file', default=None, type=str)
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument('--models', default=','.join(MODELS), type=str)
    parser.add_argument('--changes', default=','.join(CHANGES), type=str)
    parser.add_argument('--factor', default=8, type=int, help='Upsampling factor of the warp')
    parser.add_argument('--fill', default='target', choices=('target', 'mirror'),
                        help='How the strip uncovered by a move is filled (as in shift_control.py)')
    parser.add_argument('--true_smooth', default=16.0, type=float,
                        help='Smoothing (output px) of the dense flow used as the true geometry')
    args = parser.parse_args()

    set_seed(50)
    os.makedirs(args.out_dir, exist_ok=True)
    flow_net = PWCNet(load_pretrained=True, weights_path=args.pwcnet).cuda()
    metrics = FastAlignedMetrics(flow_net)
    model = load_model(args.weights)
    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                             random_flip=False, ref_id=0, shuffle_frames=False)
    indices = select_bursts(dataset.burst_list, args.bursts_file, args.limit)
    models = args.models.split(',')
    changes = args.changes.split(',')
    frames = list(range(1, args.refs))

    est_names = [f'{d}_{m}' for m in models for d in ('fwd', 'rev')]
    fields = ['burst', 'frame', 'change', 'shift_px', 'true_residual_px', 'flow_replaced_frac']
    for mt in METRICS:
        fields += [col(mt, 'truth'), col(mt, 'roundtrip_loss')] + [col(mt, f'err_{e}') for e in est_names]
    fields += [f'fidelity_{m}' for m in models] + ['region_frac']
    rows: List[Dict] = []
    path = os.path.join(args.out_dir, 'per_frame.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n_done, idx in enumerate(indices):
            burst, gt, _, _, name = dataset[idx]
            burst, gt = burst[None].cuda(), gt[None].cuda()
            with torch.no_grad():
                o0 = model(burst).clamp(0.0, 1.0)
                outs = {k: model(reorder(burst, k)).clamp(0.0, 1.0) for k in frames}
            true_geo = {k: consistent_dense(o0, outs[k], flow_net, smooth=args.true_smooth)[0] for k in frames}
            h, w = o0.shape[-2:]
            base_al = metrics.align_patches(o0, gt, burst)

            for c_name in changes:
                content = change(o0, CHANGES[c_name], seed=idx)
                truth_al = metrics.align_patches(content, gt, burst)
                for k in frames:
                    g = true_geo[k]
                    # A real frame-k output shows real scene content where frame 0's view ends; so does O_k*
                    star = warp_image(content, g, args.factor, fill=outs[k])
                    target = args.fill == 'target'
                    masks = [metric_patch_masks(coverage(g, h, w, o0.device))]
                    # O_k* holds frame 0's content and colours at frame k's position, so its colour reference is raw
                    # frame 0 moved there too (shift_control.py --colour_ref moved does the same for every moved output)
                    rb = move_reference(burst, burst[:, 0], g, burst[:, k])
                    star_al = metrics.align_patches(star, gt, rb)
                    fwd_al, rev_al, fidelity = {}, {}, {}
                    for m in models:
                        gf = estimate_geometry(o0, star, m, flow_net)
                        moved = warp_image(o0, gf, args.factor, fill=star if target else None)
                        fwd_al[m] = metrics.align_patches(moved, gt, move_reference(burst, burst[:, 0], gf, rb[:, 0]))
                        mse = torch.mean((moved[..., 64:-64, 64:-64] - star[..., 64:-64, 64:-64]) ** 2).item()
                        fidelity[m] = -10 * math.log10(mse + 1e-12)
                        gr = estimate_geometry(star, o0, m, flow_net)
                        rev_al[m] = metrics.align_patches(warp_image(star, gr, args.factor, fill=o0 if target else None),
                                                          gt, move_reference(burst, rb[:, 0], gr, burst[:, 0]))
                        masks += [metric_patch_masks(coverage(gf, h, w, o0.device)),
                                  metric_patch_masks(coverage(gr, h, w, o0.device))]
                    rt_al = None
                    if c_name == 'none':
                        rt = warp_image(star, invert(g), args.factor, fill=content if target else None)
                        rt_al = metrics.align_patches(rt, gt, move_reference(burst, rb[:, 0], invert(g), burst[:, 0]))

                    all_al = [base_al, truth_al, star_al] + list(fwd_al.values()) + list(rev_al.values()) + \
                        ([rt_al] if rt_al is not None else [])
                    region = [torch.stack([al[p]['valid'] & al[p]['in_view'] for al in all_al] +
                                          [mk[p].to(base_al[p]['valid'].device) for mk in masks]).all(dim=0)
                              for p in range(len(base_al))]
                    region_frac = float(np.mean([r.float().mean().item() for r in region]))

                    def s(al):
                        return dict(zip(METRICS, metrics.score(al, regions=region)))

                    base, truth_s, star_s = s(base_al), s(truth_al), s(star_al)
                    fwd_s = {m: s(fwd_al[m]) for m in models}
                    rev_s = {m: s(rev_al[m]) for m in models}
                    rt_s = s(rt_al) if rt_al is not None else None
                    row = {'burst': name, 'frame': k, 'change': c_name,
                           'shift_px': float(g.displacement(h, w).norm(dim=0).mean()),
                           'true_residual_px': g.max_residual(), 'flow_replaced_frac': g.info['replaced_frac'],
                           'region_frac': region_frac}
                    for mt in METRICS:
                        sg = SIGN[mt]
                        truth = sg * (truth_s[mt] - base[mt])
                        row[col(mt, 'truth')] = truth
                        row[col(mt, 'roundtrip_loss')] = sg * (truth_s[mt] - rt_s[mt]) if rt_s is not None else ''
                        for m in models:
                            fwd = sg * (star_s[mt] - fwd_s[m][mt])  # = real gain + penalty
                            rev = sg * (rev_s[m][mt] - base[mt])
                            row[col(mt, f'err_fwd_{m}')] = fwd - truth
                            row[col(mt, f'err_rev_{m}')] = rev - truth
                    for m in models:
                        row[f'fidelity_{m}'] = fidelity[m]
                    rows.append(row)
                    writer.writerow(row)
                f.flush()
                print(f'[{n_done + 1}/{len(indices)}] {name} {c_name}', flush=True)

    summarise(rows, models, args.out_dir)


def summarise(rows: List[Dict], models: List[str], out_dir: str) -> None:
    ests = [f'{d}_{m}' for m in models for d in ('fwd', 'rev', 'mean')]
    changes = sorted({r['change'] for r in rows})
    rms = lambda x: float(np.sqrt((np.asarray(x) ** 2).mean()))
    print(f'\n{len(rows)} burst-frame-change cases. Error = estimate - known answer, in each metric\'s units.')
    out = []
    for mt in METRICS:
        d = DECIMALS[mt] + 1

        def errs(sel, e):
            kind, m = e.split('_', 1)
            if kind == 'mean':
                return np.array([(r[col(mt, f'err_fwd_{m}')] + r[col(mt, f'err_rev_{m}')]) / 2 for r in sel])
            return np.array([r[col(mt, f'err_{e}')] for r in sel])

        rt = np.array([r[col(mt, 'roundtrip_loss')] for r in rows if r[col(mt, 'roundtrip_loss')] != ''])
        print(f'\n=== {LABELS[mt]} ===')
        print(f'Warp round-trip loss: mean {rt.mean():+.{d}f}, max |{np.abs(rt).max():.{d}f}|. '
              f'Size of the known answer (RMS), for scale: ' +
              ', '.join(f'{c} {rms([r[col(mt, "truth")] for r in rows if r["change"] == c]):.{d}f}' for c in changes))
        print(f'{"estimator":>18}{"bias":>10}{"interval":>22}{"RMSE":>10}{"max|err|":>10}' +
              ''.join(f'{f"RMSE {lo:.0f}-{hi:.0f}px" if hi < 1e9 else f"RMSE {lo:.0f}+px":>15}' for lo, hi in BUCKETS))
        for e in ests:
            x = errs(rows, e)
            rec = {'metric': mt, 'estimator': e, 'bias': x.mean(), 'bias_ci': bootstrap(x), 'rmse': rms(x),
                   'max_abs': float(np.abs(x).max())}
            line = f'{e:>18}{x.mean():>+10.{d}f}{"[%+.*f, %+.*f]" % (d, rec["bias_ci"][0], d, rec["bias_ci"][1]):>22}' \
                   f'{rec["rmse"]:>10.{d}f}{rec["max_abs"]:>10.{d}f}'
            for lo, hi in BUCKETS:
                sel = [r for r in rows if lo <= r['shift_px'] < hi]
                v = rms(errs(sel, e)) if sel else float('nan')
                rec[f'rmse_{lo:.0f}'] = v
                line += f'{v:>15.{d}f}'
            print(line)
            out.append(rec)
        print(f'\nRMSE by content change, {LABELS[mt]}:')
        print(f'{"estimator":>18}' + ''.join(f'{c:>12}' for c in changes))
        for e in ests:
            print(f'{e:>18}' + ''.join(f'{rms(errs([r for r in rows if r["change"] == c], e)):>12.{d}f}'
                                        for c in changes))
    print('\nGeometric fidelity (moved O_0 vs O_k*, dB, mean):  ' +
          '  '.join(f'{m} {np.mean([r[f"fidelity_{m}"] for r in rows]):.1f}' for m in models))
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        for rec in out:
            writer.writerow({k: (' '.join(f'{x:.5f}' for x in v) if isinstance(v, list) else v) for k, v in rec.items()})


if __name__ == '__main__':
    main()
