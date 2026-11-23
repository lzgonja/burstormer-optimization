"""Shift control: how much does the BurstSR metric penalise a reference frame just for its position?

The ground truth is registered to frame 0. An output made with frame k as reference is displaced from it, so the
metric must warp it further (optical flow over a larger displacement, content near the patch edges). This script
separates that penalty from real reconstruction quality:

    1. Run the model with references 0..K-1 (outputs O_0 ... O_k).
    2. Estimate the geometry between O_0 and each O_k (--geometry: a translation, an affine map, a homography, or
       dense optical flow; see shift_geometry.py). The homography falls back to the global shift when it does not
       fit its own tiles. Images are moved with an exact Fourier shift plus 8x band-limited resampling of the rest,
       which keeps detail and noise level (tests/test_shift_geometry.py). The strip a move uncovers is filled with
       the image moved onto and left out of the score (--fill target), and the moved output's own raw frame moves
       with it as its colour reference (--colour_ref moved), so that only the position changes.
    3. Reverse estimate (the default, --estimator reverse): move O_k back to frame 0's position and score it as
       reference 0. Frame k's own reconstruction is then scored without its position, so
           corrected gain = PSNR(O_k moved to frame 0, as ref 0) - PSNR(O_0, as ref 0).
    4. Forward estimate (the original control): move O_0 to frame k's position and score it as reference k. Its loss
       is the penalty for position, measured on frame 0's content, and is added back to frame k's real gain:
           corrected gain = PSNR(O_k, as ref k) - PSNR(O_0 moved to frame k, as ref k).

Which estimate is right was measured with known answers (shift_control_benchmark.py,
docs/SHIFT_CONTROL_VALIDATION.md). On 9 bursts with shifts up to 110 px, the reverse estimate is accurate to
0.068 dB RMS (bias -0.011 dB). The forward estimate is off by 0.385 dB RMS when frame k's content differs from frame
0's, because the penalty depends on the content being moved. Both are always computed. `penalty` is corrected gain minus measured gain.

Region (--region): `pair` scores frame k and frame 0 on the pixels valid and in view for every image of that
comparison, so a frame's result does not depend on which other frames are in the run. `all` uses the pixels common
to every image of the burst, as the original control did.

Two probes check the metric itself: O_0 moved by half an output pixel (resampling only) and by 8 x 8 output pixels
(displacement only). For the probes, input frame 0 is moved by the same amount (at raw scale), because the metric
colour-matches with it and it must stay in register with the output.

The original control is `--geometry translation --region all --estimator forward --fill mirror --colour_ref fixed`
(with the warp now unclipped, it reproduces the 2026-11-22 numbers to within 0.012 dB).

Example:
    python shift_control.py --bursts_file ./splits/dev.txt --limit 40 --refs 14 --out_dir ./Results/Real/sc_dev
"""

import argparse
import csv
import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from reference_selection import NUM_FRAMES, reorder_burst as reorder
from shift_geometry import (MODELS, PAD, Geometry, coverage, estimate_geometry, metric_patch_masks,  # noqa: F401
                            move_reference, shift_image, translation, warp_image)

PROBES = {'probe_half_px': (0.5, 0.5), 'probe_whole_px': (8.0, 8.0)}


def evaluate_reference_pair(outputs, reference, burst, gt, metrics, flow_net, geometry='homography', factor=8,
                            colour_ref='moved', fill='target'):
    """Return v2 reverse-corrected PSNR gain over frame 0 on one pair-valid region.

    This is the default reverse-correction path from :func:`main`, exposed so an
    experiment can reuse its reconstructions instead of running a separate sweep.
    """
    if reference == 0:
        return 0.0

    o0 = outputs[0]
    base_al = metrics.align_patches(o0, gt, burst)
    selected = outputs[reference]
    fwd_geo = estimate_geometry(o0, selected, geometry, flow_net)
    rev_geo = Geometry(-fwd_geo.shift) if geometry == 'translation' else \
        estimate_geometry(selected, o0, geometry, flow_net)
    target = fill == 'target'
    fwd_img = warp_image(o0, fwd_geo, factor, fill=selected if target else None)
    rev_img = warp_image(selected, rev_geo, factor, fill=o0 if target else None)
    burst_ref = reorder(burst, reference)
    real_al = metrics.align_patches(selected, gt, burst_ref)
    moved_ref = colour_ref == 'moved'
    fwd_al = metrics.align_patches(
        fwd_img, gt,
        move_reference(burst, burst[:, 0], fwd_geo, burst[:, reference]) if moved_ref else burst_ref)
    rev_al = metrics.align_patches(
        rev_img, gt,
        move_reference(burst, burst[:, reference], rev_geo, burst[:, 0]) if moved_ref else burst)

    masks = []
    if target:
        h, w = o0.shape[-2:]
        masks.append(metric_patch_masks(coverage(fwd_geo, h, w, o0.device) &
                                        coverage(rev_geo, h, w, o0.device)))
    regions = [torch.stack(
        [al[p]['valid'] & al[p]['in_view'] for al in (base_al, real_al, fwd_al, rev_al)] +
        [mask[p].to(base_al[p]['valid'].device) for mask in masks]
    ).all(dim=0) for p in range(len(base_al))]
    psnr_frame0 = metrics.score(base_al, regions=regions)[0]
    psnr_reverse = metrics.score(rev_al, regions=regions)[0]
    return psnr_reverse - psnr_frame0


def evaluate_same_reference_pair(baseline, candidate, burst, gt, metrics):
    """PSNR gain for two frame-0-view outputs scored on their shared valid region."""
    base_al = metrics.align_patches(baseline, gt, burst)
    candidate_al = metrics.align_patches(candidate, gt, burst)
    regions = [base['valid'] & base['in_view'] & selected['valid'] & selected['in_view']
               for base, selected in zip(base_al, candidate_al)]
    psnr_frame0 = metrics.score(base_al, regions=regions)[0]
    psnr_candidate = metrics.score(candidate_al, regions=regions)[0]
    return psnr_candidate - psnr_frame0


def bootstrap(x: np.ndarray, n: int = 10000, seed: int = 0) -> List[float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def fidelity(a: torch.Tensor, b: torch.Tensor, margin: int) -> float:
    """PSNR between two outputs away from the edges: how well a moved output matches the one it was moved onto."""
    crop = (Ellipsis, slice(margin, -margin), slice(margin, -margin))
    return -10 * math.log10(F.mse_loss(a[crop], b[crop]).item() + 1e-12)


def main() -> None:
    from datasets.burstsr_dataset import BurstSRDataset
    from pwcnet.pwcnet import PWCNet
    from reference_frame_sweep import FastAlignedMetrics, load_model, select_bursts, set_seed

    parser = argparse.ArgumentParser(description='Measure the metric penalty for the position of the reference frame')
    parser.add_argument('--weights', default='./Trained_models/Real/burstormer_burstsr_train.pth', type=str)
    parser.add_argument('--data_root', default='./burstsr_dataset', type=str)
    parser.add_argument('--pwcnet', default='./pwcnet/pwcnet-network-default.pth', type=str)
    parser.add_argument('--out_dir', default='./Results/Real/shift_control', type=str)
    parser.add_argument('--refs', default=5, type=int, help='References 0..refs-1')
    parser.add_argument('--bursts_file', default=None, type=str)
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument('--geometry', default='homography', choices=MODELS,
                        help='How an output is moved to the position of another frame (see shift_geometry.py)')
    parser.add_argument('--estimator', default='reverse', choices=('reverse', 'forward'),
                        help='Which estimate is reported as gain_corrected (both are always computed)')
    parser.add_argument('--region', default='pair', choices=('pair', 'all'),
                        help='pair: score frame k and frame 0 on the pixels common to that comparison; '
                             'all: on the pixels common to every image of the burst')
    parser.add_argument('--factor', default=8, type=int, help='Upsampling factor of the warp (non-translation)')
    parser.add_argument('--colour_ref', default='moved', choices=('moved', 'fixed'),
                        help='moved: score a moved output with its own raw reference frame moved along with it; '
                             'fixed: with the raw frame of the position it was moved to, as in the first v2 runs')
    parser.add_argument('--fill', default='target', choices=('target', 'mirror'),
                        help='target: fill the strip a move uncovers with the real content of the image moved onto, '
                             'and leave it out of the score; mirror: mirrored content, as in the first v2 runs')
    parser.add_argument('--reverse_check', action='store_true', help='No effect: both estimates are always computed')
    args = parser.parse_args()

    set_seed(50)
    os.makedirs(args.out_dir, exist_ok=True)
    flow_net = PWCNet(load_pretrained=True, weights_path=args.pwcnet).cuda()
    metrics = FastAlignedMetrics(flow_net)
    model = load_model(args.weights)
    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                             random_flip=False, ref_id=0, shuffle_frames=False)
    indices = select_bursts(dataset.burst_list, args.bursts_file, args.limit)
    refs = list(range(args.refs))

    rows: List[Dict] = []
    path = os.path.join(args.out_dir, 'per_burst.csv')
    fields = ['burst', 'ref', 'shift_dy', 'shift_dx', 'residual_max_px', 'psnr_frame0', 'psnr_real', 'psnr_forward',
              'psnr_reverse', 'gain_real', 'gain_forward', 'gain_reverse', 'gain_corrected', 'penalty',
              'fidelity_forward', 'fidelity_reverse', 'geometry_fallback', 'common_frac']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n_done, idx in enumerate(indices):
            burst, gt, _, _, name = dataset[idx]
            burst, gt = burst[None].cuda(), gt[None].cuda()
            with torch.no_grad():
                outputs = {k: model(reorder(burst, k)).clamp(0.0, 1.0) for k in refs}
            o0 = outputs[0]
            fwd_geo, rev_geo, fwd_img, rev_img, cover = {}, {}, {}, {}, {}
            h, w = o0.shape[-2:]
            for k in refs[1:]:
                fwd_geo[k] = estimate_geometry(o0, outputs[k], args.geometry, flow_net)
                # Moving O_k back: estimated in that direction (for a translation, the forward shift reversed)
                rev_geo[k] = Geometry(-fwd_geo[k].shift) if args.geometry == 'translation' else \
                    estimate_geometry(outputs[k], o0, args.geometry, flow_net)
                target = args.fill == 'target'
                fwd_img[k] = warp_image(o0, fwd_geo[k], args.factor, fill=outputs[k] if target else None)
                rev_img[k] = warp_image(outputs[k], rev_geo[k], args.factor, fill=o0 if target else None)
                # Pixels holding moved content (not fill), per metric patch; only these are scored. With mirror
                # fill the original behaviour is kept (no extra mask)
                if target:
                    both = coverage(fwd_geo[k], h, w, o0.device) & coverage(rev_geo[k], h, w, o0.device)
                    cover[k] = metric_patch_masks(both)

            # Align every image once. Images at frame k's position are scored with the burst reordered to k
            real_al = {k: metrics.align_patches(outputs[k], gt, reorder(burst, k)) for k in refs}
            # Colour reference: the moved output's own raw frame, moved along with it (or the raw frame of the new
            # position, --colour_ref fixed)
            moved_ref = args.colour_ref == 'moved'
            fwd_al = {k: metrics.align_patches(fwd_img[k], gt, move_reference(burst, burst[:, 0], fwd_geo[k], burst[:, k]) if moved_ref
                                               else reorder(burst, k)) for k in refs[1:]}
            rev_al = {k: metrics.align_patches(rev_img[k], gt, move_reference(burst, burst[:, k], rev_geo[k], burst[:, 0]) if moved_ref
                                               else burst) for k in refs[1:]}
            probe_al = {}
            for p, (dy, dx) in PROBES.items():
                moved = burst.clone()
                moved[0, 0] = shift_image(burst[:, 0], dy / 8, dx / 8)[0]
                probe_al[p] = metrics.align_patches(shift_image(o0, dy, dx), gt, moved)
            n_patches = len(real_al[0])

            def region(als, masks=()):
                return [torch.stack([al[p]['valid'] & al[p]['in_view'] for al in als] +
                                    [m[p].to(als[0][p]['valid'].device) for m in masks]).all(dim=0)
                        for p in range(n_patches)]

            common = region(list(real_al.values()) + list(fwd_al.values()) + list(rev_al.values()) +
                            list(probe_al.values()), list(cover.values()))
            regions = {}
            for k in refs + list(PROBES):
                if args.region == 'all':
                    regions[k] = common
                elif k in PROBES:
                    regions[k] = region([real_al[0], probe_al[k]])
                elif k == 0:
                    regions[k] = region([real_al[0]])
                else:
                    regions[k] = region([real_al[0], real_al[k], fwd_al[k], rev_al[k]], [cover[k]] if k in cover else [])

            def psnr(al, k):
                return metrics.score(al, regions=regions[k])[0]

            for k in refs:
                frac = float(np.mean([c.float().mean().item() for c in regions[k]]))
                p0, pk = psnr(real_al[0], k), psnr(real_al[k], k)
                row = {'burst': name, 'ref': k, 'psnr_frame0': p0, 'psnr_real': pk, 'gain_real': pk - p0,
                       'common_frac': frac}
                if k == 0:
                    row.update({'shift_dy': 0.0, 'shift_dx': 0.0, 'residual_max_px': 0.0, 'psnr_forward': p0,
                                'psnr_reverse': p0, 'gain_forward': 0.0, 'gain_reverse': 0.0})
                else:
                    pf, pr = psnr(fwd_al[k], k), psnr(rev_al[k], k)
                    margin = int(math.ceil(fwd_geo[k].shift.abs().max().item() + fwd_geo[k].max_residual())) + 8
                    row.update({'shift_dy': fwd_geo[k].shift[0].item(), 'shift_dx': fwd_geo[k].shift[1].item(),
                                'residual_max_px': fwd_geo[k].max_residual(), 'psnr_forward': pf, 'psnr_reverse': pr,
                                'gain_forward': pk - pf, 'gain_reverse': pr - p0,
                                'fidelity_forward': fidelity(fwd_img[k], outputs[k], margin),
                                'fidelity_reverse': fidelity(rev_img[k], o0, margin),
                                'geometry_fallback': int(fwd_geo[k].info.get('fallback', False)) +
                                int(rev_geo[k].info.get('fallback', False))})
                row['gain_corrected'] = row[f'gain_{args.estimator}']
                row['penalty'] = row['gain_corrected'] - row['gain_real']
                rows.append(row)
                writer.writerow(row)
            for p in PROBES:
                p0 = psnr(real_al[0], p)
                row = {'burst': name, 'ref': p, 'psnr_frame0': p0, 'psnr_forward': psnr(probe_al[p], p),
                       'penalty': p0 - psnr(probe_al[p], p),
                       'common_frac': float(np.mean([c.float().mean().item() for c in regions[p]]))}
                rows.append(row)
                writer.writerow(row)
            f.flush()
            print(f'[{n_done + 1}/{len(indices)}] {name}', flush=True)

    summarise(rows, refs, args, path)


def summarise(rows: List[Dict], refs: List[int], args, path: str) -> None:
    by_ref = {k: [r for r in rows if r['ref'] == k] for k in refs + list(PROBES)}
    n_bursts = len(by_ref[0])
    region_text = 'common to every image' if args.region == 'all' else 'common to each comparison'
    print(f'\n{n_bursts} bursts. Geometry {args.geometry}, {args.estimator} estimate, PSNR on the pixels '
          f'{region_text} ({np.mean([r["common_frac"] for r in rows]):.0%}). Penalty = corrected - measured gain. '
          f'95% bootstrap over bursts.')
    print(f'{"ref":>14}{"shift px":>10}{"fidelity":>10}{"penalty":>9}{"interval":>18}{"measured":>10}{"interval":>18}'
          f'{"corrected":>11}{"interval":>18}')
    summary = []
    for k in refs:
        sel = by_ref[k]
        pen, gr, gc = (np.array([r[c] for r in sel]) for c in ('penalty', 'gain_real', 'gain_corrected'))
        sh = np.mean([math.hypot(r['shift_dy'], r['shift_dx']) for r in sel])
        fid = np.mean([r[f'fidelity_{args.estimator}'] for r in sel]) if k else float('nan')
        s = {'ref': k, 'mean_shift_px': sh, 'fidelity': fid, 'penalty': pen.mean(), 'penalty_ci': bootstrap(pen),
             'gain_real': gr.mean(), 'gain_real_ci': bootstrap(gr), 'gain_corrected': gc.mean(),
             'gain_corrected_ci': bootstrap(gc)}
        if k:
            gf = np.array([r['gain_forward'] for r in sel])
            gv = np.array([r['gain_reverse'] for r in sel])
            s.update({'gain_forward': gf.mean(), 'gain_reverse': gv.mean(), 'reverse_minus_forward': (gv - gf).mean(),
                      'reverse_minus_forward_ci': bootstrap(gv - gf),
                      'per_burst_r': float(np.corrcoef(gf, gv)[0, 1]) if len(sel) > 2 else float('nan')})
        summary.append(s)
        print(f'{k:>14}{sh:>10.1f}{fid:>10.1f}{s["penalty"]:>+9.3f}{"[%+.3f, %+.3f]" % tuple(s["penalty_ci"]):>18}'
              f'{s["gain_real"]:>+10.3f}{"[%+.3f, %+.3f]" % tuple(s["gain_real_ci"]):>18}'
              f'{s["gain_corrected"]:>+11.3f}{"[%+.3f, %+.3f]" % tuple(s["gain_corrected_ci"]):>18}')
    for p in PROBES:
        pen = np.array([r['penalty'] for r in by_ref[p]])
        summary.append({'ref': p, 'penalty': pen.mean(), 'penalty_ci': bootstrap(pen)})
        print(f'{p:>14}{math.hypot(*PROBES[p]):>10.1f}{"":>10}{pen.mean():>+9.3f}'
              f'{"[%+.3f, %+.3f]" % tuple(bootstrap(pen)):>18}')

    print('\nForward vs reverse estimate of the corrected gain (diagnostic; the benchmark favours reverse)')
    print(f'{"ref":>6}{"forward":>10}{"reverse":>10}{"difference":>12}{"interval":>18}{"per-burst r":>13}')
    for s in summary:
        if isinstance(s['ref'], int) and s['ref'] > 0:
            print(f'{s["ref"]:>6}{s["gain_forward"]:>+10.3f}{s["gain_reverse"]:>+10.3f}'
                  f'{s["reverse_minus_forward"]:>+12.3f}{"[%+.3f, %+.3f]" % tuple(s["reverse_minus_forward_ci"]):>18}'
                  f'{s["per_burst_r"]:>13.2f}')

    bursts = sorted({r['burst'] for r in rows})
    per = {(r['burst'], r['ref']): r for r in rows if r['ref'] in refs}
    oracle_real = np.array([max(per[(b, k)]['gain_real'] for k in refs) for b in bursts])
    oracle_corr = np.array([max(per[(b, k)]['gain_corrected'] for k in refs) for b in bursts])
    print(f'\nBest frame per burst, gain over frame 0: measured {oracle_real.mean():+.3f} '
          f'[{bootstrap(oracle_real)[0]:+.3f}, {bootstrap(oracle_real)[1]:+.3f}], corrected {oracle_corr.mean():+.3f} '
          f'[{bootstrap(oracle_corr)[0]:+.3f}, {bootstrap(oracle_corr)[1]:+.3f}]')
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
