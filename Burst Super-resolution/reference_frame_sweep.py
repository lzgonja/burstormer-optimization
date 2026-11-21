"""Reference-frame sweep for Burstormer on the real BurstSR validation set.

The original evaluation always uses raw frame 0 as the reference. This script evaluates every burst once per
candidate reference frame (the chosen frame is placed first; the remaining frames stay in capture order) and
records aligned PSNR / SSIM / LPIPS, plus a cheap no-reference sharpness score for each reference frame.

Outputs (in --out_dir):
    per_burst.csv  one row per (burst, reference frame)
    summary.csv    mean metrics per reference frame, plus selection policies (oracle, sharpest frame)

Example:
    python reference_frame_sweep.py --weights ./Trained_models/Real/Burstormer.ckpt
    python reference_frame_sweep.py --weights ... --limit 20 --refs 0 3 7     # quick check
"""

import argparse
import csv
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from Network_Real_burstSR import Base_Model
from datasets.burstsr_dataset import BurstSRDataset
from pwcnet.pwcnet import PWCNet
from ref_selection import GUARD_MARGIN, guarded_choice
import utils.spatial_color_alignment as sca_utils
from utils.metrics import make_patches
from utils.warp import warp

NUM_FRAMES = 14


class FastAlignedMetrics:
    """Same numbers as utils.metrics.AlignedPSNR / AlignedSSIM / AlignedLPIPS, computed ~10x faster.

    The originals each split the output into patches and each re-run PWC-Net alignment + colour matching per patch
    (3x redundant), then compute SSIM with scipy and LPIPS on the CPU. Here every patch is aligned once and all three
    metrics are computed from that on the GPU. Equivalence is checked by `--verify_metrics`.
    """

    def __init__(self, alignment_net: torch.nn.Module) -> None:
        self.alignment_net = alignment_net
        self.gauss_kernel, self.ksz = sca_utils.get_gaussian_kernel(sd=1.5)
        self.lpips_net = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
        gauss = cv2.getGaussianKernel(11, 1.5)
        self.ssim_window = torch.from_numpy(gauss * gauss.T).cuda().view(1, 1, 11, 11)  # float64, as in utils/ssim.py

    def _ssim_map(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """GPU port of utils.ssim.cal_ssim (zero-padded 'same' convolution, float64)."""
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        img1, img2 = img1.double()[None, None], img2.double()[None, None]
        conv = lambda x: F.conv2d(x, self.ssim_window, padding=5)
        mu1, mu2 = conv(img1), conv(img2)
        s1 = conv(img1 * img1) - mu1 * mu1
        s2 = conv(img2 * img2) - mu2 * mu2
        s12 = conv(img1 * img2) - mu1 * mu2
        return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)))[0, 0]

    @torch.no_grad()
    def align_patches(self, output: torch.Tensor, labels: torch.Tensor, burst: torch.Tensor) -> List[Dict]:
        """Align and colour-match every patch, exactly as utils.metrics.AlignedPred does.

        Returns one dict per patch: pred (colour-matched output warped onto the ground truth), gt, valid (the
        metric's own mask) and in_view (False where the warp sampled outside the output, i.e. ground-truth area the
        output does not cover; the original metric fills those pixels with zeros and may still count them).
        """
        patches = []
        for p, g, b in zip(*make_patches(output, labels, burst)):
            pred, gt, b = p[None], g[None], b[None]
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))
            pred_warped = warp(pred, flow)
            ds_factor = 1.0 / 8.0
            flow_ds = F.interpolate(flow, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear',
                                    align_corners=True) * ds_factor
            burst_0_warped = warp(b[:, 0, [0, 1, 3]].contiguous(), flow_ds)
            gt_ds = F.interpolate(gt, scale_factor=ds_factor, recompute_scale_factor=True, mode='bilinear',
                                  align_corners=True)
            pred_m, valid = sca_utils.match_colors(gt_ds, burst_0_warped, pred_warped, self.ksz, self.gauss_kernel)
            in_view = warp(torch.ones_like(pred[:, :1]), flow) > 0.999
            patches.append({'pred': pred_m, 'gt': gt, 'valid': valid, 'in_view': in_view})
        return patches

    @torch.no_grad()
    def score(self, patches: List[Dict], regions: List[torch.Tensor] = None,
              per_patch: bool = False) -> Tuple[float, float, float]:
        """PSNR / SSIM / LPIPS averaged over patches.

        regions=None reproduces the original metrics (each patch scored on its own valid mask). With `regions` (one
        bool mask per patch) every patch is scored on that mask instead, e.g. the area that is valid for every
        reference frame, so that different references are compared on identical pixels. Patches whose region covers
        under 1% of the patch are skipped; LPIPS is then computed with both images zeroed outside the region.
        """
        psnrs, ssims, lpipss = [], [], []
        for i, patch in enumerate(patches):
            pred_m, gt = patch['pred'], patch['gt']
            region = patch['valid'] if regions is None else regions[i]
            regionf = region.float()
            if regions is not None and regionf.mean() < 0.01:
                psnrs.append(float('nan')), ssims.append(float('nan')), lpipss.append(float('nan'))
                continue

            # AlignedL2 -> AlignedPSNR
            mse = F.mse_loss(pred_m, gt, reduction='none')
            mse = (mse * regionf).sum() / (regionf.sum() * (mse.numel() / region.numel()) + 1e-12)
            psnrs.append(-10.0 * math.log10(mse.item() + 1e-12))

            # AlignedSSIM (first colour channel only, as in the original)
            ssim_map = self._ssim_map(pred_m[0, 0] * 255, gt[0, 0] * 255).float()
            v = regionf.squeeze()
            ssims.append(((ssim_map * v).sum() / (v.sum() * (ssim_map.numel() / v.numel()) + 1e-12)).item())

            # AlignedLPIPS (the original ignores the mask)
            if regions is None:
                lpipss.append(self.lpips_net(2 * pred_m - 1, 2 * gt - 1).item())
            else:
                lpipss.append(self.lpips_net(2 * pred_m * regionf - 1, 2 * gt * regionf - 1).item())
        if per_patch:
            return psnrs, ssims, lpipss
        if regions is None:
            return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipss))
        if all(math.isnan(x) for x in psnrs):
            return float('nan'), float('nan'), float('nan')
        return float(np.nanmean(psnrs)), float(np.nanmean(ssims)), float(np.nanmean(lpipss))

    def __call__(self, output: torch.Tensor, labels: torch.Tensor, burst: torch.Tensor) -> Tuple[float, float, float]:
        return self.score(self.align_patches(output, labels, burst))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_model(weights_path: str) -> Base_Model:
    """Load Burstormer weights from either the original .pth format or a PyTorch Lightning .ckpt."""
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'state_dict'):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    state_dict = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in checkpoint.items()}

    model = Base_Model()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # The model's own loss/metric modules carry no weights we care about; anything else missing is a real problem
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint does not match Base_Model.\nMissing: {missing}\nUnexpected: {unexpected}')
    return model.eval().cuda()


def frame_sharpness(burst: torch.Tensor) -> torch.Tensor:
    """Variance of the Laplacian of the green channel for each frame. burst: [1, N, 4, H, W] packed RGGB."""
    green = burst[0, :, 1:3].mean(dim=1, keepdim=True)  # [N, 1, H, W]
    kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=burst.device).view(1, 1, 3, 3)
    lap = F.conv2d(green, kernel)
    return lap.flatten(1).var(dim=1)  # [N]


def frame_sharpness_smoothed(burst: torch.Tensor) -> torch.Tensor:
    """Mean gradient energy of the green channel after a Gaussian blur (sigma 1), so that pixel noise, which the
    plain Laplacian variance rewards, contributes little. burst: [1, N, 4, H, W]."""
    green = burst[0, :, 1:3].mean(dim=1, keepdim=True)
    x = torch.arange(-3, 4, device=burst.device, dtype=burst.dtype)
    k1 = torch.exp(-x ** 2 / 2.0)
    k1 = k1 / k1.sum()
    green = F.conv2d(green, (k1[:, None] * k1[None, :]).view(1, 1, 7, 7))
    gx = green[..., :, 1:] - green[..., :, :-1]
    gy = green[..., 1:, :] - green[..., :-1, :]
    return gx.pow(2).flatten(1).mean(dim=1) + gy.pow(2).flatten(1).mean(dim=1)  # [N]


def global_shifts(burst: torch.Tensor) -> torch.Tensor:
    """Global translation of every frame relative to frame 0, by phase correlation on the green channel.
    burst: [1, N, 4, H, W]. Returns [N, 2] (dx, dy) in packed raw pixels, with sub-pixel refinement."""
    green = burst[0, :, 1:3].mean(dim=1)  # [N, H, W]
    n, h, w = green.shape
    green = green - green.mean(dim=(1, 2), keepdim=True)
    window = torch.hann_window(h, periodic=False, device=burst.device)[:, None] * \
        torch.hann_window(w, periodic=False, device=burst.device)[None, :]
    spec = torch.fft.fft2(green * window)
    cross = spec * spec[0:1].conj()
    corr = torch.fft.ifft2(cross / (cross.abs() + 1e-8)).real  # [N, H, W], peak at the shift
    peak = corr.flatten(1).argmax(dim=1)
    py, px = peak // w, peak % w

    def refine(c_minus: torch.Tensor, c_0: torch.Tensor, c_plus: torch.Tensor) -> torch.Tensor:
        denom = c_minus - 2 * c_0 + c_plus
        return torch.where(denom.abs() > 1e-8, 0.5 * (c_minus - c_plus) / denom, torch.zeros_like(denom)).clamp(-0.5, 0.5)

    idx = torch.arange(n, device=burst.device)
    c0 = corr[idx, py, px]
    dy = py.float() + refine(corr[idx, (py - 1) % h, px], c0, corr[idx, (py + 1) % h, px])
    dx = px.float() + refine(corr[idx, py, (px - 1) % w], c0, corr[idx, py, (px + 1) % w])
    dy = torch.where(dy > h / 2, dy - h, dy)
    dx = torch.where(dx > w / 2, dx - w, dx)
    return torch.stack([dx, dy], dim=1)


def frame_features(burst: torch.Tensor) -> Dict[str, np.ndarray]:
    """Cheap per-frame features that a reference-frame selector could use (no ground truth, no network)."""
    shifts = global_shifts(burst)
    # Motion centrality: mean distance from this frame's position to every other frame's position (lower = the
    # frame sits in the middle of the hand-shake cloud, so the others need the least warping to reach it)
    centrality = torch.cdist(shifts, shifts).sum(dim=1) / (shifts.shape[0] - 1)
    return {'sharpness': frame_sharpness(burst).cpu().numpy(),
            'sharpness_smooth': frame_sharpness_smoothed(burst).cpu().numpy(),
            'shift_x': shifts[:, 0].cpu().numpy(), 'shift_y': shifts[:, 1].cpu().numpy(),
            'centrality': centrality.cpu().numpy()}


def spread_across_scenes(burst_names: List[str], n: int) -> List[int]:
    """Pick n burst indices round-robin over scenes (fixed-seed random crop order within each scene)."""
    by_scene = defaultdict(list)
    for idx, name in enumerate(burst_names):
        by_scene[name.split('_')[0]].append(idx)
    rng = random.Random(0)
    queues = [rng.sample(by_scene[s], len(by_scene[s])) for s in sorted(by_scene)]
    picked: List[int] = []
    round_ = 0
    while len(picked) < min(n, len(burst_names)):
        for q in queues:
            if round_ < len(q) and len(picked) < n:
                picked.append(q[round_])
        round_ += 1
    return sorted(picked)


def select_bursts(burst_names: List[str], bursts_file: str = None, limit: int = None) -> List[int]:
    """Dataset indices to evaluate: optionally restricted to the names in a split file (see ./splits), then
    optionally thinned to `limit` bursts spread across scenes."""
    indices = list(range(len(burst_names)))
    if bursts_file is not None:
        with open(bursts_file) as f:
            wanted = {line.strip() for line in f if line.strip()}
        missing = wanted - set(burst_names)
        if missing:
            raise ValueError(f'{len(missing)} bursts in {bursts_file} are not in the dataset, e.g. {sorted(missing)[:3]}')
        indices = [i for i in indices if burst_names[i] in wanted]
    if limit is not None:
        picked = spread_across_scenes([burst_names[i] for i in indices], limit)
        indices = [indices[i] for i in picked]
    return indices


def bilinear_reference(burst: torch.Tensor) -> torch.Tensor:
    """No-network baseline: bilinear upsampling of the reference frame's R, G, B planes. burst: [1, N, 4, H, W]."""
    return F.interpolate(burst[:, 0, [0, 1, 3]], scale_factor=8, mode='bilinear', align_corners=False).clamp(0.0, 1.0)


def reorder(burst: torch.Tensor, ref: int) -> torch.Tensor:
    """Put frame `ref` first and keep the remaining frames in capture order."""
    perm = [ref] + [i for i in range(burst.shape[1]) if i != ref]
    return burst[:, perm]


def select_reference(burst: torch.Tensor, policy: str, margin: float = GUARD_MARGIN) -> int:
    """Index of the frame to use as reference, from the input burst alone. burst: [1, N, 4, H, W].
    Policies: frame0 (original behaviour), frame2 (best fixed frame on dev), guarded (see ref_selection.py)."""
    if policy == 'frame0':
        return 0
    if policy == 'frame2':
        return 2
    if policy == 'guarded':
        sharp = frame_sharpness_smoothed(burst).tolist()
        return guarded_choice(dict(enumerate(sharp)), margin=margin)
    raise ValueError(f'Unknown reference policy {policy}')


def summarise(rows: List[Dict], refs: List[int]) -> List[Dict]:
    by_burst = defaultdict(dict)
    for r in rows:
        by_burst[r['burst']][r['ref']] = r
    complete = [b for b, d in by_burst.items() if all(ref in d for ref in refs)]

    def common(row: Dict) -> float:
        return -math.inf if math.isnan(row['psnr_common']) else row['psnr_common']

    def means(sel: List[Dict]) -> Dict:
        return {k: float(np.nanmean([s[k] for s in sel])) for k in ('psnr', 'ssim', 'lpips', 'psnr_common')}

    summary = []
    for ref in refs:
        row = {'policy': f'fixed_ref_{ref}', **means([by_burst[b][ref] for b in complete])}
        row['times_best_common'] = sum(max(by_burst[b].values(), key=common)['ref'] == ref for b in complete)
        summary.append(row)

    # Selection policies. The oracles use the ground truth (upper bounds); the others only look at the input.
    policies = {
        'oracle_best_psnr': lambda d: max(d.values(), key=lambda x: x['psnr']),
        'oracle_best_common': lambda d: max(d.values(), key=common),
        'sharpest_frame': lambda d: max(d.values(), key=lambda x: x['sharpness']),
        'sharpest_smoothed': lambda d: max(d.values(), key=lambda x: x['sharpness_smooth']),
        'most_central': lambda d: min(d.values(), key=lambda x: x['centrality']),
    }
    for name, pick in policies.items():
        summary.append({'policy': name, **means([pick(by_burst[b]) for b in complete]), 'times_best_common': ''})
    for s in summary:
        s['n_bursts'] = len(complete)
    return summary


def verify_metrics(model: Base_Model, metrics: FastAlignedMetrics, alignment_net: torch.nn.Module,
                   args: argparse.Namespace) -> None:
    """Check FastAlignedMetrics reproduces the original AlignedPSNR/SSIM/LPIPS on a few bursts and refs."""
    from utils.metrics import AlignedPSNR, AlignedSSIM, AlignedLPIPS
    psnr_fn, ssim_fn = AlignedPSNR(alignment_net=alignment_net), AlignedSSIM(alignment_net=alignment_net)
    lpips_fn = AlignedLPIPS(alignment_net=alignment_net)

    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                             random_flip=False, ref_id=0, shuffle_frames=False)
    worst = np.zeros(3)
    for idx in range(args.verify_metrics):
        burst, labels, _, _, name = dataset[idx * (len(dataset) // args.verify_metrics)]
        burst, labels = burst[None].cuda(), labels[None].cuda()
        for ref in (0, 7):
            burst_ref = reorder(burst, ref)
            with torch.no_grad():
                output = model(burst_ref).clamp(0.0, 1.0)
            t0 = time.time()
            orig = np.array([float(psnr_fn(output, labels, burst_ref)), float(ssim_fn(output, labels, burst_ref)),
                             float(lpips_fn(output, labels, burst_ref))])
            t1 = time.time()
            fast = np.array(metrics(output, labels, burst_ref))
            t2 = time.time()
            worst = np.maximum(worst, np.abs(orig - fast))
            print(f'{name} ref={ref:2d}  original PSNR/SSIM/LPIPS = {orig.round(5)} ({t1 - t0:.1f}s)   '
                  f'fast = {fast.round(5)} ({t2 - t1:.1f}s)')
    print(f'Max abs difference  PSNR {worst[0]:.2e}  SSIM {worst[1]:.2e}  LPIPS {worst[2]:.2e}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Sweep the reference frame for real BurstSR evaluation')
    parser.add_argument('--weights', default='./Trained_models/Real/Burstormer.ckpt', type=str)
    parser.add_argument('--data_root', default='./burstsr_dataset', type=str)
    parser.add_argument('--pwcnet', default='./pwcnet/pwcnet-network-default.pth', type=str)
    parser.add_argument('--out_dir', default='./Results/Real/ref_frame_sweep', type=str)
    parser.add_argument('--refs', default=list(range(NUM_FRAMES)), type=int, nargs='+',
                        help='Reference frame indices to evaluate (default: all 14)')
    parser.add_argument('--limit', default=None, type=int,
                        help='Only evaluate N bursts, spread round-robin across the 23 scenes '
                             '(burst names are <scene>_<crop>, so the first N alphabetically would all be one scene)')
    parser.add_argument('--bursts_file', default=None, type=str,
                        help='Text file of burst names to evaluate, e.g. ./splits/dev.txt (default: whole val set)')
    parser.add_argument('--random_weights', action='store_true',
                        help='Skip loading weights (pipeline smoke test only; metrics are meaningless)')
    parser.add_argument('--verify_metrics', default=0, type=int,
                        help='Compare the fast metrics against the original utils.metrics classes on N bursts, then exit')
    args = parser.parse_args()

    set_seed(50)
    os.makedirs(args.out_dir, exist_ok=True)

    alignment_net = PWCNet(load_pretrained=True, weights_path=args.pwcnet).cuda()
    metrics = FastAlignedMetrics(alignment_net)

    model = Base_Model().eval().cuda() if args.random_weights else load_model(args.weights)

    if args.verify_metrics:
        verify_metrics(model, metrics, alignment_net, args)
        return

    # Load in capture order (frame 0 first, no shuffling); reference reordering happens on the GPU tensor
    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                             random_flip=False, ref_id=0, shuffle_frames=False)
    if args.limit is not None or args.bursts_file is not None:
        dataset = torch.utils.data.Subset(dataset, select_bursts(dataset.burst_list, args.bursts_file, args.limit))
    loader = DataLoader(dataset, batch_size=1, num_workers=2)

    per_burst_path = os.path.join(args.out_dir, 'per_burst.csv')
    feature_names = ['sharpness', 'sharpness_smooth', 'shift_x', 'shift_y', 'centrality']
    fields = ['burst', 'ref', 'psnr', 'ssim', 'lpips', 'psnr_common', 'ssim_common', 'lpips_common', 'common_frac',
              *feature_names, 'patch_psnr_common']
    rows: List[Dict] = []
    start = time.time()

    with open(per_burst_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (burst, labels, _, _, burst_name) in enumerate(loader):
            burst, labels = burst.cuda(), labels.cuda()
            features = frame_features(burst)

            # Pass 1: run the model with every reference and align each output to the ground truth
            aligned = {}
            for ref in args.refs:
                burst_ref = reorder(burst, ref)
                with torch.no_grad():
                    output = model(burst_ref).clamp(0.0, 1.0)
                # The metrics colour-match using burst_ref[:, 0], i.e. the frame the output is aligned to
                aligned[ref] = metrics.align_patches(output, labels, burst_ref)

            # The ground-truth crop is framed on frame 0, so other references cover less of it. For a like-for-like
            # comparison, also score every reference on the pixels that are valid and in view for all of them.
            n_patches = len(aligned[args.refs[0]])
            common = [torch.stack([aligned[r][p]['valid'] & aligned[r][p]['in_view'] for r in args.refs]).all(dim=0)
                      for p in range(n_patches)]
            common_frac = float(np.mean([c.float().mean().item() for c in common]))

            # Pass 2: original metrics, and the same metrics restricted to the common region
            for ref in args.refs:
                psnr, ssim, lpips_val = metrics.score(aligned[ref])
                patch_psnr, patch_ssim, patch_lpips = metrics.score(aligned[ref], regions=common, per_patch=True)
                any_patch = not all(math.isnan(x) for x in patch_psnr)
                row = {'burst': burst_name[0], 'ref': ref, 'psnr': psnr, 'ssim': ssim, 'lpips': lpips_val,
                       'psnr_common': float(np.nanmean(patch_psnr)) if any_patch else float('nan'),
                       'ssim_common': float(np.nanmean(patch_ssim)) if any_patch else float('nan'),
                       'lpips_common': float(np.nanmean(patch_lpips)) if any_patch else float('nan'),
                       'common_frac': common_frac,
                       **{name: float(features[name][ref]) for name in feature_names},
                       'patch_psnr_common': ';'.join(f'{x:.4f}' for x in patch_psnr)}
                rows.append(row)
                writer.writerow(row)
            del aligned
            f.flush()
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start
                print(f'[{i + 1}/{len(dataset)}] {elapsed / (i + 1):.2f}s/burst, '
                      f'ETA {elapsed / (i + 1) * (len(dataset) - i - 1) / 60:.1f} min', flush=True)

    summary = summarise(rows, args.refs)
    summary_path = os.path.join(args.out_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['policy', 'n_bursts', 'psnr', 'ssim', 'lpips', 'psnr_common',
                                               'times_best_common'])
        writer.writeheader()
        writer.writerows(summary)

    print(f'\nPSNR = original metric. PSNR_common = same pixels for every reference (fair comparison).')
    print(f'{"policy":<20}{"PSNR":>9}{"SSIM":>8}{"LPIPS":>8}{"PSNR_common":>13}{"#best":>7}')
    for s in summary:
        print(f'{s["policy"]:<20}{s["psnr"]:>9.3f}{s["ssim"]:>8.4f}{s["lpips"]:>8.4f}{s["psnr_common"]:>13.3f}'
              f'{str(s["times_best_common"]):>7}')
    print(f'\nWrote {per_burst_path} and {summary_path}')
    print('For confidence intervals and the split-half oracle check: python analyze_reference_policies.py '
          f'--sweep_dir {args.out_dir}')


if __name__ == '__main__':
    main()
