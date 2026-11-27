"""
Algorithm 2: Central, Exposure-Consistent Reference Selection.

Goal:
    Pick the frame of the burst that Burstormer should use as its reference
    (the frame it aligns everything else to). Uses the input burst only.

Step 1 - Centrality (how close is each frame to all the others?):
    Measure how far each frame is shifted from every other frame (phase
    correlation on the green channel). A frame in the middle of the burst's
    motion has the least to align, so a smaller average shift is better.

Step 2 - Brightness agreement (does the frame match the rest of the burst?):
    Compare each frame's brightness with the burst's median brightness.
    A reference brighter or darker than the others hurts Burstormer, so a
    smaller difference is better (exposure normalisation, Buades et al.).

Step 3 - Health check (is the frame damaged?):
    Align each frame to its 4 nearest frames in time and measure how much
    it differs from them. Noise, blur, bad exposure and clipping all make a
    frame disagree with its neighbours. A frame more than 30% worse than the
    burst's typical frame is never chosen.

Step 4 - Choose:
    score = centrality + 0.5 x brightness difference (each standardised
    over the burst). The healthy frame with the lowest score is returned.

Why not just frame 0:
    Once the metric's penalty for a moved reference is removed (shift
    control), frame 0 is on average the worst reference: it sits at the edge
    of the burst. Middle frames are the best. On 40 test bursts this rule
    gains +0.13 dB PSNR over frame 0, and improves SSIM and LPIPS too.

IMPORTANT:
    Ground truth is never used to choose the frame. Image-quality scores
    (sharpness, algorithm 1's Q score, noise) did not predict the best
    frame, so they are not part of the score. Results: docs/ALGORITHM2.md.

The first version (keep frame 0 unless it looks damaged) is kept at the
bottom of this file as select_reference_frame_v1, for comparison.

References:
    A. Buades, O. Martorell, M. Sanchez-Beeckman, "Joint denoising and HDR
    for RAW video sequences", arXiv:2201.07066, 2022.
    S. W. Hasinoff et al., "Burst photography for high dynamic range and
    low-light imaging on mobile cameras", SIGGRAPH Asia 2016.
"""

import math
from typing import Dict

import numpy as np
import torch

# Current rule
BRIGHTNESS_WEIGHT = 0.5  # chosen on 40 dev bursts with shift-control labels (0.25 and 0.5 tied, best of 0-2)
BRIGHTNESS_FLOOR = 0.01  # brightness spreads below 1% are noise: never standardise them up to the motion's scale
HEALTH_MARGIN = 0.3      # never pick a frame whose consistency cost is >30% above the burst median
NEAREST = 4              # the health gate compares each frame with this many temporal neighbours

# First version (select_reference_frame_v1)
DEFAULT_REF = 0
WINDOW = 5
NEIGHBOURS = 8  # each candidate is compared with this many frames that follow it
MARGIN = 0.2    # chosen on clean dev bursts (score_ref_policies.py)

_EPS = 1e-6


def _as_frames(burst) -> torch.Tensor:
    """Accept a torch tensor or numpy array of shape (N, 4, H, W), or (1, N, 4, H, W) as in the evaluation scripts,
    packed RGGB in [0, 1]. Returns a float tensor (N, 4, H, W); the input is never modified. Raises ValueError on a
    wrong shape, an empty burst or non-finite values, like algorithm 1."""
    if not isinstance(burst, torch.Tensor):
        burst = torch.from_numpy(np.array(burst, dtype=np.float32))  # copy, so the caller's array is untouched
    if burst.dim() == 5:
        if burst.shape[0] != 1:
            raise ValueError(f'Expected a single burst, got batch size {burst.shape[0]}')
        burst = burst[0]
    if burst.dim() != 4 or burst.shape[0] == 0 or burst.shape[1] != 4:
        raise ValueError(f'Expected burst shape (N, 4, H, W), got {tuple(burst.shape)}.')
    burst = burst.float()
    if not torch.isfinite(burst).all():
        raise ValueError('Burst contains non-finite values.')
    return burst


def _phase_shifts(green: torch.Tensor, ref: int) -> torch.Tensor:
    """Sub-pixel translation (dy, dx) that moves every frame onto frame `ref`. green: [N, H, W]. Returns [N, 2]."""
    n, h, w = green.shape
    g = green - green.mean(dim=(1, 2), keepdim=True)
    window = torch.hann_window(h, periodic=False, device=g.device)[:, None] * \
        torch.hann_window(w, periodic=False, device=g.device)[None, :]
    spec = torch.fft.fft2(g * window)
    # If frame j is frame `ref` displaced by d, this peaks at -d: the translation that moves j back onto ref
    # (checked in tests/test_algorithm2.py)
    cross = spec[ref:ref + 1] * spec.conj()
    corr = torch.fft.ifft2(cross / (cross.abs() + 1e-12)).real
    peak = corr.flatten(1).argmax(dim=1)
    py, px = peak // w, peak % w
    idx = torch.arange(n, device=g.device)

    def refine(c_minus: torch.Tensor, c_0: torch.Tensor, c_plus: torch.Tensor) -> torch.Tensor:
        denom = c_minus - 2 * c_0 + c_plus
        return torch.where(denom.abs() > 1e-12, 0.5 * (c_minus - c_plus) / denom,
                           torch.zeros_like(denom)).clamp(-0.5, 0.5)

    c0 = corr[idx, py, px]
    dy = py.float() + refine(corr[idx, (py - 1) % h, px], c0, corr[idx, (py + 1) % h, px])
    dx = px.float() + refine(corr[idx, py, (px - 1) % w], c0, corr[idx, py, (px + 1) % w])
    dy = torch.where(dy > h / 2, dy - h, dy)
    dx = torch.where(dx > w / 2, dx - w, dx)
    return torch.stack([dy, dx], dim=1)


def _fourier_shift(images: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Translate each image [N, H, W] by (dy, dx) [N, 2] with a Fourier phase ramp: exact sub-pixel translation
    without the smoothing of bilinear interpolation (which would make warped frames look less noisy). Circular, so
    the caller must ignore a border as wide as the shift."""
    n, h, w = images.shape
    fy = torch.fft.fftfreq(h, device=images.device)[None, :, None]
    fx = torch.fft.fftfreq(w, device=images.device)[None, None, :]
    ramp = torch.exp(-2j * math.pi * (fy * shifts[:, 0, None, None] + fx * shifts[:, 1, None, None]))
    return torch.fft.ifft2(torch.fft.fft2(images) * ramp).real


def _aligned_residuals(green: torch.Tensor, c: int, shifts: torch.Tensor, others: list) -> torch.Tensor:
    """mean|c - j| for each j in `others`, after aligning them to c, on the shared pixels, divided by the typical
    brightness of the frames j there (one scale for c and all of them, so a brightness difference stays in the
    residual)."""
    _, h, w = green.shape
    aligned = _fourier_shift(green, shifts)
    # Shared pixels: drop the wrapped-around border, keeping at least the central half of the image
    border = min(int(math.ceil(shifts[others + [c]].abs().max().item())) + 1, min(h, w) // 4)
    a = aligned[:, border:h - border, border:w - border]
    typical = a[others].mean(dim=(1, 2)).clamp(min=_EPS).median()
    return (a[others] - a[c:c + 1]).abs().flatten(1).mean(dim=1) / typical


def _standardise(x: np.ndarray, floor: float = 0.0) -> np.ndarray:
    return (x - x.mean()) / max(x.std(), floor, 1e-9)


def frame_features(burst, nearest: int = NEAREST) -> Dict[str, np.ndarray]:
    """Per-frame features of a burst (N, 4, H, W), each an array of N values:
        centrality   mean translation (px) from the frame to every other frame
        brightness   |log(brightness / burst median brightness)|
        consistency  median aligned residual against the `nearest` temporal neighbours, exposure-normalised
    """
    frames = _as_frames(burst)
    n = frames.shape[0]
    if n < 2:
        raise ValueError('Needs at least two frames')
    green = frames[:, 1:3].mean(dim=1)  # [N, H, W], best-sampled colour
    level = green.mean(dim=(1, 2)).clamp(min=_EPS)
    centrality, consistency = np.zeros(n), np.zeros(n)
    for c in range(n):
        # Step 1: average shift from frame c to every other frame
        shifts = _phase_shifts(green, c)
        centrality[c] = shifts.norm(dim=1).sum().item() / (n - 1)
        # Step 3: how much frame c disagrees with its nearest frames in time, once aligned
        near = sorted((j for j in range(n) if j != c), key=lambda j: (abs(j - c), j))[:min(nearest, n - 1)]
        consistency[c] = _aligned_residuals(green, c, shifts, near).median().item()
    # Step 2: how far each frame's brightness is from the burst's median brightness
    brightness = (level / level.median()).log().abs().numpy().astype(np.float64)
    return {'centrality': centrality, 'brightness': brightness, 'consistency': consistency}


def choose_from_features(features: Dict[str, np.ndarray], brightness_weight: float = BRIGHTNESS_WEIGHT,
                         margin: float = HEALTH_MARGIN) -> int:
    """The selection rule on precomputed features: the lowest score among the frames that pass the health gate."""
    # Step 4: lower score = more central and closer to the burst's brightness
    score = (_standardise(features['centrality'])
             + brightness_weight * _standardise(features['brightness'], floor=BRIGHTNESS_FLOOR))
    # Step 3: drop damaged frames (never empty: at least half the frames are at or below the median)
    cost = features['consistency']
    healthy = cost <= (1.0 + margin) * np.median(cost)
    # The healthy frame with the lowest score is the reference
    return int(np.where(healthy, score, np.inf).argmin())


@torch.no_grad()
def select_reference_frame(burst, brightness_weight: float = BRIGHTNESS_WEIGHT, margin: float = HEALTH_MARGIN) -> int:
    """Select the reference frame of an input burst of shape (N, 4, H, W) (torch tensor or numpy array, packed RGGB
    in [0, 1]); (1, N, 4, H, W) is also accepted. Same interface as algorithm1.select_reference_frame: returns the
    frame index as a Python int in [0, N-1]. The keyword arguments are for the evaluation sweeps only."""
    frames = _as_frames(burst)
    if frames.shape[0] == 1:
        return 0
    return choose_from_features(frame_features(frames), brightness_weight=brightness_weight, margin=margin)


# ---------------------------------------------------------------------------------------------------------------
# First version, kept for comparison (score_ref_policies.py, docs/ALGORITHM2.md): keep frame 0 unless its cost
#
#     cost(c) = median over the NEIGHBOURS frames j after c of mean|c - j|  /  typical brightness of the frames j
#
# is more than MARGIN above the median cost of frames 1..WINDOW-1; then take the earliest frame within MARGIN.
# ---------------------------------------------------------------------------------------------------------------

def _comparison_frames(c: int, n: int, candidates: list, neighbours: int) -> list:
    """Frames that candidate c is compared with: the `neighbours` frames after it, fewer if the burst is short, so
    that every candidate sees the same time gaps. A candidate with no later frame falls back to all other frames."""
    k = max(1, min(neighbours, n - 1 - max(candidates)))
    later = [j for j in range(c + 1, c + 1 + k) if j < n]
    return later or [j for j in range(n) if j != c]


def frame_costs(burst, window: int = WINDOW, neighbours: int = NEIGHBOURS) -> Dict[int, float]:
    """First version: cost of each of the first `window` frames. Lower is better."""
    frames = _as_frames(burst)
    n = frames.shape[0]
    if n < 2:
        raise ValueError('Needs at least two frames')
    candidates = list(range(min(window, n)))
    green = frames[:, 1:3].mean(dim=1)
    costs = {}
    for c in candidates:
        others = _comparison_frames(c, n, candidates, neighbours)
        costs[c] = _aligned_residuals(green, c, _phase_shifts(green, c), others).median().item()
    return costs


def choose_from_costs(costs: Dict[int, float], default: int = DEFAULT_REF, margin: float = MARGIN) -> int:
    """First version's guarded rule on precomputed costs: keep `default` unless its cost is more than `margin` above
    the median cost of the other candidates; then take the earliest candidate within `margin` of that median."""
    if default not in costs:
        raise ValueError(f'Default frame {default} is not among the candidates {sorted(costs)}')
    others = sorted(c for c in costs if c != default)
    if not others:
        return default
    values = sorted(costs[c] for c in others)
    mid = len(values) // 2
    typical = values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])
    limit = (1.0 + margin) * typical
    if costs[default] <= limit:
        return default
    # Never empty: at least half of the other candidates are at or below their median
    return next(c for c in others if costs[c] <= limit)


@torch.no_grad()
def select_reference_frame_v1(burst, default: int = DEFAULT_REF, window: int = WINDOW, margin: float = MARGIN) -> int:
    """First version of algorithm 2 (keep frame 0 unless it looks damaged). Same interface as
    select_reference_frame."""
    frames = _as_frames(burst)
    if frames.shape[0] == 1:
        return 0
    return int(choose_from_costs(frame_costs(frames, window), default=default, margin=margin))
