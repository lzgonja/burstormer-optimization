"""Algorithm 2: exposure-aware consistency selection of the reference frame.

Shared interface (ALGORITHMS.md): a burst of N frames in, the index of the reference frame out, using the burst alone.

Idea
    Burstormer reconstructs the scene as seen by its reference frame, so a damaged reference (wrongly exposed, noisy,
    blurred) costs far more than any other damaged frame. Every frame of a burst sees the same scene, so a good
    reference is one that the rest of the burst agrees with, in detail and in brightness. Each candidate frame c is
    compared with the frames that follow it (j = c+1 ... c+8), each aligned to c by a global translation (phase
    correlation, sub-pixel Fourier shift), on the pixels they share:

        cost(c) = median over j of mean|c - j|  /  typical brightness of the frames j

    - A noisy, blurred, clipped or misaligned frame disagrees with the others in detail. Unlike a sharpness score,
      noise can only raise the cost.
    - A frame brighter or darker than the rest of the burst disagrees in level: a frame 2x too bright or too dark
      adds about 1 or 0.5 to its cost, while healthy frames of clean val bursts cost 0.06 (median; at most 0.18).
      Burstormer is hurt by such a reference even without extra noise or clipping: on 12 dev bursts a reference
      twice as bright as the others cost 4 dB. The natural brightness spread within a BurstSR burst is about 4%.
    - Dividing by the typical brightness of the burst is the exposure normalisation of Buades et al. (2022,
      Sec. 3.2), done without exposure times. It makes the cost independent of the scene's overall level. The
      brightness is measured on the shared pixels, so content moving into or out of the frame does not count.

    Three choices were tested and dropped (docs/ALGORITHM2.md):
    - Normalising each frame by its own brightness, plus a separate brightness-mismatch term,
      cost = residual * (1 + |median log(brightness c / brightness j)|): it picked the same frame as this single
      cost in all 440 clean and damaged val cases tested, with two terms instead of one.
    - Comparing with the whole burst instead of the following frames favoured frames near the middle of the burst,
      because frames further apart in time differ more after alignment.
    - The well-exposedness weight of Buades et al. (2022, Sec. 4), w(x) = 1 - (2x - 1)^12, rewards brighter frames
      in BurstSR's dark scenes. It therefore preferred an over-exposed reference that the model handles badly.

Selection rule (guarded)
    - Only the first `window` frames are candidates: on BurstSR the best reference is always one of the first five,
      and quality falls with frame index (docs/PLAN.md, Track A).
    - The default frame is kept unless its cost is more than `margin` above the median cost of the other candidates.
      On clean bursts the early frames are about equally good, so switching would only add risk.
    - When the default is rejected, the earliest candidate whose cost is within `margin` of that median is used,
      not the cheapest one: among acceptable frames, earlier ones are better references.

Reference: A. Buades, O. Martorell, M. Sanchez-Beeckman, "Joint denoising and HDR for RAW video sequences",
arXiv:2201.07066, 2022.
"""

import math
from typing import Dict

import numpy as np
import torch

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


def _comparison_frames(c: int, n: int, candidates: list, neighbours: int) -> list:
    """Frames that candidate c is compared with: the `neighbours` frames after it, fewer if the burst is short, so
    that every candidate sees the same time gaps. A candidate with no later frame falls back to all other frames."""
    k = max(1, min(neighbours, n - 1 - max(candidates)))
    later = [j for j in range(c + 1, c + 1 + k) if j < n]
    return later or [j for j in range(n) if j != c]


def frame_costs(burst, window: int = WINDOW, neighbours: int = NEIGHBOURS) -> Dict[int, float]:
    """Cost of each of the first `window` frames as defined in the module docstring. Lower is better."""
    frames = _as_frames(burst)
    n, _, h, w = frames.shape
    if n < 2:
        raise ValueError('Needs at least two frames')
    candidates = list(range(min(window, n)))
    green = frames[:, 1:3].mean(dim=1)  # [N, H, W], best-sampled colour

    costs = {}
    for c in candidates:
        others = _comparison_frames(c, n, candidates, neighbours)
        shifts = _phase_shifts(green, c)
        aligned = _fourier_shift(green, shifts)
        # Shared pixels: drop the wrapped-around border, keeping at least the central half of the image
        border = min(int(math.ceil(shifts[others + [c]].abs().max().item())) + 1, min(h, w) // 4)
        a = aligned[:, border:h - border, border:w - border]
        # Typical brightness of the comparison frames on the shared pixels: one scale for c and all of them, so a
        # brightness difference stays in the residual
        typical = a[others].mean(dim=(1, 2)).clamp(min=_EPS).median()
        costs[c] = ((a[others] - a[c:c + 1]).abs().flatten(1).mean(dim=1).median() / typical).item()
    return costs


def choose_from_costs(costs: Dict[int, float], default: int = DEFAULT_REF, margin: float = MARGIN) -> int:
    """The guarded rule on precomputed costs: keep `default` unless its cost is more than `margin` above the median
    cost of the other candidates; then take the earliest candidate within `margin` of that median."""
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
def select_reference_frame(burst, default: int = DEFAULT_REF, window: int = WINDOW, margin: float = MARGIN) -> int:
    """Select the reference frame of an input burst of shape (N, 4, H, W) (torch tensor or numpy array, packed RGGB
    in [0, 1]); (1, N, 4, H, W) is also accepted. Same interface as algorithm1.select_reference_frame: returns the
    frame index as a Python int in [0, N-1]. The keyword arguments are for the evaluation sweeps only."""
    frames = _as_frames(burst)
    if frames.shape[0] == 1:
        return 0
    return int(choose_from_costs(frame_costs(frames, window), default=default, margin=margin))
