"""Geometry between two super-resolved outputs, and a low-loss warp that applies it.

Used by shift_control.py and shift_control_benchmark.py. A geometry moves image a onto image b:
warp_image(a, g) is approximately b. It is stored as a sampling map split into two parts:

    warp_image(a, g)(p) = a(p - g.shift + g.residual(p))

g.shift is a global translation, applied with the exact Fourier shift of shift_image (no blur, no ringing).
g.residual is a small per-pixel offset field (rotation, scale, perspective, parallax) or None. It is applied by
sampling an 8x Fourier-upsampled copy of the image with bicubic interpolation, which keeps the detail and the
noise level of the image (plain bilinear warping would lose detail; 4x upsampling still raises white noise by
4%). The warp does not clip to [0, 1]. Pixel coordinates are (row, column), in output pixels.

Models (estimate_geometry):
    translation  one global shift, from phase correlation (the original shift control)
    affine       6 parameters, fitted to tile-wise phase correlation
    homography   8 parameters, fitted to tile-wise phase correlation
    dense        PWC-Net optical flow between the two outputs, after the global shift, smoothed
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

PAD = 64  # minimum mirror padding (output pixels) around a shifted image; grown for larger shifts
MODELS = ('translation', 'affine', 'homography', 'dense')


# --------------------------------------------------------------------------------------------------------------
# Translation: phase correlation and the exact Fourier shift
# --------------------------------------------------------------------------------------------------------------

def phase_correlation(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Sub-pixel (dy, dx) that moves image a onto image b, and the height of the correlation peak (0..1, a
    confidence). Green channels, Hann window. a, b: [1, 3, H, W]."""
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
    return torch.tensor([dy, dx]), float(corr[py, px].item())


def translation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sub-pixel (dy, dx) that moves image a onto image b, by phase correlation of the green channels."""
    return phase_correlation(a, b)[0]


def shift_image(img: torch.Tensor, dy: float, dx: float, clamp: bool = True) -> torch.Tensor:
    """Move img [1, C, H, W] by (dy, dx) output pixels: mirror-pad, exact Fourier shift, crop back.

    clamp=True clips the result to [0, 1], as the original control did. That clipping is not neutral in dark,
    noisy areas (it cuts the slight undershoot of an exact shift), so warp_image does not clip."""
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
    out = shifted[..., pad:pad + h, pad:pad + w]
    return out.clamp(0.0, 1.0) if clamp else out


# --------------------------------------------------------------------------------------------------------------
# General geometry and the low-loss warp
# --------------------------------------------------------------------------------------------------------------

@dataclass
class Geometry:
    """Moves an image a onto b: warp_image(a, g)(p) = a(p - shift + residual(p))."""
    shift: torch.Tensor  # [2] (dy, dx), global translation
    residual: Optional[torch.Tensor] = None  # [2, H, W] (dy, dx) per output pixel, or None
    model: str = 'translation'
    info: Dict = field(default_factory=dict)

    def displacement(self, h: int, w: int) -> torch.Tensor:
        """Total displacement of the content at every pixel, [2, H, W] (dy, dx): shift - residual."""
        d = self.shift.float().view(2, 1, 1).expand(2, h, w)
        return d if self.residual is None else d - self.residual.to(d.device)

    def max_residual(self) -> float:
        return 0.0 if self.residual is None else float(self.residual.abs().max())


def upsample(img: torch.Tensor, factor: int) -> torch.Tensor:
    """Band-limited upsampling by an integer factor: sample i * factor + j of the result is img at i + j / factor,
    made with the exact Fourier shift. [1, C, H, W] -> [1, C, H * factor, W * factor]."""
    if factor == 1:
        return img
    _, c, h, w = img.shape
    out = img.new_empty(1, c, h, factor, w, factor)
    for jy in range(factor):
        for jx in range(factor):
            phase = img if jy == jx == 0 else shift_image(img, -jy / factor, -jx / factor, clamp=False)
            out[:, :, :, jy, :, jx] = phase
    return out.view(1, c, h * factor, w * factor)


def sample(img: torch.Tensor, coords: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """img sampled at fractional (row, col) coordinates [2, H, W]: bicubic on a band-limited upsampled copy,
    mirrored at the edges. Sampling at integer coordinates returns img exactly (up to float error)."""
    up = upsample(img, factor)
    hu, wu = up.shape[-2:]
    # Coordinate c in img is index c * factor in the upsampled image (align_corners=True convention)
    gy = 2.0 * coords[0] * factor / (hu - 1) - 1.0
    gx = 2.0 * coords[1] * factor / (wu - 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)[None].to(img.dtype)
    return F.grid_sample(up, grid, mode='bicubic', padding_mode='reflection', align_corners=True)


def pixel_grid(h: int, w: int, device) -> torch.Tensor:
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    return torch.stack(torch.meshgrid(ys, xs, indexing='ij'))  # [2, H, W]


def warp_image(img: torch.Tensor, g: Geometry, factor: int = 8) -> torch.Tensor:
    """Apply geometry g to img [1, C, H, W]: exact Fourier shift, then low-loss sampling of the residual."""
    moved = shift_image(img, *g.shift.tolist(), clamp=False)
    if g.residual is None:
        return moved
    h, w = img.shape[-2:]
    return sample(moved, pixel_grid(h, w, img.device) + g.residual.to(img.device), factor)


def invert(g: Geometry, iters: int = 20) -> Geometry:
    """Geometry that undoes g: warp_image(warp_image(a, g), invert(g)) is approximately a.

    warp_image(a, g) samples a at s(p) = p - shift + r(p). The inverse samples at t(p) with s(t(p)) = p, found by
    the fixed point t = p + shift - r(t). Assumes a smooth residual field, as all models here produce.
    """
    if g.residual is None:
        return Geometry(-g.shift, None, g.model, dict(g.info))
    r = g.residual
    _, h, w = r.shape
    p = pixel_grid(h, w, r.device)
    t = p + g.shift.to(r.device).view(2, 1, 1)
    for _ in range(iters):
        gy, gx = 2.0 * t[0] / (h - 1) - 1.0, 2.0 * t[1] / (w - 1) - 1.0
        r_t = F.grid_sample(r[None], torch.stack((gx, gy), dim=-1)[None], mode='bilinear',
                            padding_mode='border', align_corners=True)[0]
        t = p + g.shift.to(r.device).view(2, 1, 1) - r_t
    shift = -g.shift
    return Geometry(shift, t - (p + g.shift.to(r.device).view(2, 1, 1)), g.model, dict(g.info))


# --------------------------------------------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------------------------------------------

def _fit_projective(src: torch.Tensor, dst: torch.Tensor, weights: torch.Tensor, model: str) -> torch.Tensor:
    """Weighted least-squares map dst = M(src) for points [N, 2] (row, col). Returns a 3x3 matrix acting on
    homogeneous (col, row, 1). Coordinates are normalised for conditioning."""
    c = src.mean(0)
    s = (src - c).abs().mean() + 1e-9
    xs, ys = (src[:, 1] - c[1]) / s, (src[:, 0] - c[0]) / s
    xd, yd = (dst[:, 1] - c[1]) / s, (dst[:, 0] - c[0]) / s
    one, zero = torch.ones_like(xs), torch.zeros_like(xs)
    if model == 'affine':
        rows_x = torch.stack([xs, ys, one, zero, zero, zero], 1)
        rows_y = torch.stack([zero, zero, zero, xs, ys, one], 1)
    else:  # homography with h33 = 1: xd (h31 xs + h32 ys + 1) = h11 xs + h12 ys + h13, same for yd
        rows_x = torch.stack([xs, ys, one, zero, zero, zero, -xd * xs, -xd * ys], 1)
        rows_y = torch.stack([zero, zero, zero, xs, ys, one, -yd * xs, -yd * ys], 1)
    a = torch.cat([rows_x, rows_y]).double()
    b = torch.cat([xd, yd]).double()
    sw = torch.cat([weights, weights]).double().sqrt()
    h = torch.linalg.lstsq(a * sw[:, None], (b * sw)[:, None]).solution[:, 0]
    m = torch.eye(3, dtype=torch.float64)
    if model == 'affine':
        m[0], m[1] = h[0:3], h[3:6]
    else:
        m[0], m[1], m[2, 0], m[2, 1] = h[0:3], h[3:6], h[6], h[7]
    # Undo the normalisation: M = T^-1 Mn T with T(x) = (x - c) / s
    t = torch.tensor([[1 / s, 0, -c[1] / s], [0, 1 / s, -c[0] / s], [0, 0, 1]], dtype=torch.float64)
    return torch.linalg.inv(t) @ m @ t


def _apply_projective(m: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Map points [..., 2] (row, col) with a 3x3 matrix on homogeneous (col, row, 1)."""
    x, y = pts[..., 1].double(), pts[..., 0].double()
    den = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    xo = (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / den
    yo = (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / den
    return torch.stack([yo, xo], -1).float()


def tile_shifts(a: torch.Tensor, b: torch.Tensor, tile: int = 128, grid: int = 5):
    """Local (dy, dx) moving a onto b, measured by phase correlation on a grid of tiles.
    Returns tile centres [N, 2] (row, col), shifts [N, 2] and peak heights [N]."""
    h, w = a.shape[-2:]
    cy = torch.linspace(tile / 2, h - tile / 2, grid)
    cx = torch.linspace(tile / 2, w - tile / 2, grid)
    centres, shifts, peaks = [], [], []
    for y in cy.tolist():
        for x in cx.tolist():
            y0, x0 = int(round(y - tile / 2)), int(round(x - tile / 2))
            d, pk = phase_correlation(a[..., y0:y0 + tile, x0:x0 + tile], b[..., y0:y0 + tile, x0:x0 + tile])
            centres.append([y0 + (tile - 1) / 2, x0 + (tile - 1) / 2])
            shifts.append(d.tolist())
            peaks.append(pk)
    return torch.tensor(centres), torch.tensor(shifts), torch.tensor(peaks)


def _parametric(a: torch.Tensor, b: torch.Tensor, model: str, tile: int, grid: int) -> Geometry:
    shift = translation(a, b)
    a1 = shift_image(a, *shift.tolist())
    centres, local, peaks = tile_shifts(a1, b, tile, grid)
    # b at centre c shows a1 at c - local, i.e. a at c - local - shift: the sampling map of warp_image
    dst = centres - local - shift[None]
    w = peaks.clamp(min=1e-3)
    m = None
    for _ in range(5):  # iteratively reweighted least squares with Huber weights (scale 0.25 px)
        m = _fit_projective(centres, dst, w, model)
        err = (_apply_projective(m, centres) - dst).norm(dim=1)
        w = peaks.clamp(min=1e-3) * torch.where(err < 0.25, torch.ones_like(err), 0.25 / err)
    h, wd = a.shape[-2:]
    p = pixel_grid(h, wd, 'cpu')
    src = _apply_projective(m, p.permute(1, 2, 0)).permute(2, 0, 1)  # [2, H, W] sampling map
    residual = src - (p - shift.view(2, 1, 1))
    fit_err = (_apply_projective(m, centres) - dst).norm(dim=1)
    return Geometry(shift, residual.to(a.device), model,
                    {'fit_rms_px': float(fit_err.pow(2).mean().sqrt()), 'tiles': int(len(peaks)),
                     'peak_median': float(peaks.median())})


def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, [r, r, r, r], mode='replicate')
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def _dense(a: torch.Tensor, b: torch.Tensor, flow_net, smooth: float) -> Geometry:
    shift = translation(a, b)
    a1 = shift_image(a, *shift.tolist())
    with torch.no_grad():
        flow = flow_net(a1 / (a1.max() + 1e-6), b / (b.max() + 1e-6))  # [1, 2, H, W] (x, y): b(p) = a1(p + flow)
    flow = _gauss_blur(flow, smooth)[0]
    residual = torch.stack([flow[1], flow[0]])  # (dy, dx)
    return Geometry(shift, residual, 'dense', {'smooth': smooth})


def sampling_map(g: Geometry, h: int, w: int, device) -> torch.Tensor:
    """Where warp_image(a, g) samples a for every output pixel: [2, H, W] (row, col)."""
    p = pixel_grid(h, w, device)
    m = p - g.shift.to(device).view(2, 1, 1)
    return m if g.residual is None else m + g.residual.to(device)


def consistent_dense(a: torch.Tensor, b: torch.Tensor, flow_net, smooth: float = 16.0, tol: float = 1.0):
    """Dense geometry from a to b, trusted only where the flow is consistent both ways.

    The flow from a to b and the flow from b to a must compose to the identity within `tol` output pixels. Where
    they don't (flow failures, occlusions), the homography fit is used instead. The combined map is then smoothed.
    Returns the geometry and the fraction of pixels where the flow was replaced. Used to build realistic true
    geometries for the benchmark.
    """
    h, w = a.shape[-2:]
    s_ab = sampling_map(_dense(a, b, flow_net, 0.0), h, w, a.device)  # b pixel -> location in a
    s_ba = sampling_map(_dense(b, a, flow_net, 0.0), h, w, a.device)  # a pixel -> location in b
    gy, gx = 2.0 * s_ab[0] / (h - 1) - 1.0, 2.0 * s_ab[1] / (w - 1) - 1.0
    back = F.grid_sample(s_ba[None], torch.stack((gx, gy), -1)[None], mode='bilinear', padding_mode='border',
                         align_corners=True)[0]
    bad = (back - pixel_grid(h, w, a.device)).norm(dim=0) > tol
    hom = _parametric(a, b, 'homography', 128, 5)
    p = pixel_grid(h, w, a.device)
    s = torch.where(bad[None], sampling_map(hom, h, w, a.device), s_ab)
    s = p + _gauss_blur((s - p)[None], smooth)[0]  # smooth the offsets, not the coordinates (edges stay unbiased)
    shift = hom.shift.to(a.device)
    residual = s - (p - shift.view(2, 1, 1))
    return Geometry(shift, residual, 'dense', {'smooth': smooth, 'replaced_frac': float(bad.float().mean())}), \
        float(bad.float().mean())


def estimate_geometry(a: torch.Tensor, b: torch.Tensor, model: str = 'translation', flow_net=None,
                      tile: int = 128, grid: int = 5, smooth: float = 4.0) -> Geometry:
    """Geometry that moves output a onto output b (both [1, 3, H, W])."""
    if model == 'translation':
        return Geometry(translation(a, b), None, 'translation')
    if model in ('affine', 'homography'):
        return _parametric(a, b, model, tile, grid)
    if model == 'dense':
        if flow_net is None:
            raise ValueError('The dense model needs a flow network (PWC-Net)')
        return _dense(a, b, flow_net, smooth)
    raise ValueError(f'Unknown geometry model {model!r}; choose from {MODELS}')
