"""CPU-only Shared Coordinate Evaluation (SCE) for precomputed BurstSR outputs.

The module deliberately has no dependency on Burstormer, CUDA, PWC-Net, or the
reference-selection code.  Images use NumPy's ``(H, W[, C])`` convention and
translations are always returned as ``(dx, dy)`` in output-image pixels.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage, optimize


def _gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError("images must have shape (H,W) or (H,W,C)")
    if image.shape[2] == 1:
        return image[..., 0]
    if image.shape[2] < 3:
        return image.mean(axis=2)
    return image[..., :3] @ np.array([0.299, 0.587, 0.114])


def estimate_translation(reference: np.ndarray, moving: np.ndarray,
                         subpixel: bool = True) -> Tuple[float, float]:
    """Return ``(dx, dy)`` to apply to *moving* to align it to *reference*.

    Phase correlation supplies the integer peak.  A continuous evaluation of
    the normalized cross-power spectrum then refines that peak without
    resampling either input image.  A Hann window reduces wrap-around edges.
    """
    ref, mov = _gray(reference), _gray(moving)
    if ref.shape != mov.shape:
        raise ValueError("reference and moving images must have the same shape")
    h, w = ref.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    ref = (ref - ref.mean()) * window
    mov = (mov - mov.mean()) * window
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    magnitude = np.abs(cross)
    cross /= np.maximum(magnitude, np.finfo(np.float64).eps)
    corr = np.fft.ifft2(cross).real
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    dy = float(py if py <= h // 2 else py - h)
    dx = float(px if px <= w // 2 else px - w)
    if not subpixel:
        return dx, dy

    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)

    def negative_correlation(shift: np.ndarray) -> float:
        sx, sy = shift
        phase_y = np.exp(2j * np.pi * fy * sy)
        phase_x = np.exp(2j * np.pi * fx * sx)
        return -float(np.real(phase_y @ cross @ phase_x) / (h * w))

    result = optimize.minimize(
        negative_correlation, np.array([dx, dy]), method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 100},
    )
    refined = result.x if result.success else np.array([dx, dy])
    refined = np.clip(refined, [dx - 1.0, dy - 1.0], [dx + 1.0, dy + 1.0])
    return float(refined[0]), float(refined[1])


def estimate_reconstruction_positions(images: Sequence[np.ndarray]) -> np.ndarray:
    """Estimate globally consistent ``(x,y)`` positions from every image pair.

    Position is defined as the displacement of image content.  The arbitrary
    translation gauge is removed by requiring the mean position to be zero.
    """
    n = len(images)
    if n < 2:
        raise ValueError("at least two reconstructions are required")
    rows, values = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = estimate_translation(images[i], images[j])
            row = np.zeros(n)
            row[i], row[j] = -1.0, 1.0
            rows.append(row)
            values.append((-dx, -dy))
    rows.append(np.ones(n))
    values.append((0.0, 0.0))
    return np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)[0]


def shared_coordinate_origin(positions: np.ndarray) -> np.ndarray:
    """Geometric centre (centroid) of reconstruction positions."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N,2)")
    return positions.mean(axis=0)


def estimate_ground_truth_position(reconstructions: Sequence[np.ndarray],
                                   positions: np.ndarray,
                                   ground_truth: np.ndarray) -> np.ndarray:
    """Robustly locate ground truth in the reconstruction position system."""
    candidates = []
    for image, position in zip(reconstructions, positions):
        shift = np.asarray(estimate_translation(image, ground_truth))
        candidates.append(position - shift)
    return np.median(np.asarray(candidates), axis=0)


def align_image_and_mask(image: np.ndarray, shift_xy: Sequence[float],
                         order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Translate an image once and return a conservative genuine-content mask."""
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (H,W) or (H,W,C)")
    dx, dy = map(float, shift_xy)
    shift = (dy, dx) if image.ndim == 2 else (dy, dx, 0.0)
    aligned = ndimage.shift(image, shift, order=order, mode="constant", cval=0.0,
                            prefilter=(order > 1))
    h, w = image.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float64)
    source_x, source_y = xx - dx, yy - dy
    radius = 0 if order == 0 else (1 if order == 1 else 2)
    valid = ((source_x >= radius) & (source_x <= w - 1 - radius) &
             (source_y >= radius) & (source_y <= h - 1 - radius))
    return aligned, valid


def common_valid_region(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Return the pixelwise intersection of all validity masks."""
    if not masks:
        raise ValueError("at least one mask is required")
    shape = np.asarray(masks[0]).shape
    if any(np.asarray(mask).shape != shape for mask in masks):
        raise ValueError("all masks must have the same shape")
    return np.logical_and.reduce([np.asarray(mask, dtype=bool) for mask in masks])


def masked_psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray,
                data_range: float = 1.0) -> float:
    """PSNR over exactly *mask*, with one fixed caller-supplied intensity range."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape[:2] != mask.shape:
        raise ValueError("prediction, target, and mask shapes are incompatible")
    if not mask.any():
        raise ValueError("the common valid region is empty")
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    selected = difference[mask]
    mse = float(np.mean(selected * selected))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(float(data_range) ** 2 / mse)


@dataclass
class BurstResult:
    positions: np.ndarray
    origin: np.ndarray
    ground_truth_position: np.ndarray
    aligned_reconstructions: List[np.ndarray]
    aligned_ground_truth: np.ndarray
    masks: List[np.ndarray]
    ground_truth_mask: np.ndarray
    common_mask: np.ndarray
    psnr: np.ndarray

    @property
    def optimal_reference(self) -> int:
        return int(np.argmax(self.psnr))


def evaluate_burst(reconstructions: Sequence[np.ndarray], ground_truth: np.ndarray,
                   data_range: float = 1.0, interpolation_order: int = 3,
                   origin: Optional[Sequence[float]] = None) -> BurstResult:
    """Evaluate all precomputed reconstructions in one shared coordinate system."""
    images = [np.asarray(image) for image in reconstructions]
    if not images or any(image.shape != images[0].shape for image in images):
        raise ValueError("reconstructions must be a non-empty, equal-shaped sequence")
    if np.asarray(ground_truth).shape != images[0].shape:
        raise ValueError("ground truth and reconstructions must have the same shape")
    positions = estimate_reconstruction_positions(images)
    shared = shared_coordinate_origin(positions) if origin is None else np.asarray(origin, dtype=float)
    if shared.shape != (2,):
        raise ValueError("origin must contain (x,y)")
    gt_position = estimate_ground_truth_position(images, positions, ground_truth)
    aligned, masks = [], []
    for image, position in zip(images, positions):
        warped, mask = align_image_and_mask(image, shared - position, interpolation_order)
        aligned.append(warped)
        masks.append(mask)
    aligned_gt, gt_mask = align_image_and_mask(
        np.asarray(ground_truth), shared - gt_position, interpolation_order)
    common = common_valid_region(masks + [gt_mask])
    scores = np.asarray([masked_psnr(image, aligned_gt, common, data_range) for image in aligned])
    return BurstResult(positions, shared, gt_position, aligned, aligned_gt,
                       masks, gt_mask, common, scores)


def load_burst_directory(path: Path, expected_references: Optional[int] = 14
                         ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load one burst from NumPy files or ground_truth.png/ref_XX.png files."""
    stack_path, gt_path = path / "reconstructions.npy", path / "ground_truth.npy"
    if stack_path.is_file() and gt_path.is_file():
        stack, gt = np.load(str(stack_path)), np.load(str(gt_path))
        if stack.ndim not in (3, 4):
            raise ValueError("reconstructions.npy must have shape (N,H,W) or (N,H,W,C)")
        if expected_references is not None and stack.shape[0] != expected_references:
            raise ValueError("{} has {} reconstructions; expected {}".format(
                path, stack.shape[0], expected_references))
        return [stack[i] for i in range(stack.shape[0])], gt

    png_gt = path / "ground_truth.png"
    count = expected_references if expected_references is not None else 14
    png_refs = [path / "ref_{:02d}.png".format(index) for index in range(count)]
    png_paths = [png_gt] + png_refs
    missing = [str(image_path.name) for image_path in png_paths if not image_path.is_file()]
    if missing:
        raise FileNotFoundError("{} is missing {}".format(path, ", ".join(missing)))

    bit_depths = [_png_bit_depth(image_path) for image_path in png_paths]
    if len(set(bit_depths)) != 1:
        raise ValueError("PNG bit depths are inconsistent in {}: {}".format(path, bit_depths))
    bit_depth = bit_depths[0]
    scale = float((1 << bit_depth) - 1)
    images = []
    for image_path in png_paths:
        with Image.open(str(image_path)) as image:
            array = np.asarray(image)
        if array.dtype.kind != "u" or array.dtype.itemsize * 8 < bit_depth:
            raise ValueError("{} was not decoded at its {}-bit precision".format(
                image_path, bit_depth))
        images.append(array.astype(np.float64) / scale)
    if any(image.shape != images[0].shape for image in images[1:]):
        raise ValueError("PNG image shapes are inconsistent in {}".format(path))
    return images[1:], images[0]


def _png_bit_depth(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(26)
    if len(header) != 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("{} is not a valid PNG file".format(path))
    bit_depth = header[24]
    if bit_depth not in (8, 16):
        raise ValueError("{} uses unsupported {}-bit PNG samples".format(path, bit_depth))
    return bit_depth


def _write_outputs(results: Dict[str, BurstResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_fields = ["burst_id", "reference_frame", "displacement_x", "displacement_y",
                  "sce_psnr", "psnr_gain_vs_frame_0", "common_valid_fraction", "is_optimal"]
    with (output_dir / "per_burst.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_fields)
        writer.writeheader()
        for burst_id, result in results.items():
            best = result.optimal_reference
            for ref, (position, score) in enumerate(zip(result.positions, result.psnr)):
                writer.writerow(dict(burst_id=burst_id, reference_frame=ref,
                    displacement_x=position[0], displacement_y=position[1], sce_psnr=score,
                    psnr_gain_vs_frame_0=score - result.psnr[0],
                    common_valid_fraction=result.common_mask.mean(), is_optimal=(ref == best)))

    fields = ["record_type", "reference_frame", "burst_id", "mean_psnr",
              "mean_psnr_gain_vs_frame_0", "optimal_reference_frame", "mean_oracle_gain_vs_frame_0"]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        n = len(next(iter(results.values())).psnr)
        for ref in range(n):
            scores = np.asarray([result.psnr[ref] for result in results.values()])
            bases = np.asarray([result.psnr[0] for result in results.values()])
            writer.writerow(dict(record_type="reference_mean", reference_frame=ref,
                                 mean_psnr=scores.mean(), mean_psnr_gain_vs_frame_0=(scores - bases).mean()))
        gains = []
        for burst_id, result in results.items():
            gains.append(result.psnr.max() - result.psnr[0])
            writer.writerow(dict(record_type="burst_optimum", burst_id=burst_id,
                                 optimal_reference_frame=result.optimal_reference))
        writer.writerow(dict(record_type="oracle_mean", mean_oracle_gain_vs_frame_0=np.mean(gains)))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="directory containing one subdirectory per burst")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-range", type=float, default=1.0)
    parser.add_argument("--num-references", type=int, default=14,
                        help="expected stack length; use 0 to accept any length")
    args = parser.parse_args(argv)
    burst_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not burst_dirs:
        parser.error("input_dir contains no burst subdirectories")
    results = {}
    for path in burst_dirs:
        images, gt = load_burst_directory(path, args.num_references or None)
        results[path.name] = evaluate_burst(images, gt, args.data_range)
    _write_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
