"""Synthetic validation for the CPU-only SCE implementation."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


MODULE_PATH = Path(__file__).parents[1] / "shared_coordinate_evaluation.py"
SPEC = importlib.util.spec_from_file_location("shared_coordinate_evaluation", str(MODULE_PATH))
sce = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sce
SPEC.loader.exec_module(sce)


def scene(shape=(256, 288), seed=4):
    rng = np.random.RandomState(seed)
    image = ndimage.gaussian_filter(rng.normal(size=shape), 1.2)
    image += 0.5 * ndimage.gaussian_filter(rng.normal(size=shape), 5.0)
    image -= image.min()
    return image / image.max()


def translated_fourier(image, dx, dy):
    """Periodic synthetic translation, independent of SCE's spatial warp."""
    spectrum = np.fft.fftn(image)
    return np.fft.ifftn(ndimage.fourier_shift(spectrum, (dy, dx))).real


def test_known_integer_translation():
    base = scene()
    moving = translated_fourier(base, 17, -9)
    dx, dy = sce.estimate_translation(base, moving)
    np.testing.assert_allclose((dx, dy), (-17, 9), atol=0.08)


def test_known_subpixel_translation():
    base = scene()
    moving = translated_fourier(base, 3.35, -4.7)
    dx, dy = sce.estimate_translation(base, moving)
    np.testing.assert_allclose((dx, dy), (-3.35, 4.7), atol=0.12)


def test_large_110_pixel_translation():
    base = scene((320, 384))
    moving = translated_fourier(base, 110, -73)
    dx, dy = sce.estimate_translation(base, moving)
    np.testing.assert_allclose((dx, dy), (-110, 73), atol=0.12)


def test_equal_quality_translations_have_similar_psnr_and_same_pixels():
    base = scene((220, 240))
    positions = [(-11.0, 4.0), (-4.0, -3.0), (2.0, 7.0), (9.0, -6.0)]
    noise = 0.006 * scene(base.shape, seed=19)
    reconstructions = [translated_fourier(base + noise, x, y) for x, y in positions]
    ground_truth = translated_fourier(base, -2.0, 1.0)
    result = sce.evaluate_burst(reconstructions, ground_truth)
    assert np.ptp(result.psnr) < 0.18
    assert all(np.array_equal(result.common_mask, result.common_mask & mask)
               for mask in result.masks + [result.ground_truth_mask])


def test_invalid_border_pixels_do_not_affect_psnr():
    target = np.zeros((30, 32, 3), dtype=np.float64)
    prediction = target.copy()
    prediction[:4], prediction[-4:], prediction[:, :4], prediction[:, -4:] = 1000, 1000, 1000, 1000
    mask = np.zeros(target.shape[:2], dtype=bool)
    mask[4:-4, 4:-4] = True
    assert np.isinf(sce.masked_psnr(prediction, target, mask))


def test_blur_and_noise_lower_psnr():
    base = scene((100, 120))
    mask = np.ones(base.shape, dtype=bool)
    good = base + 0.002 * np.random.RandomState(1).normal(size=base.shape)
    noisy = base + 0.04 * np.random.RandomState(2).normal(size=base.shape)
    blurred = ndimage.gaussian_filter(base, 2.0)
    good_score = sce.masked_psnr(good, base, mask)
    assert sce.masked_psnr(noisy, base, mask) < good_score
    assert sce.masked_psnr(blurred, base, mask) < good_score


def test_changed_shared_origin_preserves_ranking_on_same_scene_region():
    base = scene((190, 210))
    positions = [(-7, 2), (0, -4), (8, 5)]
    qualities = [0.004, 0.012, 0.025]
    rng = np.random.RandomState(8)
    reconstructions = [translated_fourier(base + sigma * rng.normal(size=base.shape), x, y)
                       for (x, y), sigma in zip(positions, qualities)]
    first = sce.evaluate_burst(reconstructions, base, origin=(0, 0))
    second = sce.evaluate_burst(reconstructions, base, origin=(3.25, -2.5))
    common = first.common_mask & second.common_mask
    scores_a = [sce.masked_psnr(x, first.aligned_ground_truth, common) for x in first.aligned_reconstructions]
    scores_b = [sce.masked_psnr(x, second.aligned_ground_truth, common) for x in second.aligned_reconstructions]
    assert np.array_equal(np.argsort(scores_a), np.argsort(scores_b))


def test_alignment_mask_excludes_padding():
    image = np.ones((40, 50))
    aligned, mask = sce.align_image_and_mask(image, (8.5, -6.25), order=3)
    assert mask.dtype == bool
    assert 0 < mask.mean() < 1
    assert np.all(aligned[mask] > 0.99)
