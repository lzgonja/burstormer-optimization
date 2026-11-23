"""Tests for shift_geometry.py: the low-loss warp and the geometry estimates (CPU, under a minute).

The refined control moves outputs by geometries that are more than a translation. That is only valid if the warp
changes position and nothing else, and if the estimated geometry is the one that maps one output onto the other.

Run from `Burst Super-resolution/`:  python -m unittest tests.test_shift_geometry -v
"""

import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shift_geometry import (Geometry, _apply_projective, coverage, estimate_geometry, invert,  # noqa: E402
                            metric_patch_masks, move_reference, pixel_grid, to_raw,
                            sample, shift_image, upsample, warp_image)

SIZE = 320
EDGE = 40
INNER = (Ellipsis, slice(EDGE, -EDGE), slice(EDGE, -EDGE))


def texture(seed: int = 0, size: int = SIZE) -> torch.Tensor:
    """[1, 3, size, size] image in [0.05, 0.95] with detail at several scales, down to 2 px."""
    gen = torch.Generator().manual_seed(seed)
    img = torch.zeros(1, 3, size, size)
    for scale in (2, 4, 8, 32):
        coarse = torch.rand(1, 3, size // scale, size // scale, generator=gen)
        img += F.interpolate(coarse, size=(size, size), mode='bicubic', align_corners=False) / scale ** 0.5
    img = img - img.amin()
    return 0.05 + 0.9 * img / img.amax()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return -10 * math.log10(F.mse_loss(a, b).item() + 1e-20)


def grad_energy(x: torch.Tensor) -> float:
    return (x[..., 1:, :] - x[..., :-1, :]).pow(2).mean().item() + (x[..., :, 1:] - x[..., :, :-1]).pow(2).mean().item()


def projective_geometry(m: torch.Tensor, size: int = SIZE) -> Geometry:
    """Geometry whose sampling map is the 3x3 matrix m (on homogeneous (col, row, 1)), split into the translation
    at the image centre and a residual field."""
    p = pixel_grid(size, size, 'cpu')
    src = _apply_projective(m, p.permute(1, 2, 0)).permute(2, 0, 1)
    c = (size - 1) / 2
    centre_src = _apply_projective(m, torch.tensor([[c, c]]))[0]
    shift = torch.tensor([c, c]) - centre_src
    return Geometry(shift, src - (p - shift.view(2, 1, 1)), 'homography')


def rotation_matrix(deg: float, scale: float = 1.0, ty: float = 0.0, tx: float = 0.0, px: float = 0.0,
                    size: int = SIZE) -> torch.Tensor:
    """Sampling map: rotate by deg and scale about the centre, then shift by (ty, tx); px adds perspective."""
    c = (size - 1) / 2
    a = math.radians(deg)
    t1 = torch.tensor([[1, 0, -c], [0, 1, -c], [0, 0, 1]], dtype=torch.float64)
    r = torch.tensor([[scale * math.cos(a), -scale * math.sin(a), 0], [scale * math.sin(a), scale * math.cos(a), 0],
                      [px, 0, 1]], dtype=torch.float64)
    t2 = torch.tensor([[1, 0, c - tx], [0, 1, c - ty], [0, 0, 1]], dtype=torch.float64)
    return t2 @ r @ t1


class TestUpsampleAndSample(unittest.TestCase):
    def test_upsample_keeps_original_samples(self):
        img = texture(1)
        up = upsample(img, 2)
        self.assertTrue(torch.allclose(up[..., ::2, ::2], img, atol=1e-6))

    def test_sampling_on_the_grid_is_identity(self):
        img = texture(2)
        out = sample(img, pixel_grid(SIZE, SIZE, 'cpu'))
        self.assertGreater(psnr(out, img), 70.0)

    def test_constant_residual_equals_fourier_shift(self):
        # A sub-pixel residual of (0.3, -0.7) samples img at p + r, i.e. moves it by (-0.3, +0.7)
        img = texture(3)
        g = Geometry(torch.zeros(2), torch.tensor([0.3, -0.7]).view(2, 1, 1).expand(2, SIZE, SIZE).clone())
        self.assertGreater(psnr(warp_image(img, g)[INNER], shift_image(img, -0.3, 0.7)[INNER]), 45.0)

    def test_half_pixel_residual_keeps_detail(self):
        # Bilinear interpolation at half a pixel would lose about a third of the gradient energy
        img = texture(4)
        g = Geometry(torch.zeros(2), torch.full((2, SIZE, SIZE), 0.5))
        moved = warp_image(img, g)
        self.assertAlmostEqual(grad_energy(moved[INNER]) / grad_energy(img[INNER]), 1.0, delta=0.05)


class TestNoiseLevel(unittest.TestCase):
    # The metric scores noise as error, so a warp that smooths or boosts noise changes the score by itself.
    # 4x upsampling raised white noise power by 4% (0.18 dB); clipping to [0, 1] cut it by 25% near black.
    def noise_change_db(self, level: float, seed: int) -> float:
        gen = torch.Generator().manual_seed(seed)
        img = (torch.full((1, 3, SIZE, SIZE), level) + 0.01 * torch.randn(1, 3, SIZE, SIZE, generator=gen)).clamp(0, 1)
        out = warp_image(img, Geometry(torch.tensor([0.6, -0.3]), torch.full((2, SIZE, SIZE), 0.37)))
        return 10 * math.log10(out[INNER].var().item() / img[INNER].var().item())

    def test_white_noise_power_is_kept(self):
        self.assertLess(abs(self.noise_change_db(0.3, 0)), 0.01)

    def test_noise_near_black_is_kept(self):
        self.assertLess(abs(self.noise_change_db(0.005, 1)), 0.01)


class TestFillAndCoverage(unittest.TestCase):
    def test_fill_replaces_only_the_uncovered_strip(self):
        img, other = texture(12), texture(13)
        g = Geometry(torch.tensor([20.0, -9.0]))
        cov = coverage(g, SIZE, SIZE, 'cpu')
        out = warp_image(img, g, fill=other)
        # The strip the move uncovers: 20 rows at the top and 9 columns at the right, plus the 2 px margin
        self.assertFalse(cov[:20].any())
        self.assertFalse(cov[:, -9:].any())
        self.assertTrue(cov[25:-5, 5:-14].all())
        self.assertTrue(torch.equal(out[..., ~cov], other[..., ~cov]))
        self.assertTrue(torch.allclose(out[..., cov], warp_image(img, g)[..., cov]))

    def test_metric_patch_masks_follow_the_metric_patches(self):
        # A 640 px output (80 raw px) is scored as 3 x 3 patches of 384 px with stride 128 (utils.metrics)
        mask = torch.zeros(640, 640, dtype=torch.bool)
        mask[:, 300:] = True
        tiles = metric_patch_masks(mask, erode=0)
        self.assertEqual(len(tiles), 9)
        self.assertEqual(tuple(tiles[0].shape), (1, 1, 384, 384))
        # Patch columns start at 0, 128 and 256 of the output: the mask edge (column 300) falls at 300, 172 and 44
        for j, edge in enumerate((300, 172, 44)):
            for i in range(3):
                t = tiles[3 * i + j][0, 0]
                self.assertTrue(t[:, edge:].all() and not t[:, :edge].any(), msg=f'patch ({i}, {j})')


class TestRawReference(unittest.TestCase):
    def test_raw_geometry_is_the_output_geometry_scaled_down(self):
        g = projective_geometry(rotation_matrix(0.3, 1.001, ty=16.0, tx=-8.0))
        r = to_raw(g)
        self.assertTrue(torch.allclose(r.shift, g.shift / 8))
        self.assertEqual(tuple(r.residual.shape), (2, SIZE // 8, SIZE // 8))
        # Raw pixel q samples where its 8 output pixels sample, divided by 8
        q = (5, 7)
        out_mean = g.residual[:, 8 * q[0]:8 * q[0] + 8, 8 * q[1]:8 * q[1] + 8].mean(dim=(1, 2)) / 8
        self.assertTrue(torch.allclose(r.residual[:, q[0], q[1]], out_mean, atol=1e-5))

    def test_move_reference_moves_only_the_first_frame(self):
        burst = torch.stack([texture(20 + i, size=40)[:, :3] for i in range(3)], dim=1)  # [1, 3, 3, 40, 40]
        g = Geometry(torch.tensor([16.0, 8.0]))  # output pixels: 2 and 1 raw pixels
        moved = move_reference(burst, burst[:, 2], g, burst[:, 1])
        self.assertTrue(torch.equal(moved[:, 1:], burst[:, 1:]))
        expected = shift_image(burst[:, 2], 2.0, 1.0, clamp=False)
        # Content comes from raw frame 2 at least 2 px (the coverage margin) inside its edges
        self.assertTrue(torch.allclose(moved[:, 0, :, 4:-2, 3:-2], expected[..., 4:-2, 3:-2], atol=1e-5))
        self.assertTrue(torch.equal(moved[:, 0, :, :2], burst[:, 1, :, :2]))  # uncovered strip from fill


class TestWarpAndInverse(unittest.TestCase):
    def test_translation_only_geometry_equals_shift_image(self):
        img = texture(5)
        g = Geometry(torch.tensor([7.25, -3.5]))
        self.assertTrue(torch.allclose(warp_image(img, g), shift_image(img, 7.25, -3.5)))

    def test_rotation_round_trip_recovers_image(self):
        img = texture(6)
        g = projective_geometry(rotation_matrix(0.5, 1.003, ty=4.2, tx=-9.6))
        back = warp_image(warp_image(img, g), invert(g))
        self.assertGreater(psnr(back[INNER], img[INNER]), 45.0)

    def test_inverse_composes_to_identity_map(self):
        g = projective_geometry(rotation_matrix(0.8, 0.997, ty=-12.0, tx=6.0, px=2e-5))
        gi = invert(g)
        p = pixel_grid(SIZE, SIZE, 'cpu')
        # s(t(p)) must be p, where s and t are the sampling maps of g and its inverse
        t = p - gi.shift.view(2, 1, 1) + gi.residual
        s_of_t = t - g.shift.view(2, 1, 1) + F.grid_sample(
            g.residual[None], torch.stack((2 * t[1] / (SIZE - 1) - 1, 2 * t[0] / (SIZE - 1) - 1), -1)[None],
            mode='bilinear', align_corners=True, padding_mode='border')[0]
        self.assertLess((s_of_t - p)[INNER].abs().max().item(), 0.01)


class TestEstimate(unittest.TestCase):
    def check_recovers(self, model: str, m: torch.Tensor, tol: float, seed: int):
        a = texture(seed)
        g_true = projective_geometry(m)
        b = warp_image(a, g_true)
        g = estimate_geometry(a, b, model, tile=96, grid=5)
        err = (g.displacement(SIZE, SIZE) - g_true.displacement(SIZE, SIZE))[INNER].abs().max().item()
        self.assertLess(err, tol, msg=f'{model}: max displacement error {err:.3f} px')
        return g

    def test_homography_recovers_rotation_scale_and_shift(self):
        self.check_recovers('homography', rotation_matrix(0.4, 1.002, ty=6.0, tx=-11.0), 0.1, seed=7)

    def test_homography_recovers_perspective(self):
        self.check_recovers('homography', rotation_matrix(0.2, 1.0, ty=-3.0, tx=4.5, px=3e-5), 0.12, seed=8)

    def test_affine_recovers_rotation(self):
        self.check_recovers('affine', rotation_matrix(-0.3, 0.999, ty=2.0, tx=15.0), 0.1, seed=9)

    def test_estimated_geometry_lands_a_on_b(self):
        # The direction used by the control: warp_image(O_0, estimate_geometry(O_0, O_k)) must land on O_k
        a = texture(10)
        b = warp_image(a, projective_geometry(rotation_matrix(0.6, 1.0, ty=-8.0, tx=3.0)))
        g = estimate_geometry(a, b, 'homography', tile=96, grid=5)
        moved = warp_image(a, g)
        translated = warp_image(a, estimate_geometry(a, b, 'translation'))
        self.assertGreater(psnr(moved[INNER], b[INNER]), 38.0)
        self.assertGreater(psnr(moved[INNER], b[INNER]), psnr(translated[INNER], b[INNER]) + 10.0)

    def test_unreliable_tiles_fall_back_to_translation(self):
        # Content swamped by noise: the tile shifts disagree, so the fit falls back to the global translation
        a = texture(14)
        gen = torch.Generator().manual_seed(3)
        b = 0.5 + 0.3 * torch.randn(1, 3, SIZE, SIZE, generator=gen)
        g = estimate_geometry(a, b, 'homography', tile=96, grid=5)
        self.assertTrue(g.info['fallback'])
        self.assertIsNone(g.residual)

    def test_pure_translation_gives_near_zero_residual(self):
        a = texture(11)
        b = shift_image(a, 4.4, -2.1)
        g = estimate_geometry(a, b, 'homography', tile=96, grid=5)
        self.assertLess(g.residual[INNER].abs().max().item(), 0.1)


if __name__ == '__main__':
    unittest.main()
