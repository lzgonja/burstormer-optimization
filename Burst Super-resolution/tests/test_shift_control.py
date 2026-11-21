"""Tests for the shift operations in shift_control.py (CPU, a few seconds).

The control is only valid if moving an image changes its position and nothing else, and if the measured shift is
the one that moves the frame-0 output onto the frame-k output. These tests check both on synthetic images.

Run from `Burst Super-resolution/`:  python -m unittest tests.test_shift_control -v
"""

import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shift_control import PAD, shift_image, translation  # noqa: E402

SIZE = 256
INNER = (Ellipsis, slice(PAD, -PAD), slice(PAD, -PAD))  # away from the image edges, where padding is used


def texture(seed: int = 0) -> torch.Tensor:
    """[1, 3, SIZE, SIZE] image in [0.05, 0.95] with detail at several scales."""
    gen = torch.Generator().manual_seed(seed)
    img = torch.zeros(1, 3, SIZE, SIZE)
    for scale in (2, 4, 8, 32):
        coarse = torch.rand(1, 3, SIZE // scale, SIZE // scale, generator=gen)
        img += F.interpolate(coarse, size=(SIZE, SIZE), mode='bicubic', align_corners=False) / scale ** 0.5
    img = img - img.amin()
    return 0.05 + 0.9 * img / img.amax()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return -10 * math.log10(F.mse_loss(a, b).item() + 1e-20)


class TestShiftImage(unittest.TestCase):
    def test_zero_shift_is_identity(self):
        img = texture()
        self.assertTrue(torch.allclose(shift_image(img, 0.0, 0.0), img, atol=1e-5))

    def test_integer_shift_equals_moving_pixels(self):
        img = texture(1)
        moved = shift_image(img, 5.0, -7.0)
        expected = torch.roll(img, shifts=(5, -7), dims=(-2, -1))
        self.assertGreater(psnr(moved[INNER], expected[INNER]), 80.0)

    def test_sub_pixel_round_trip_loses_nothing(self):
        # Moving there and back must give the original: the shift adds no blur
        img = texture(2)
        back = shift_image(shift_image(img, 2.3, -1.7), -2.3, 1.7)
        self.assertGreater(psnr(back[INNER], img[INNER]), 60.0)

    def test_half_pixel_shift_keeps_detail(self):
        # Bilinear interpolation at half a pixel would blur; the Fourier shift keeps the image's detail energy
        img = texture(3)
        moved = shift_image(img, 0.5, 0.5)
        grad = lambda x: (x[..., 1:, :] - x[..., :-1, :]).pow(2).mean().item()
        self.assertAlmostEqual(grad(moved[INNER]) / grad(img[INNER]), 1.0, delta=0.05)

    def test_shift_larger_than_padding_is_rejected(self):
        with self.assertRaises(ValueError):
            shift_image(texture(), PAD, 0.0)


class TestTranslation(unittest.TestCase):
    def test_recovers_integer_shift(self):
        a = texture(4)
        b = shift_image(a, 6.0, -11.0)
        dy, dx = translation(a, b).tolist()
        self.assertAlmostEqual(dy, 6.0, delta=0.05)
        self.assertAlmostEqual(dx, -11.0, delta=0.05)

    def test_recovers_sub_pixel_shift(self):
        a = texture(5)
        for true in ((3.25, -6.5), (-0.4, 0.7), (12.6, 9.3)):
            b = shift_image(a, *true)
            dy, dx = translation(a, b).tolist()
            self.assertAlmostEqual(dy, true[0], delta=0.25, msg=f'shift {true}')
            self.assertAlmostEqual(dx, true[1], delta=0.25, msg=f'shift {true}')

    def test_sub_pixel_accuracy_over_random_shifts(self):
        # Measured on 30 random shifts in [-20, 20]: mean error 0.07 px, max 0.13 px. A shift error this small does
        # not change the metric (a deliberate 0.7 px shift costs 0.00 dB, docs/SHIFT_CONTROL_VALIDATION.md)
        gen = torch.Generator().manual_seed(0)
        errors = []
        for s in range(10):
            true = (torch.rand(2, generator=gen) * 40 - 20).tolist()
            a = texture(10 + s)
            errors.append((translation(a, shift_image(a, *true)) - torch.tensor(true)).abs().max().item())
        self.assertLess(max(errors), 0.15)

    def test_measured_shift_moves_a_onto_b(self):
        # The direction used by the control: shift_image(O_0, translation(O_0, O_k)) must land on O_k
        a = texture(6)
        b = shift_image(a, -4.3, 17.8)
        moved = shift_image(a, *translation(a, b).tolist())
        wrong = shift_image(a, *(-translation(a, b)).tolist())
        self.assertGreater(psnr(moved[INNER], b[INNER]), 35.0)  # limited by the ~0.1 px estimation error
        self.assertGreater(psnr(moved[INNER], b[INNER]), psnr(wrong[INNER], b[INNER]) + 15.0)


if __name__ == '__main__':
    unittest.main()
