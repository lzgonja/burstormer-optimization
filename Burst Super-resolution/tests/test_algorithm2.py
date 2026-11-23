"""Tests for reference_selection/algorithm2.py on synthetic bursts (CPU, a few seconds).

Run from `Burst Super-resolution/`:  python -m unittest tests.test_algorithm2 -v
"""

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference_selection import algorithm2 as alg2  # noqa: E402

SHOT_NOISE = 1.8e-4  # BurstSR EXIF noise profile at ISO 50: variance = 1.8e-4 * x


def make_scene(size: int, gen: torch.Generator, mean_level: float = 0.08) -> torch.Tensor:
    """Textured 4-channel scene [4, size, size] with detail at several scales, scaled to a dark mean level."""
    scene = torch.zeros(1, 1, size, size)
    for scale in (2, 4, 8, 16):
        coarse = torch.rand(1, 1, size // scale + 1, size // scale + 1, generator=gen)
        scene += F.interpolate(coarse, size=(size, size), mode='bicubic', align_corners=False) / scale ** 0.5
    scene = scene - scene.min()
    scene = scene / scene.mean() * mean_level
    gains = torch.tensor([0.6, 1.0, 1.0, 0.5]).view(4, 1, 1)  # RGGB: raw green is brightest
    return (scene[0] * gains).clamp(min=0.0)


def make_burst(n: int = 14, size: int = 64, max_shift: int = 6, seed: int = 0, mean_level: float = 0.08):
    """Burst [n, 4, size, size]: shifted crops of one scene plus shot noise. Returns (burst, generator)."""
    gen = torch.Generator().manual_seed(seed)
    scene = make_scene(size + 2 * max_shift, gen, mean_level)
    frames = []
    for i in range(n):
        dy, dx = (0, 0) if i == 0 else torch.randint(0, 2 * max_shift + 1, (2,), generator=gen).tolist()
        dy, dx = (max_shift, max_shift) if i == 0 else (dy, dx)
        frames.append(scene[:, dy:dy + size, dx:dx + size])
    burst = torch.stack(frames)
    burst = burst + torch.randn(burst.shape, generator=gen) * (burst * SHOT_NOISE).sqrt()
    return burst.clamp(0.0, 1.0), gen


def blur(frame: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(3 * sigma + 0.999))
    x = torch.arange(-radius, radius + 1, dtype=frame.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    kernel = (k[:, None] * k[None, :]).view(1, 1, 2 * radius + 1, 2 * radius + 1)
    return F.conv2d(F.pad(frame[:, None], [radius] * 4, mode='reflect'), kernel)[:, 0]


def add_noise(frame: torch.Tensor, shot: float, gen: torch.Generator) -> torch.Tensor:
    return frame + torch.randn(frame.shape, generator=gen) * (frame.clamp(min=0) * shot + shot ** 2).sqrt()


def expose(frame: torch.Tensor, k: float, gen: torch.Generator) -> torch.Tensor:
    """The same frame captured with k times the light: the extra shot noise a shorter exposure has (k < 1),
    clipping at white (k > 1), and 10-bit quantisation."""
    out = frame * k
    if k < 1:
        out = out + torch.randn(frame.shape, generator=gen) * (SHOT_NOISE * k * (1 - k) * frame.clamp(min=0)).sqrt()
    return (out.clamp(0.0, 1.0) * 1023).round() / 1023


class TestAlignment(unittest.TestCase):
    def test_phase_correlation_recovers_shift_and_aligns(self):
        gen = torch.Generator().manual_seed(1)
        scene = make_scene(80, gen)[1]
        a, b = scene[8:72, 8:72], scene[11:75, 5:69]  # b shows the scene moved by (-3, +3)
        shifts = alg2._phase_shifts(torch.stack([a, b]), ref=0)
        self.assertAlmostEqual(shifts[1, 0].item(), 3.0, delta=0.25)
        self.assertAlmostEqual(shifts[1, 1].item(), -3.0, delta=0.25)
        aligned = alg2._fourier_shift(torch.stack([a, b]), shifts)
        inner = (slice(5, -5), slice(5, -5))
        err = (aligned[1][inner] - a[inner]).abs().mean() / a.mean()
        self.assertLess(err.item(), 0.05)


class TestCost(unittest.TestCase):
    def test_brightness_difference_raises_the_cost(self):
        burst, _ = make_burst(seed=2)
        clean = alg2.frame_costs(burst)
        for k in (2.0, 0.5):  # no clipping at these levels
            damaged = burst.clone()
            damaged[0] = burst[0] * k
            cost = alg2.frame_costs(damaged)[0]
            # mean|k x - x| / mean x = |k - 1| on top of the noise residual of a healthy frame
            self.assertAlmostEqual(cost, abs(k - 1), delta=0.1 + clean[0])
            self.assertGreater(cost, 3 * clean[0])

    def test_cost_does_not_depend_on_the_scene_level(self):
        # Exposure normalisation: the same burst twice as bright (no clipping) has the same costs
        burst, _ = make_burst(seed=2)
        clean, scaled = alg2.frame_costs(burst), alg2.frame_costs(burst * 2)
        for c in clean:
            self.assertAlmostEqual(scaled[c], clean[c], delta=1e-4 * clean[c])


class TestSelection(unittest.TestCase):
    def assert_switches(self, burst: torch.Tensor, msg: str, avoid=(0,)):
        ref = alg2.select_reference_frame(burst)
        self.assertNotIn(ref, avoid, f'{msg}: kept frame {ref}, costs {alg2.frame_costs(burst)}')
        self.assertLess(ref, alg2.WINDOW)

    def test_clean_bursts_keep_default(self):
        for seed in range(6):
            burst, _ = make_burst(seed=seed)
            self.assertEqual(alg2.select_reference_frame(burst), alg2.DEFAULT_REF,
                             f'seed {seed}: costs {alg2.frame_costs(burst)}')

    def test_interface(self):
        burst, _ = make_burst(n=8)
        ref = alg2.select_reference_frame(burst[None])  # [1, N, 4, H, W] as in the evaluation scripts
        self.assertIsInstance(ref, int)
        self.assertTrue(0 <= ref < 8)
        self.assertEqual(alg2.select_reference_frame(burst[:1]), 0)
        self.assertEqual(alg2.select_reference_frame(burst[:3]), alg2.select_reference_frame(burst[:3]))
        with self.assertRaises(ValueError):
            alg2.select_reference_frame(torch.zeros(2, 3, 4, 8, 8))

    def test_same_interface_as_algorithm1(self):
        # Same inputs, outputs and errors as algorithm1.select_reference_frame
        burst, gen = make_burst(seed=4)
        burst[0] = add_noise(burst[0], 1e-2, gen)  # so the answer is not just the default frame
        ref = alg2.select_reference_frame(burst)
        self.assertIsInstance(ref, int)
        array = burst.numpy().copy()
        self.assertEqual(alg2.select_reference_frame(array), ref)  # numpy input gives the same answer
        np.testing.assert_array_equal(array, burst.numpy())       # and is left unchanged
        for bad in (torch.zeros(0, 4, 16, 16), torch.zeros(3, 3, 16, 16), torch.zeros(4, 16, 16)):
            with self.assertRaises(ValueError):
                alg2.select_reference_frame(bad)
        for value in (float('nan'), float('inf')):
            damaged = burst.clone()
            damaged[2, 1, 5, 5] = value
            with self.assertRaises(ValueError):
                alg2.select_reference_frame(damaged)

    def test_blurred_default_is_replaced(self):
        for sigma in (1.0, 2.0):
            burst, _ = make_burst(seed=3)
            burst[0] = blur(burst[0], sigma)
            self.assert_switches(burst, f'blur sigma {sigma}')

    def test_noisy_default_is_replaced(self):
        for shot in (3e-3, 1e-2):
            burst, gen = make_burst(seed=4)
            burst[0] = add_noise(burst[0], shot, gen)
            self.assert_switches(burst, f'noise {shot}')

    def test_under_exposed_default_is_replaced(self):
        for k in (0.25, 0.125):
            burst, gen = make_burst(seed=5)
            burst[0] = expose(burst[0], k, gen)
            self.assert_switches(burst, f'exposure x{k}')

    def test_over_bright_default_is_replaced(self):
        # Twice as bright as the rest of the burst, no clipping and no extra noise: still a bad reference
        burst, gen = make_burst(seed=10)
        burst[0] = expose(burst[0], 2.0, gen)
        self.assert_switches(burst, 'exposure x2')

    def test_clipped_default_is_replaced(self):
        burst, gen = make_burst(seed=6, mean_level=0.15)
        burst[0] = expose(burst[0], 8.0, gen)
        self.assert_switches(burst, 'exposure x8 (clipped)')

    def test_never_moves_to_a_damaged_frame(self):
        # Frame 1 is very noisy and frame 0 is blurred: the selector must leave frame 0 and not pick frame 1
        burst, gen = make_burst(seed=7)
        burst[1] = add_noise(burst[1], 1e-2, gen)
        burst[0] = blur(burst[0], 2.0)
        self.assert_switches(burst, 'blurred default with a noisy neighbour', avoid=(0, 1))
        # A damaged non-default frame alone does not cause a switch
        burst, gen = make_burst(seed=8)
        burst[2] = expose(burst[2], 0.125, gen)
        self.assertEqual(alg2.select_reference_frame(burst), 0)

    def test_fallback_is_the_earliest_acceptable_frame(self):
        for seed in range(3):
            burst, gen = make_burst(seed=20 + seed)
            burst[0] = add_noise(burst[0], 1e-2, gen)
            self.assertEqual(alg2.select_reference_frame(burst), 1, f'seed {seed}: costs {alg2.frame_costs(burst)}')

    def test_choose_from_costs_rule(self):
        self.assertEqual(alg2.choose_from_costs({0: 1.10, 1: 1.0, 2: 1.0, 3: 0.9}, margin=0.15), 0)
        self.assertEqual(alg2.choose_from_costs({0: 1.20, 1: 1.0, 2: 1.0, 3: 0.9}, margin=0.15), 1)
        self.assertEqual(alg2.choose_from_costs({0: 1.20, 1: 1.5, 2: 1.0, 3: 0.9}, margin=0.15), 2)
        self.assertEqual(alg2.choose_from_costs({2: 5.0, 0: 1.0, 1: 2.0}, default=2, margin=0.15), 0)
        self.assertEqual(alg2.choose_from_costs({0: 1.0}), 0)
        with self.assertRaises(ValueError):
            alg2.choose_from_costs({1: 1.0, 2: 1.0}, default=0)


if __name__ == '__main__':
    unittest.main()
