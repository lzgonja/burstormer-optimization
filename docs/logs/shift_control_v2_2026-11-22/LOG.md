# Shift control v2: work log, 2026-11-22

Refining the shift control so that its corrected gains have a measured accuracy. Method:
[SHIFT_CONTROL.md](../../SHIFT_CONTROL.md). Evidence: [SHIFT_CONTROL_VALIDATION.md](../../SHIFT_CONTROL_VALIDATION.md).
Earlier logs: [2026-11-21](../shift_control_2026-11-21/LOG.md) (frames 0-4, 40 dev and 40 test bursts) and
[2026-11-22](../shift_control_2026-11-22/LOG.md) (all 14 frames on the 5 bursts).

Setup: model `burstormer_burstsr_train.pth` (T02), BurstSR val, the 5 bursts in [bursts.txt](bursts.txt), all 14
reference frames. Branch `shift-control-v2`. RTX 4070 Super.

## 1. The limits being addressed

From the 14-frame run of 2026-11-22:

- **L1, translation only.** The moved copy matched the real output at 42-49 dB on average, 31-32 dB at worst.
- **L2, the two estimates disagreed.** Forward (move O_0 to frame k) and reverse (move O_k back to frame 0) gave
  per-burst correlations near zero or negative, and differed by up to 0.19 dB on average at frames 9-13.
- **L3, no ground truth.** Every check was self-consistency, so there was no way to say which estimate was right.
- **L4, the region depended on the run.** Scores used the pixels common to all 14 outputs (58%).

## 2. New code

- **`shift_geometry.py`.** Shared geometry code:
  - Four models: translation; affine and homography, fitted with robust least squares to phase correlation on a
    5 x 5 grid of 128 px tiles; dense, PWC-Net flow between the two outputs after the global shift.
  - The warp: the existing exact Fourier shift, plus band-limited sampling of the rest of the geometry.
- **`shift_control_benchmark.py`.** A known-answer benchmark (section 3).
- **`shift_control.py`.**
  - New options: `--geometry`, `--estimator {reverse,forward}`, `--region {pair,all}` and `--factor`.
  - Both estimates are now always computed.
  - New defaults: homography, reverse, pair.
- **Tests.** `tests/test_shift_geometry.py` has 14 new tests. With the 10 existing ones, all 24 pass.

## 3. Known-answer benchmark

For each burst and frame k:

1. Take the real geometry from O_0 to O_k.
2. Change O_0's content in a known way, C = D(O_0): none, Gaussian blur of 0.5 or 1.0 output px, or noise with
   σ = 0.01.
3. Build a synthetic O_k* by moving C with that geometry.
4. The known answer is score(C as frame 0) minus score(O_0 as frame 0). Both images are at frame 0's position, so
   no penalty is involved.

Every estimator sees only O_0, O_k* and the burst. 5 bursts x 13 frames x 4 changes = 260 cases.

### First attempt ([benchmark_first_attempt/](benchmark_first_attempt/))

With a 4x-upsampled warp that clipped to [0, 1], and the raw dense flow as true geometry:

- Forward was biased: +0.10 dB, RMSE 0.29 dB. Reverse was much better: RMSE 0.064 dB with homography.
- Two problems showed in the worst cases:
  - **Noise.** Burst 0008_0024 with noise had a +0.2 dB error even at frame 1, which barely moves. Tested
    directly: 4x upsampling plus bicubic raises white-noise power by 4% (0.18 dB). Clipping to [0, 1] cuts it by
    25% near black. BurstSR outputs are linear RGB and often dark, so this matters on real data too.
  - **Flow failure.** On 0010_0027 frame 7 the "true" geometry had an 11 px flow failure in one corner.

### Fixes

- **The warp now upsamples 8x and does not clip.** Noise power is now kept within 0.002 dB, on mid-grey and near
  black. There are two new tests for it.
- **The true geometry keeps the flow only where it is consistent both ways (within 1 px).** Elsewhere it uses the
  homography fit (`consistent_dense`). For 0010_0027 frame 7 this replaced 10% of the pixels and removed the
  corner failure.

### Final benchmark ([benchmark/](benchmark/))

Warp round-trip loss: mean −0.006 dB, worst 0.05 dB. This is the floor for measuring errors here.

| Estimator | Bias (dB) | RMSE (dB) | Worst error (dB) | RMSE, 24-60 px | RMSE, 60+ px |
|---|---|---|---|---|---|
| Forward, translation (original method) | +0.090 | 0.285 | 1.44 | 0.39 | 0.65 |
| Forward, homography | +0.088 | 0.283 | 1.45 | 0.39 | 0.63 |
| Reverse, translation | +0.001 | 0.029 | 0.12 | 0.041 | 0.050 |
| **Reverse, homography (default)** | **−0.003** | **0.030** | **0.09** | **0.039** | **0.037** |
| Reverse, dense | −0.003 | 0.024 | 0.08 | 0.035 | 0.025 |
| Mean of forward and reverse, homography | +0.042 | 0.142 | 0.73 | 0.20 | 0.31 |

RMSE by content change:

| Estimator | None | Blur 0.5 | Blur 1.0 | Noise |
|---|---|---|---|---|
| Forward, translation | 0.040 | 0.072 | 0.346 | 0.447 |
| Reverse, homography | 0.026 | 0.024 | 0.023 | 0.042 |

- **The forward estimate is accurate only when frame k's content matches frame 0's.** With different content
  (blur, noise) it is off by 0.35-0.45 dB RMS, and the error grows with the shift. The penalty depends on the
  content being moved, which explains the forward/reverse disagreement (L2).
- **The reverse estimate is unbiased and accurate to 0.02-0.03 dB RMS** at every shift and with every content
  change. Its worst case is 0.08-0.12 dB.
- **The geometry model matters less than the direction:**
  - Reverse with translation is about as good as homography on these bursts.
  - Dense is best, but the true geometry is built from the same flow network, which flatters it.
  - Homography is the default because it is fitted independently of the flow and cannot follow content changes.

## 4. Regression

`--geometry translation --region all --estimator forward` is the original method. It reproduces the 2026-11-22
numbers to within 0.012 dB per row: gain_real within 0.0002, forward within 0.008, reverse within 0.011.
The difference comes from the warp no longer clipping ([original_mode_check/](original_mode_check/)).

## 5. Refined control on the 5 bursts x 14 frames ([five_bursts_14_frames/](five_bursts_14_frames/))

Defaults: homography, reverse, pair region (61% of pixels).

| Frame | Shift (px) | Penalty (dB) | Measured gain (dB) | Corrected gain (dB) |
|---|---|---|---|---|
| 1-3 | 2-7 | +0.03 | −0.04 to −0.00 | −0.00 to +0.03 |
| 4-7 | 9-17 | +0.18 to +0.32 | −0.10 to −0.27 | +0.04 to +0.08 |
| 8-9 | 21-23 | +0.55 to +0.56 | −0.54 to −0.58 | −0.04 to +0.02 |
| 10-13 | 29-40 | +1.04 to +1.55 | −1.02 to −1.63 | −0.08 to +0.01 |

- Frame 13: measured −1.63 dB, penalty +1.55 dB [+0.04, +4.26], corrected −0.08 dB [−0.40, +0.23].
- **No frame beats frame 0 once corrected.** Every interval includes zero.
- **Best frame per burst:** measured +0.03 dB, corrected +0.18 dB [+0.06, +0.33]. Part of the corrected figure is
  selection noise over 13 noisy estimates.
- **The probes are unchanged:** half-pixel −0.015 dB, 8 x 8 px +0.012 dB.
- **Fidelity:** with homography, the moved copy matches the real output at 43-49 dB. With dense it is 47-52 dB.
  Content differences between reconstructions, not geometry, limit how far this goes.

Compared with the 2026-11-22 forward/translation numbers, the corrected gains of frames 10-13 fall from
+0.05 to +0.13 dB to −0.08 to +0.01 dB. On the benchmark the old estimator ran high (+0.09 dB bias overall, larger
at large shifts), which matches this.

## 6. What is still open

- **The 40 dev and 40 test runs of 2026-11-21 used the forward estimator.** Their claims that frames 3 and 4
  beat frame 0 rest on the estimate now known to run high. Their reverse column (+0.03 to +0.08 dB on test) is
  the better number. Rerunning them with v2 defaults would settle it.
- **The benchmark changes content only by blur and noise.** Real reconstructions differ in more complex ways.
  The reverse estimate needs no assumption about content, but its geometry estimate could in principle be
  disturbed by structural differences.
- **Five bursts:** intervals stay wide.
