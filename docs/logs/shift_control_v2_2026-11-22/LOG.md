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
  - New defaults: homography, reverse, pair. Section 5 adds `--fill target` and `--colour_ref moved`.
- **Tests.** `tests/test_shift_geometry.py` has 14 new tests. With the 10 existing ones, all 24 pass (29 after
  section 5).

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

### Second benchmark, 5 bursts ([benchmark_5_bursts_mirror_fill/](benchmark_5_bursts_mirror_fill/))

Warp round-trip loss: mean −0.006 dB, worst 0.05 dB. This is the floor for measuring errors here.

| Estimator | Bias (dB) | RMSE (dB) | Worst error (dB) | RMSE, 24-60 px | RMSE, 60+ px |
|---|---|---|---|---|---|
| Forward, translation (original method) | +0.090 | 0.285 | 1.44 | 0.39 | 0.65 |
| Forward, homography | +0.088 | 0.283 | 1.45 | 0.39 | 0.63 |
| Reverse, translation | +0.001 | 0.029 | 0.12 | 0.041 | 0.050 |
| **Reverse, homography** | **−0.003** | **0.030** | **0.09** | **0.039** | **0.037** |
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
  - Homography became the default because it is fitted independently of the flow and cannot follow content
    changes.
- **These 5 bursts have shifts of at most 40 px, except 0019_0020.** Section 5 shows what they missed.

## 4. Regression

The original method is `--geometry translation --region all --estimator forward --fill mirror --colour_ref fixed`.
It reproduces the 2026-11-22 numbers to within 0.012 dB per row ([original_mode_check/](original_mode_check/)):
- gain_real within 0.0002 dB;
- forward within 0.008 dB;
- reverse within 0.011 dB.

The only difference is that the warp no longer clips.

## 5. First runs on 40 dev and 40 test bursts

With the section 3 method (mirror fill, colour reference fixed), the 40-burst runs disagreed far more than the
benchmark allowed. Forward and reverse were up to 7.5 dB apart on single bursts:
- test 0105_0001 frame 9: measured +0.61 dB, forward +4.79 dB, reverse −2.76 dB;
- dev 0212_0020 and 0156_0051.

**Known-answer benchmark on those 4 bursts.** Reverse with homography was off by 1.11 dB RMS on 0212_0020 and
0.36 dB on 0156_0051. Moving an image there and back alone lost up to 2.7 dB. Three causes were found.

### Cause 1: the strip a move uncovers

Moving an image by 40-70 px leaves a strip with no content, which the warp filled by mirroring. The metric fits
one colour correction per 384 px patch by least squares over all its pixels. The mirrored strip spoils the flow
and the colour fit for the whole patch, even though the strip itself is not scored.

- **Fix: `--fill target` (default).** The strip is filled with the real content of the image being moved onto:
  frame 0's output when moving back, frame k's when moving forward. It is left out of the score with an exact
  coverage mask (`coverage`, `metric_patch_masks`).
- **The benchmark is fixed too.** The synthetic O_k* now shows the real frame-k output where frame 0's view ends,
  as a real output would.
- **Effect.** On 0212_0020 the warp's round-trip loss fell from 2.7 dB to under 0.01 dB, and the reverse RMSE from
  1.11 dB to 0.04 dB.

### Cause 2: the homography on noisy content

With strong noise, the 128 px tile correlations of 0212_0020 were useless, and the homography missed its own tiles
by 14-19 px (at most 0.7 px otherwise). Blurring before the correlation did not help.

- **Fix:** when the fit misses its tiles by more than 1 px RMS, use the global translation instead.
- **On real data** this happens in 35-60% of comparisons (47% on the 5 bursts, 60% dev, 35% test), because outputs
  from different reference frames differ in content. Reverse with translation is nearly as accurate as with a
  homography (section 7), so the fallback costs little.

### Cause 3: the colour reference

The metric fits its colour correction from the reference input frame to the ground truth, then applies it to the
output. That assumes the output has its reference frame's colours. Before, a moved output was scored with the raw
frame of the position it was moved to: frame k's output, moved back, was colour-matched with frame 0's raw
frame. When exposure or lighting changed between frames, the fit was wrong and both estimates paid several dB.

- **Fix: `--colour_ref moved` (default).** The output's own raw frame moves along with it (`move_reference`, with
  the geometry scaled to raw pixels by `to_raw`). The probes already worked this way. Only the position changes.
- **Effect on the problem bursts** (gain over frame 0, dB). "Before" already has the target fill (cause 1); only the
  colour reference differs:

| Burst, frame | Measured | Forward before | Reverse before | Forward after | Reverse after |
|---|---|---|---|---|---|
| 0105_0001, 9 | +0.61 | +4.79 | −2.76 | −0.03 | +0.04 |
| 0105_0001, 13 | +0.44 | +2.46 | −0.65 | −0.25 | +0.08 |
| 0077_0014, 11 | −4.30 | +1.26 | −1.34 | −0.20 | +0.22 |
| 0156_0051, 9 | −4.98 | −0.98 | −2.40 | −0.79 | −2.31 |
| 0212_0020, 11 | −3.28 | −2.21 | −0.55 | −2.34 | −0.57 |

On 0105_0001 and 0077_0014 the two estimates now agree. On 0156_0051 and 0212_0020 (60-70 px shifts) they still
differ. The benchmark puts reverse at 0.11-0.13 dB RMS on those two bursts and forward at 0.46-0.90 dB.

## 6. Tests

- 29 tests pass: 10 in `tests/test_shift_control.py`, 19 in `tests/test_shift_geometry.py`.
- New since section 2: the fill replaces only the uncovered strip; the patch masks follow the metric's patches;
  the homography falls back on unreliable tiles; the raw geometry is the output geometry scaled down; moving the
  reference changes only the first frame.

## 7. Final benchmark, 9 bursts ([benchmark_9_bursts/](benchmark_9_bursts/))

The 5 bursts plus the 4 hardest from the 40-burst runs (0105_0001, 0077_0014, 0212_0020, 0156_0051). 468 cases,
shifts up to 110 px. Warp round-trip loss: mean −0.004 dB, worst 0.10 dB.

| Estimator | Bias (dB) | RMSE (dB) | Worst error (dB) | RMSE 0-8 px | 8-24 px | 24-60 px | 60+ px |
|---|---|---|---|---|---|---|---|
| Forward, translation (original method) | +0.038 [+0.001, +0.071] | 0.385 | 4.17 | 0.132 | 0.161 | 0.543 | 0.708 |
| Forward, homography | +0.036 | 0.384 | 4.17 | 0.133 | 0.161 | 0.541 | 0.705 |
| Reverse, translation | −0.008 | 0.070 | 0.38 | 0.031 | 0.048 | 0.089 | 0.131 |
| **Reverse, homography (default)** | **−0.011 [−0.017, −0.005]** | **0.068** | **0.38** | **0.033** | **0.045** | **0.085** | **0.126** |
| Reverse, dense | −0.004 | 0.072 | 0.80 | 0.037 | 0.042 | 0.108 | 0.088 |
| Mean of both, homography | +0.012 | 0.195 | 2.07 | 0.067 | 0.090 | 0.277 | 0.347 |

RMSE by content change (dB):

| Estimator | None | Blur 0.5 | Blur 1.0 | Noise |
|---|---|---|---|---|
| Forward, translation | 0.085 | 0.094 | 0.283 | 0.705 |
| Reverse, homography | 0.063 | 0.063 | 0.052 | 0.088 |

- **The default is accurate to 0.068 dB RMS, with a bias of −0.011 dB.** Its error grows with the shift: 0.03 dB
  under 8 px, 0.13 dB at 60 px and more.
- **Per burst its RMSE is 0.005-0.059 dB,** except on the two dev bursts with 60-70 px shifts (0.109 and
  0.132 dB).
- **The original forward method is 6 times less accurate** (0.385 dB). Its worst case is 4.2 dB.

## 8. Final results ([five_bursts_14_frames/](five_bursts_14_frames/), [dev_40_bursts/](dev_40_bursts/), [test_40_bursts/](test_40_bursts/))

Defaults: homography (with fallback), reverse, pair region, target fill, colour reference moved, all 14 frames.
Gains over frame 0 in dB, 95% bootstrap intervals over bursts. The bursts for dev and test are the 40 that
`--limit 40` picks from `splits/dev.txt` and `splits/test.txt`, spread across scenes.

### Test, 40 bursts (4 scenes)

| Frame | Shift (px) | Penalty | Measured | Corrected |
|---|---|---|---|---|
| 1 | 4.5 | +0.05 | +0.00 | +0.05 [−0.01, +0.10] |
| 3 | 14.7 | +0.13 | −0.01 | +0.12 [+0.06, +0.18] |
| 5 | 19.7 | +0.26 | −0.15 | +0.12 [+0.06, +0.17] |
| 7 | 27.2 | +0.45 | −0.36 | +0.09 [+0.03, +0.16] |
| 9 | 37.5 | +0.73 | −0.64 | +0.09 [+0.03, +0.15] |
| 10 | 42.2 | +0.97 | −0.94 | +0.03 [−0.02, +0.08] |
| 13 | 50.9 | +1.29 | −1.28 | +0.01 [−0.05, +0.08] |

- **Frames 3-9 are better references than frame 0 by 0.08-0.12 dB.** Every interval is above zero. Frames 10-13
  are level with frame 0.
- **Forward and reverse now agree:** they differ by at most 0.08 dB, with per-burst correlation 0.43-0.91 (it was
  negative at frames 7-13 before the fixes). The medians are close to the means (+0.05 to +0.10 dB at frames
  1-9).
- **Frame 0 is the best reference in only 1 of 40 test bursts.** The best frame per burst gains +0.18 dB measured
  and +0.28 dB [+0.22, +0.34] corrected. The corrected figure is a maximum over 13 noisy estimates, so it runs
  high.

### Dev, 40 bursts (4 scenes, larger shifts)

| Frame | Shift (px) | Penalty | Measured | Corrected | Median corrected |
|---|---|---|---|---|---|
| 1 | 8.2 | +0.04 | +0.14 | +0.18 [−0.01, +0.36] | +0.01 |
| 3 | 20.0 | +0.31 | −0.10 | +0.21 [−0.03, +0.45] | +0.01 |
| 5 | 35.2 | +1.01 | −0.81 | +0.20 [−0.07, +0.49] | −0.04 |
| 9 | 56.3 | +2.40 | −2.25 | +0.15 [−0.16, +0.48] | −0.12 |
| 13 | 67.8 | +3.02 | −2.81 | +0.21 [−0.07, +0.50] | −0.08 |

- **The penalty is large on dev: up to 3.0 dB at frame 13** (68 px shifts). Almost all of the measured loss of the
  later frames is penalty.
- **The corrected gains are inconclusive.** The means are +0.12 to +0.22 dB with every interval including zero,
  and the medians are near zero or slightly negative. A few bursts with large positive values drive the means.
- **Forward and reverse still disagree at frames 5-13** (by 0.2-0.4 dB). These are the shifts where the benchmark
  measures forward at 0.5-0.7 dB RMS and reverse at 0.09-0.13 dB.

### The 5 bursts

Penalty +1.57 dB at frame 13 (measured −1.65, corrected −0.08 [−0.35, +0.23]). Corrected gains lie between −0.08
and +0.14 dB; frames 4 and 7 have intervals just above zero. Best frame per burst: +0.03 dB measured, +0.20 dB
corrected.

## 9. What is still open

- **The geometry is mostly a translation on real data.** The homography falls back in 35-60% of comparisons. A
  more robust non-rigid model could help on dev's 60-70 px shifts, where the benchmark error is largest
  (0.13 dB RMS).
- **The benchmark changes content only by blur and noise, on 9 bursts.** Real reconstructions differ in more complex
  ways.
- **Dev and test disagree on whether later frames beat frame 0.** Test supports frames 3-9; dev cannot tell. With
  4 scenes per split, scene-to-scene variation is not captured by intervals over bursts.
