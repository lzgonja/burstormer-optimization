# Algorithm 2: results

Method and parameters: [ALGORITHMS.md](../ALGORITHMS.md#algorithm-2-central-exposure-consistent-reference-selection).
Code: `Burst Super-resolution/reference_selection/algorithm2.py`.

## Current version: central, exposure-consistent selection (2026-11-27)

Raw outputs: [logs/algorithm2_central_2026-11-27/](logs/algorithm2_central_2026-11-27/). Model:
`burstormer_burstsr_train.pth`. Labels: the shift-control runs of 2026-11-24
([logs/shift_control_all_metrics_2026-11-24/](logs/shift_control_all_metrics_2026-11-24/)). They give, for each of
40 dev and 40 test bursts, the shift-corrected PSNR, SSIM and LPIPS gain of every one of the 14 frames as reference
over frame 0. A selector's gain on a burst is the labelled gain of the frame it picks, so every selector is scored
on the same numbers without running the network (`score_ref_selectors.py`).

### Summary

- **Algorithm 2 now beats frame 0 on all three metrics on the test scenes,** with every interval above zero:
  +0.127 dB PSNR, +0.217e-3 SSIM and +1.369e-3 LPIPS. That is 37-46% of what a perfect per-burst choice would gain.
  The first version gained +0.011 dB, because it kept frame 0 in 95% of bursts.
- **The end-to-end pipeline agrees.** `Track_2_evaluation.py` on the same 40 test bursts gives exactly the same
  three numbers ([track2_test_40_per_burst.csv](logs/algorithm2_central_2026-11-27/track2_test_40_per_burst.csv)).
- **It is ahead of GBRS on all three metrics on test** (GBRS: +0.059 dB, +0.019e-3, +0.928e-3), though the paired
  difference is not significant on any single metric.
- **On dev it is positive but less certain:** +0.197 dB PSNR, +0.079e-3 SSIM, +6.80e-3 LPIPS. Only the LPIPS
  interval excludes zero.
- **On clean bursts it is tied with a fixed middle frame.** Most of the gain comes from moving the reference from
  the edge of the burst to the middle. No burst-only feature tested ranks frames within a burst well (next
  section).
- **It still avoids damaged frames.** When the frame it would pick is damaged, it takes a neighbouring healthy frame
  (1.4 frames away on average).

### Clean bursts

Gain over frame 0, 95% bootstrap over bursts. SSIM and LPIPS are in units of 1e-3. A positive LPIPS gain means a
lower (better) LPIPS.

| Selector | Test PSNR (dB) | Test SSIM | Test LPIPS | Dev PSNR (dB) | Dev SSIM | Dev LPIPS |
|---|---|---|---|---|---|---|
| Frame 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Algorithm 1 (GBRS) | +0.059 [+0.005, +0.118] | +0.019 [-0.091, +0.149] | +0.928 [+0.278, +1.652] | +0.172 [-0.071, +0.454] | -0.088 [-0.794, +0.635] | +5.911 [+2.796, +9.644] |
| Algorithm 2, first version | +0.011 [+0.000, +0.031] | +0.039 [+0.000, +0.116] | +0.280 [+0.000, +0.813] | -0.001 [-0.005, +0.002] | +0.006 [+0.000, +0.015] | +0.024 [-0.002, +0.071] |
| **Algorithm 2** | **+0.127 [+0.056, +0.204]** | **+0.217 [+0.020, +0.425]** | **+1.369 [+0.525, +2.269]** | **+0.197 [-0.078, +0.492]** | **+0.079 [-0.561, +0.762]** | **+6.799 [+3.440, +10.767]** |
| Fixed frame 3 | +0.116 | +0.218 | +1.182 | +0.211 | +0.031 | +6.031 |
| Fixed frame 5 | +0.116 | +0.101 | +0.592 | +0.204 | +0.036 | +7.401 |
| Oracle (uses ground truth) | +0.276 | +0.585 | +3.145 | +0.609 | +1.095 | +8.655 |

Frames picked: frames 5-8 in 77 of 80 bursts (test: 6 and 7 in 32 of 40; dev: 7 in 22 of 40).

### What predicts the best frame

Per-frame features from the burst alone, inspired by the papers behind algorithms 1 and 2. Mean within-burst rank
correlation with the corrected gain, over 40 bursts
([feature_analysis.txt](logs/algorithm2_central_2026-11-27/feature_analysis.txt)):

| Feature | Source | Dev PSNR | Dev SSIM | Dev LPIPS | Test PSNR | Test SSIM | Test LPIPS |
|---|---|---|---|---|---|---|---|
| Mean motion to the other frames | centrality | +0.00 | -0.05 | -0.21 | -0.18 | -0.11 | -0.02 |
| Gradient magnitude | HDR+ sharpness | -0.01 | -0.06 | -0.00 | -0.13 | -0.06 | +0.17 |
| Q score | Zhu & Milanfar (algorithm 1) | -0.09 | -0.08 | +0.17 | -0.01 | +0.01 | +0.03 |
| Consistency with all frames | algorithm 2, first version | +0.05 | +0.04 | -0.16 | -0.02 | -0.05 | +0.07 |
| Brightness deviation | Buades et al. | -0.15 | -0.17 | -0.23 | -0.16 | -0.08 | -0.01 |
| Noise estimate | | +0.05 | +0.00 | +0.04 | -0.04 | +0.01 | +0.13 |

None is above 0.25 in size, and most change sign between metrics or splits. Image quality as these scores measure
it does not tell which frame Burstormer reconstructs best. Brightness deviation is the most consistent (negative in
all six columns), which is why it is in the rule.

Averaged over bursts, the picture is clearer ([features_by_frame.txt](logs/algorithm2_central_2026-11-27/features_by_frame.txt)):
- **Frame 0 is at the edge of the burst.** On dev it is on average 5.4 px from the other frames, against
  2.6-2.8 px for frames 5-8 (test: 3.7 against 2.0-2.2).
- **Its brightness differs most from the burst median.** The deviation is 2.5-2.7%, against 1% or less for frames 6-8.
- **Every other frame beats it on average.** On test, frames 3-9 gain +0.08 to +0.12 dB.

### How the rule was chosen

On dev only. Candidates: fixed frames, the most central frame within a window of frames, and
`z(centrality) + w * z(brightness) + m * z(consistency)` for w = 0-2 and m = 0-0.5. The rules were ranked by the
fraction of the oracle captured, averaged over the three metrics.
- **The top dev rule, "always frame 2", does not hold up:** 43% of the oracle on dev but 11% on test. The leading
  rules are all within noise of each other on dev, so fixed frames were set aside. They also cannot avoid a damaged
  frame.
- **Among the adaptive rules, centrality plus brightness with w = 0.25 or 0.5 was best on dev** (39% of the
  oracle). Adding the consistency term did not help.
- **A synthetic test showed that standardising the brightness can amplify sub-percent noise.** The brightness
  spread is therefore floored at 1%. Floors from 0.5% to 2% are tied on dev.
- **The health gate (30%) changes no pick on the clean bursts.**

### Damaged frames

No network: on each of the 80 bursts, the frame the rule picks on the clean burst is damaged with
`robustness_sweep.degrade`, and the rule is run again
([damage_selection.txt](logs/algorithm2_central_2026-11-27/damage_selection.txt)).

| Damage | Damaged frame avoided | Damage | Damaged frame avoided |
|---|---|---|---|
| Blur 0.5 | 3/80 | Noise 0.001 | 55/80 |
| Blur 1 | 71/80 | Noise 0.003 | 78/80 |
| Blur 2 | 79/80 | Noise 0.01, 0.03 | 80/80 |
| Under x0.5, x0.25, x0.125 | 80/80 | Over x2, x4, x8 | 80/80 |

When it avoids the damaged frame, it moves 1.4 frames on average, to a neighbour that is about as central. Blur 0.5
costs frame 0 only 0.3 dB (first-version table below), and the first version did not avoid it either.

### Limits

- **The labels are estimates.** The shift control's reverse estimate has an RMSE of 0.03-0.13 dB
  (`docs/SHIFT_CONTROL.md`), about the size of the differences between neighbouring frames. Part of the oracle's
  headroom is selection noise.
- **40 bursts per split.** On dev only LPIPS is significant.
- **The damage test checks the selection only, not the reconstruction.** The neighbour it moves to is about as good
  on clean bursts, so the reconstruction should be too, but this was not run through the network.

### Commands

From `Burst Super-resolution/`:

```
python -m pytest tests/test_algorithm2.py
python score_ref_selectors.py --labels ../docs/logs/shift_control_all_metrics_2026-11-24/test_40_bursts/per_burst.csv
python score_ref_selectors.py --labels ../docs/logs/shift_control_all_metrics_2026-11-24/dev_40_bursts/per_burst.csv
```

## First version: exposure-aware consistency selection (2026-11-21)

Raw summaries: [logs/algorithm2_2026-11-21/](logs/algorithm2_2026-11-21/). Now `select_reference_frame_v1`.

Model: `burstormer_burstsr_train.pth` (T02, step 10,000). Metric: aligned PSNR on BurstSR val crops. The selector
takes about 18 ms per burst on the CPU.

## Summary

- **Clean bursts: no gain and no loss.** The selector keeps frame 0 in 92-95% of bursts. Its result is within
  0.01 dB of frame 0 on dev and test. A perfect per-burst choice would add 0.13 dB on the test scenes as
  measured, but the cost cannot tell which early frame that is.
- **The metric favours frame 0 (shift control, below).** Once the metric's penalty for a displaced reference is
  removed, frames 3 and 4 beat frame 0 on the test scenes by 0.07-0.12 dB. The per-burst ceiling rises to
  +0.20-0.23 dB. Validation: `docs/SHIFT_CONTROL_VALIDATION.md`.
- **Damaged reference: almost all of the loss is recovered.** With only frame 0 damaged, fixed frame 0 loses
  1.5 to 22 dB. Algorithm 2 stays within 0.25 dB of the clean result in every case tested except blur sigma 1
  on dev, where it is 0.8 dB below. It uses frame 1 instead.
- **Brightness: the mismatch matters, not the exposure.** A reference twice as bright as the rest of the burst
  costs 4-5 dB, even with no clipping and no extra noise. The well-exposedness weight of Buades et al. (2022)
  points the wrong way here: it rewards the brighter frame. It was replaced by a brightness-mismatch term, since
  merged into the main cost with no change in any pick (see "Simplification: one cost term").

## Background: how much brightness varies in BurstSR

One val burst per scene (23 bursts), all 14 frames:
- ISO is constant in every burst, and exposure time varies in only 5 of 23.
- The median within-burst brightness spread is 4%. The largest spreads (20-25%) come from content moving in
  the crop.
- Clipping appears in one burst only (8.8% of pixels).
- Scenes are dark: mean level 0.02-0.24 of the raw range.

Brightness alone therefore cannot separate frames on real BurstSR bursts. It matters only when a frame is
wrongly exposed, which the damage tests below simulate.

## Clean bursts

40 dev and 40 test bursts spread over scenes, references 0-4 (`reference_frame_sweep.py`), scored with
`score_ref_policies.py`. PSNR_common scores every reference on the same pixels. Gain is paired with frame 0,
95% bootstrap over bursts.

| Policy | Dev gain (common) | Dev switched | Test gain (common) | Test switched |
|---|---|---|---|---|
| Frame 2 | +0.020 [-0.187, +0.222] | 100% | -0.060 [-0.134, +0.014] | 100% |
| Guarded (sharpness, `ref_selection.py`) | +0.020 [-0.187, +0.222] | 100% | -0.069 [-0.146, +0.007] | 100% |
| **Algorithm 2 (margin 0.2)** | **-0.002 [-0.009, +0.002]** | 8% | **+0.005 [-0.002, +0.017]** | 5% |
| Algorithm 2, lowest cost always | -0.034 [-0.250, +0.158] | 82% | -0.025 [-0.087, +0.033] | 85% |
| Oracle (best frame, uses ground truth) | +0.329 [+0.188, +0.487] | 65% | +0.131 [+0.079, +0.195] | 68% |

The margin was chosen on dev: 0.2 is the smallest margin with no loss there. At 0.1 the dev gain was +0.05
(interval touching zero), but the test gain was -0.004, so it was not used.

## Damaged reference

`robustness_sweep.py --degrade_frames 0`: only frame 0 is damaged. Guarded keeps frame 2 by default, so for it
frame 2 is damaged instead. 12 bursts per split. Mean PSNR in dB.

Damage types:
- **blur:** Gaussian blur; the level is sigma in packed raw pixels.
- **noise:** shot plus read noise at the given shot-noise level.
- **under:** the frame got k times the light, with the matching extra shot noise from the ISO 50 noise profile.
- **over:** k times brighter, clipped at white.

Both exposure damages are quantised to 10 bits. For exposure, the metric colour-matches with the clean burst
scaled by k, so only the noise and clipping count, not the brightness change itself.

### Dev scenes

| Damage on the default frame | Fixed frame 0 | Guarded (frame 2 damaged) | Algorithm 2, first version | **Algorithm 2** | Algorithm 2 reference used |
|---|---|---|---|---|---|
| None | 46.54 | 46.48 | 46.55 | **46.54** | 0 in 12/12 |
| Blur 0.5 | 46.25 | 46.40 | 46.25 | **46.25** | 0 in 12/12 |
| Blur 1 | 43.38 | 46.52 | 44.62 | **45.71** | 1 in 9/12 |
| Blur 2 | 38.05 | 46.60 | 45.22 | **46.66** | 1 in 12/12 |
| Noise 0.003 | 45.13 | 45.82 | 45.13 | **46.61** | 1 in 11/12 |
| Noise 0.01 | 42.27 | 45.86 | 45.19 | **46.67** | 1 in 12/12 |
| Noise 0.03 | 38.19 | 46.11 | 45.12 | **46.71** | 1 in 12/12 |
| Noise 0.1 | 24.30 | 46.48 | 45.12 | **46.52** | 1 in 12/12 |
| Under x0.5 | 42.11 | 46.58 | 44.77 | **46.61** | 1 in 12/12 |
| Under x0.25 | 39.98 | 46.52 | 45.16 | **46.52** | 1 in 12/12 |
| Under x0.125 | 38.10 | 46.53 | 45.15 | **46.51** | 1 in 12/12 |
| Under x0.0625 | 35.93 | 46.56 | 45.13 | **46.47** | 1 in 12/12 |
| Over x2 | 42.50 | 46.57 | 42.59 | **46.79** | 1 in 12/12 |
| Over x4 | 38.73 | 46.59 | 42.28 | **46.54** | 1 in 12/12 |
| Over x8 | 34.39 | 46.41 | 42.97 | **46.49** | 1 in 12/12 |
| Over x16 | 31.56 | 46.43 | 43.02 | **46.40** | 1 in 12/12 |

### Test scenes (confirmation)

| Damage on frame 0 | Fixed frame 0 | **Algorithm 2** |
|---|---|---|
| None | 50.07 | **50.08** (switched in 1/12) |
| Blur 1 | 48.62 | **50.08** |
| Blur 2 | 43.45 | **50.08** |
| Noise 0.01 | 45.44 | **50.05** |
| Noise 0.03 | 39.39 | **50.03** |
| Under x0.5 | 43.91 | **50.16** |
| Under x0.25 | 40.36 | **50.15** |
| Over x2 | 45.05 | **50.04** |
| Over x4 | 44.07 | **49.84** |

### What changed from the first version

The first version (dev column "first version") was `cost = consistency / well-exposedness`. Each candidate was
compared with all 13 other frames, and the cheapest frame was taken after rejecting the default. It had three
faults:

1. **It favoured frames near the middle of the burst.** Pairwise differences after alignment grow with the time
   gap (about +40% from 1 to 12 frames apart). Comparing with all other frames therefore made frame 4 look 10-14%
   better than frame 0 on clean bursts. Fix: every candidate is compared with the 8 frames that follow it.
2. **It picked late fallbacks.** After rejecting frame 0, the cheapest frame was often frame 3 or 4, about 1 dB
   worse references than frame 1. Recovery stopped at about 45.1 dB. Fix: take the earliest acceptable frame.
3. **It missed over-exposure.** The well-exposedness weight rewards brighter frames in these dark scenes, and the
   brightness normalisation hides a gain difference. So a 2x reference was kept in 11 of 12 bursts. Fix: replace
   the weight by the brightness mismatch with the following frames, measured on the pixels they share.

### Simplification: one cost term (2026-11-23)

The version scored above used two terms, `residual * (1 + mismatch)`: each frame divided by its own brightness,
then a separate brightness mismatch `|median log(brightness of c / brightness of j)|`. It was merged into one term:
divide the candidate and its 8 comparison frames by the same typical brightness (their median), so a brightness
difference stays in the residual.

Both versions were run on 440 cases: 40 clean dev and 40 clean test bursts, and 12 dev and 12 test bursts with
frame 0 damaged at every level in the tables above (blur, noise, under- and over-exposure). They picked the same
reference in all 440, so every PSNR in this document holds for the current version unchanged. Healthy frames cost
0.06 (median; at most 0.18), and a frame 2x too bright or too dark adds about 1 or 0.5.

Two other simplifications were tested the same way and rejected:
- **Whole-pixel alignment instead of sub-pixel:** 22 of 440 picks changed. It switched away from frame 0 on 6
  clean bursts and kept a blurred frame 0 on test.
- **Aligning every frame to frame 0 once, instead of to each candidate:** 29 of 440 picks changed, and the test
  that the fallback is the earliest acceptable frame failed.

## Compared with the guarded selector (T16)

- **Algorithm 2 also catches moderate noise.** Guarded relies on sharpness, which noise inflates. It kept the
  noisy reference in 11 of 12 bursts at noise 0.003 and in 6 of 12 at 0.01 (45.8-45.9 dB). Algorithm 2 replaced
  it in 11 and 12 of 12 (46.6-46.7 dB).
- **Guarded is better on moderate blur (sigma 1): 46.52 against 45.71 on dev.** Algorithm 2 kept the blurred
  frame in 3 of 12 bursts. Light blur also lowers the noise, which partly offsets the extra residual. On the test
  scenes it switched in all 12.
- **Guarded defaults to frame 2.** That is 0.07 dB below frame 0 on clean test bursts. Algorithm 2 keeps frame 0.

## Shift control: the metric favours frame 0

The ground truth is registered to frame 0. Every other reference produces an output that is displaced, which the
metric must warp further. `shift_control.py` (also on the `shift-control` branch, guide in `docs/SHIFT_CONTROL.md`) measures what that costs on its own. Raw results: [logs/shift_control_2026-11-21/](logs/shift_control_2026-11-21/).
1. Take the frame-0 output.
2. Move it by the measured displacement of the frame-k output, with an exact Fourier shift and mirror padding.
3. Score it exactly as a frame-k output would be scored.

The content is identical to the frame-0 output, so any PSNR drop is the metric's penalty for position. The
corrected gain of frame k is its measured gain plus that penalty. 40 dev and 40 test bursts (the same as the
clean sweep), scored on the pixels common to every image (62-63%). Gain vs frame 0, 95% bootstrap over bursts.

| Frame | Shift (output px), dev / test | Penalty, dev | Penalty, test | Measured gain, test | **Corrected gain, test** | Corrected gain, dev |
|---|---|---|---|---|---|---|
| 1 | 8 / 5 | +0.03 [-0.02, +0.10] | +0.06 [+0.02, +0.11] | +0.00 | **+0.06 [+0.01, +0.12]** | +0.18 [-0.02, +0.38] |
| 2 | 16 / 11 | +0.17 [+0.07, +0.29] | +0.13 [+0.07, +0.19] | -0.06 | **+0.07 [+0.01, +0.13]** | +0.19 [-0.02, +0.42] |
| 3 | 20 / 15 | +0.22 [+0.12, +0.33] | +0.13 [+0.06, +0.20] | -0.01 | **+0.12 [+0.04, +0.20]** | +0.13 [-0.11, +0.35] |
| 4 | 27 / 16 | +0.44 [+0.27, +0.62] | +0.17 [+0.10, +0.25] | -0.08 | **+0.10 [+0.05, +0.14]** | +0.03 [-0.19, +0.26] |

- **The penalty grows with the displacement:** up to 0.44 dB for frame 4 on dev.
- **Bilinear resampling is not the cause.** A half-pixel shift costs nothing (-0.01 and +0.00 dB). A whole
  11-pixel shift, with the input frame moved along, costs 0.09-0.14 dB. The penalty comes from larger
  displacements (optical-flow error, patch content near the edges).
- **"Earlier frames are better" was mostly a metric artifact.** Corrected, every one of frames 1-4 beats frame 0
  on the test scenes with this estimate. A second, independent estimate (the reverse check in
  `docs/SHIFT_CONTROL_VALIDATION.md`) is 0.01-0.05 dB lower. Under both, frames 3 and 4 still beat frame 0
  (+0.07 to +0.12 dB), while frames 1 and 2 are not clearly better. After correction, frame 0 is the best frame in
  only 3-4 of 40 test bursts (12 of 40 on dev).
- **The headroom is larger than measured.** A perfect per-burst choice gains +0.20 to +0.23 dB on test, against
  +0.13 measured, and about +0.43 dB on dev, against +0.33. These per-burst maxima include some selection noise.
- **The shifted copy is close to the real frame-k output:** 41-45 dB between them. Outputs from different
  references differ mostly by a translation, so the correction is meaningful.

Algorithm 2 with no default (lowest cost among frames 0-4), scored with corrected gains:

| | Dev | Test |
|---|---|---|
| Algorithm 2, lowest cost | +0.15 [-0.05, +0.36] | +0.12 [+0.07, +0.18] |
| Random frame among 0-4 | +0.11 | +0.07 |
| Picks the best frame | 20% | 25% (random: 20%) |
| Within-burst rank correlation with corrected PSNR | +0.09 | +0.04 |

Without a default, Algorithm 2 gains +0.12 dB on test. Most of that comes from not using frame 0: a random frame
gets +0.07. The cost barely ranks healthy frames.

## Limits

- **The damage is synthetic and hits one frame.** Real bursts with a shaky or badly exposed frame have not been
  tested. Two or more damaged early frames would weaken the median comparison.
- **Under-exposure uses the ISO 50 noise profile.** Three val scenes are at ISO 125-160 and have more noise.
- **Samples are small:** 40 bursts for clean, 12 for damage. The margin was chosen on dev scenes only.
- **The model is not changed.** On clean bursts the selector cannot gain anything; it only protects against a bad
  reference.

## Commands

From `Burst Super-resolution/`:

```
python -m unittest tests.test_algorithm2 -v
W=./Trained_models/Real/burstormer_burstsr_train.pth
python reference_frame_sweep.py --weights $W --refs 0 1 2 3 4 --bursts_file ./splits/dev.txt --limit 40 \
    --out_dir ./Results/Real/alg2_sweep_dev            # same with ./splits/test.txt
python score_ref_policies.py --sweep_dir ./Results/Real/alg2_sweep_dev
python shift_control.py --weights $W --bursts_file ./splits/dev.txt --limit 40 --out_dir ./Results/Real/shift_control_dev
python robustness_sweep.py --weights $W --degradation exposure --levels 1 0.5 0.25 0.125 0.0625 \
    --degrade_frames 0 --ref_policy algorithm2 --bursts_file ./splits/dev.txt --limit 12 \
    --out_dir ./Results/Real/alg2_damage/under_algorithm2_f0
# blur: --degradation blur --levels 0 0.5 1 2; noise: --degradation noise --levels 0 0.003 0.01 0.03 0.1
# over-exposure: --degradation exposure --levels 1 2 4 8 16; baselines: --ref_policy frame0, or guarded with
# --degrade_frames 2
```
