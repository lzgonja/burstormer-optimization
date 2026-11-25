# Shift control: does the metric favour frame 0?

Script: `Burst Super-resolution/shift_control.py` (v2, branch `shift-control-v2`). Why the correction can be trusted:
[SHIFT_CONTROL_VALIDATION.md](SHIFT_CONTROL_VALIDATION.md).

Work logs:
- [All three metrics, 2026-11-24](logs/shift_control_all_metrics_2026-11-24/LOG.md): SSIM and LPIPS added to the
  control and the benchmark, with the benchmark and the 5 bursts rerun.
- [v2, 2026-11-22](logs/shift_control_v2_2026-11-22/LOG.md): the refined method, the known-answer benchmark, and
  the runs on 5 bursts and on 40 dev and 40 test bursts, all 14 reference frames.
- [v1, 2026-11-21](logs/shift_control_2026-11-21/LOG.md): frames 0-4 on 40 dev and 40 test bursts.
- [v1, 2026-11-22](logs/shift_control_2026-11-22/LOG.md): 14 frames on 5 bursts.

## What it is

The BurstSR ground truth is lined up with frame 0 of each burst. When Burstormer uses another frame as its
reference, the output is displaced by a few to over a hundred output pixels. To score it, the metric has to align
the output further back onto the ground truth, and that costs PSNR. A later frame can therefore look worse only
because of its position, not because its reconstruction is worse.

The shift control removes that effect:

1. Run the model with every reference frame k.
2. Estimate the geometry between the frame-0 output O_0 and the frame-k output O_k: a homography fitted to
   tile-wise phase correlation. If the homography does not fit its own tiles to within 1 px, the global shift is
   used instead. On real outputs that happens in about half of the comparisons.
3. Move O_k back to frame 0's position and score it as frame 0 would be scored. Three details make this a pure
   change of position:
   - **The move keeps detail and noise level.** It is an exact Fourier shift, plus 8x band-limited resampling of
     the rest of the geometry, without clipping.
   - **The strip the move uncovers is filled with frame 0's output and left out of the score.** The metric fits
     one colour correction per patch over all its pixels, so mirrored content there would spoil the whole patch.
   - **Frame k's raw input frame moves along with its output.** The metric colour-matches an output with its
     reference input frame, so the two must stay together, even when exposure changes between frames.

This is the **reverse** estimate. Frame k's own reconstruction is scored at frame 0's position, so its **corrected
gain** over frame 0 contains no penalty for position. The **penalty** is the corrected gain minus the gain the
standard metric measures.

The original method (v1) did it the other way round: **forward**, moving O_0 to frame k's position and adding its
loss back as the penalty.

A known-answer benchmark compared the two on 9 bursts, including the hardest. The reverse estimate is accurate to
0.068 dB RMS. The forward estimate is off by 0.385 dB RMS, because the penalty depends on the content being moved.
Both are still computed, and the forward one is kept as a diagnostic.

Two probes check the metric itself. Both use O_0, scored as reference 0, with input frame 0 moved along:
- **probe_half_px** moves it by half an output pixel, so it measures resampling only.
- **probe_whole_px** moves it by 8 x 8 output pixels, so it measures displacement only.

Each comparison of frame k with frame 0 uses only the pixels that every image of that comparison covers
(`--region pair`). A frame's result therefore does not depend on which other frames are in the run.

## How to run it

Requirements: the environment from `Burst Super-resolution/Readme.md` (CUDA GPU, CuPy for PWC-Net), and the
following files:

| File | Where to get it | Default path (from `Burst Super-resolution/`) |
|---|---|---|
| BurstSR val set | `python scripts/download_burstsr_dataset.py .` | `./burstsr_dataset/val` |
| Burstormer weights | `burstormer_burstsr_train.pth`, from the [t02-burstsr-v1 release](https://github.com/lzgonja/burstormer-optimization/releases/tag/t02-burstsr-v1) (see `docs/BURSTSR_TRAINING.md`) | `./Trained_models/Real/burstormer_burstsr_train.pth` |
| PWC-Net weights | as for the evaluation scripts (`pwcnet/download.bash`) | `./pwcnet/pwcnet-network-default.pth` |

From `Burst Super-resolution/`:

```
# Unit tests of the shift, warp and geometry operations (CPU, a few seconds)
python -m unittest tests.test_shift_control tests.test_shift_geometry -v

# The 5 bursts, all 14 reference frames
python shift_control.py --bursts_file ../docs/logs/shift_control_v2_2026-11-22/bursts.txt --refs 14 --out_dir ./Results/Real/sc_five

# 40 bursts spread across the scenes of a split
python shift_control.py --bursts_file ./splits/test.txt --limit 40 --refs 14 --out_dir ./Results/Real/sc_test

# Known-answer benchmark of every estimator, on the 9 bursts of the final benchmark
python shift_control_benchmark.py --bursts_file ../docs/logs/shift_control_v2_2026-11-22/benchmark_9_bursts/bursts.txt --out_dir ./Results/Real/sc_benchmark
```

Options of `shift_control.py`:
- `--geometry {translation,affine,homography,dense}`: how outputs are moved. The default is `homography`.
- `--estimator {reverse,forward}`: which estimate is reported as `gain_corrected`. The default is `reverse`.
  Both are always computed.
- `--region {pair,all}`: the pixels scored. The default is `pair`. `all` uses the pixels common to every image of
  the burst, as v1 did.
- `--fill {target,mirror}`: what fills the strip a move uncovers. The default is `target`.
- `--colour_ref {moved,fixed}`: whether the raw reference frame moves along with the output. The default is
  `moved`.
- `--refs N`: test references 0 to N-1 (default 5; use 14 for all).
- `--bursts_file`, `--limit N`: which bursts.
- `--factor`: the upsampling factor of the warp (default 8).
- `--weights`, `--data_root`, `--pwcnet`: file locations.
- The v1 method is `--geometry translation --region all --estimator forward --fill mirror --colour_ref fixed`.
  It reproduces the v1 numbers to within 0.012 dB.

**All three metrics.** Every image is scored with PSNR, SSIM and LPIPS on the same pixels, and the control reports
every gain, estimate and penalty for each one. PSNR keeps the original column names. The SSIM and LPIPS versions of
the derived columns have an `ssim_` or `lpips_` prefix, for example `lpips_gain_corrected`. LPIPS is lower-is-better,
so its gains are reductions (frame 0 minus frame k). For every metric, a positive gain means frame k is better, and a
positive penalty means its position costs it score.

Outputs in `--out_dir`:
- **`per_burst.csv`**: one row per burst and reference frame, plus two probe rows per burst. Columns:
  - `shift_dy`, `shift_dx`: the global shift, in output pixels.
  - `residual_max_px`: the largest part of the geometry beyond that shift.
  - For each metric `m` in `psnr`, `ssim`, `lpips`:
    - `m_frame0`, `m_real`: O_0 and O_k as the standard metric scores them.
    - `m_forward`: O_0 moved to frame k. `m_reverse`: O_k moved back to frame 0.
  - `gain_real` (measured), `gain_forward`, `gain_reverse`, `gain_corrected` (the chosen estimate), `penalty`: PSNR,
    in dB. The same columns with an `ssim_` or `lpips_` prefix hold SSIM and LPIPS.
  - `fidelity_forward`, `fidelity_reverse`: how closely each moved output matches the one it was moved onto, in
    dB. High values mean the outputs differ mainly by the geometry.
  - `geometry_fallback`: how many of the two geometry fits fell back to the global shift.
  - `common_frac`: the share of pixels scored.
- **`summary.csv`** and the printed tables, one set per metric (column `metric`): the mean per frame with 95%
  bootstrap intervals over bursts, the forward-vs-reverse comparison, and the best frame per burst before and after
  correction.

`shift_control_benchmark.py` checks the estimators against known answers for all three metrics in the same way. Its
`per_frame.csv` has `truth`, `roundtrip_loss` and `err_*` columns for PSNR, plus `ssim_` and `lpips_` versions.

**Sanity checks.**
- **Frame 0:** penalty exactly 0.
- **`probe_half_px`:** about 0 (resampling costs nothing).
- **`fidelity_*`:** around 40-50 dB on BurstSR outputs. Much lower values point to a failed geometry estimate.

## Results (v2, 2026-11-22)

All 14 reference frames, default settings. Gains are relative to frame 0, in dB, with 95% bootstrap intervals over
bursts. Full tables: [the log](logs/shift_control_v2_2026-11-22/LOG.md), section 8.

### Test scenes, 40 bursts

| Frame | Shift (output px) | Penalty | Measured gain | Corrected gain |
|---|---|---|---|---|
| 1 | 4.5 | +0.05 | +0.00 | +0.05 [−0.01, +0.10] |
| 3 | 14.7 | +0.13 | −0.01 | +0.12 [+0.06, +0.18] |
| 5 | 19.7 | +0.26 | −0.15 | +0.12 [+0.06, +0.17] |
| 7 | 27.2 | +0.45 | −0.36 | +0.09 [+0.03, +0.16] |
| 9 | 37.5 | +0.73 | −0.64 | +0.09 [+0.03, +0.15] |
| 10 | 42.2 | +0.97 | −0.94 | +0.03 [−0.02, +0.08] |
| 13 | 50.9 | +1.29 | −1.28 | +0.01 [−0.05, +0.08] |

- **The metric charges a penalty for position that grows with the shift, to 1.3 dB at frame 13.** Frames that are
  as good as frame 0 or better show up as measured losses.
- **Corrected, frames 3-9 are better references than frame 0 by 0.08-0.12 dB.** Every interval is above zero.
  Frames 10-13 are level with frame 0.
- **The forward and reverse estimates agree here,** to within 0.08 dB.
- **Frame 0 is the best reference in only 1 of 40 test bursts.** Picking the best frame per burst gains +0.18 dB
  measured and +0.28 dB [+0.22, +0.34] corrected. That figure is a maximum over 13 noisy estimates, so it runs
  high.

### Dev scenes, 40 bursts

- **The shifts are larger** (up to 68 px on average at frame 13), and so is the penalty: up to 3.0 dB.
- **The corrected gains are inconclusive.** The means are +0.12 to +0.22 dB, but every interval includes zero and
  the medians are near zero (+0.04 to −0.12 dB). A few bursts drive the means.

### The 5 bursts

- **The penalty reaches 1.57 dB at frame 13.**
- **Corrected gains lie between −0.08 and +0.14 dB.** Frames 4 and 7 have intervals just above zero.

## SSIM and LPIPS (2026-11-24)

Details: [the log](logs/shift_control_all_metrics_2026-11-24/LOG.md).

- **The default estimator (reverse, homography) is the most accurate for all three metrics.** On the 9-burst
  benchmark its RMSE is 0.068 dB PSNR, 0.00022 SSIM and 0.00098 LPIPS. The forward (v1) method is 6, 30 and 16 times
  less accurate.
- **SSIM penalises position like PSNR,** up to 0.0047 at frame 13 on the 5 bursts. The benchmark error is about 7%
  of that.
- **LPIPS hardly penalises position,** at most about 0.001. Its benchmark error is about 0.0001 without noise and
  0.002 with noise, so the LPIPS correction is small and, for noise-like changes, within the error.
- **Test scenes, 40 bursts: all three metrics agree that frames 3-4 beat frame 0.** Corrected gains: PSNR +0.12 dB,
  SSIM +0.0002, LPIPS +0.0012 to +0.0016, every interval above zero. LPIPS also favours frames 7-13.
- **Dev scenes, 40 bursts: LPIPS says every frame beats frame 0** (+0.0045 to +0.0074, every interval above zero,
  positive even before correction). PSNR is inconclusive, as before, and SSIM is flat. A few bursts drive the means
  for both PSNR and LPIPS.
- **SSIM's position penalty explains almost all of its measured loss.** Corrected, every frame is within ±0.0002 of
  frame 0 on both splits.
- On the 5 bursts alone, corrected SSIM and LPIPS show no frame clearly better than frame 0.

## Results of v1 (frames 0-4, 2026-11-21)

These runs used the forward estimate with a translation. The benchmark shows that estimate is off by 0.39 dB RMS,
so the v1 numbers are superseded by the v2 results above. The v1 claim that frames 3 and 4 beat frame 0 on the
test scenes is confirmed by v2 (+0.12 dB each), for different reasons.

## Limits

- **Corrected numbers are our own evaluation, not the official BurstSR metric.** Published results, including
  Burstormer's 48.82 dB, use the standard metric. Under it, frame 0 keeps its advantage.
- **The correction is accurate to 0.068 dB RMS on the benchmark,** with a bias of −0.011 dB. The error is larger
  at large shifts: 0.13 dB RMS at 60 px and more.
- **The benchmark changes content only by blur and noise, on 9 bursts.** Real reconstructions from different
  reference frames differ in more complex ways.
- **Each split has only 4 scenes.** Intervals over bursts do not capture scene-to-scene variation, which may be why
  dev and test differ.
- **Repeated runs differ by about 0.01 dB.** Optical flow on the GPU is not bit-exact between runs.
