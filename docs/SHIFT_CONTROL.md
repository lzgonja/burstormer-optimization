# Shift control: does the metric favour frame 0?

Script: `Burst Super-resolution/shift_control.py` (v2, branch `shift-control-v2`). Why the correction can be trusted:
[SHIFT_CONTROL_VALIDATION.md](SHIFT_CONTROL_VALIDATION.md).

Work logs:
- [v2, 2026-11-22](logs/shift_control_v2_2026-11-22/LOG.md): the refined method, the known-answer benchmark and
  the 14-frame run on 5 bursts.
- [v1, 2026-11-21](logs/shift_control_2026-11-21/LOG.md): frames 0-4 on 40 dev and 40 test bursts.
- [v1, 2026-11-22](logs/shift_control_2026-11-22/LOG.md): 14 frames on 5 bursts.

## What it is

The BurstSR ground truth is lined up with frame 0 of each burst. When Burstormer uses another frame as its
reference, the output is displaced by a few to over a hundred output pixels. To score it, the metric has to align
the output further back onto the ground truth, and that costs PSNR. A later frame can therefore look worse only
because of its position, not because its reconstruction is worse.

The shift control removes that effect:

1. Run the model with every reference frame k.
2. Estimate the geometry between the frame-0 output O_0 and the frame-k output O_k. The default is a homography
   fitted to tile-wise phase correlation. It covers the shift, and also the rotation, scale and perspective of
   handheld camera motion.
3. Move O_k back to frame 0's position and score it as frame 0 would be scored.

This is the **reverse** estimate. Frame k's own reconstruction is scored at frame 0's position, so its **corrected
gain** over frame 0 contains no penalty for position. The **penalty** is the corrected gain minus the gain the
standard metric measures.

The images are moved with an exact Fourier shift. Any rotation, scale or perspective left over is applied by
sampling an 8x band-limited upsampled copy. The move keeps detail and noise level, and the result is not clipped
(see the validation document for why both matter).

The original method (v1) did it the other way round: **forward**, moving O_0 to frame k's position and adding its
loss back as the penalty. A known-answer benchmark shows that the forward estimate is biased whenever frame k's
content differs from frame 0's (RMSE 0.28 dB), while the reverse estimate is accurate to 0.03 dB RMS. Both are
still computed, and the forward one is kept as a diagnostic.

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

# The 5 bursts, all 14 reference frames (a few minutes on an RTX 4070 Super)
python shift_control.py --bursts_file ../docs/logs/shift_control_v2_2026-11-22/bursts.txt --refs 14 --out_dir ./Results/Real/sc_five

# 40 bursts spread across the scenes of a split
python shift_control.py --bursts_file ./splits/test.txt --limit 40 --refs 14 --out_dir ./Results/Real/sc_test

# Known-answer benchmark of every estimator (several times longer: 4 content changes x 8 estimates per frame)
python shift_control_benchmark.py --bursts_file ../docs/logs/shift_control_v2_2026-11-22/bursts.txt --out_dir ./Results/Real/sc_benchmark
```

Options of `shift_control.py`:
- `--geometry {translation,affine,homography,dense}`: how outputs are moved. The default is `homography`.
- `--estimator {reverse,forward}`: which estimate is reported as `gain_corrected`. The default is `reverse`.
  Both are always computed.
- `--region {pair,all}`: the pixels scored. The default is `pair`. `all` uses the pixels common to every image of
  the burst, as v1 did.
- `--refs N`: test references 0 to N-1 (default 5; use 14 for all).
- `--bursts_file`, `--limit N`: which bursts.
- `--factor`: the upsampling factor of the warp (default 8).
- `--weights`, `--data_root`, `--pwcnet`: file locations.
- The v1 method is `--geometry translation --region all --estimator forward`.

Outputs in `--out_dir`:
- **`per_burst.csv`**: one row per burst and reference frame, plus two probe rows per burst. Columns:
  - `shift_dy`, `shift_dx`: the global shift, in output pixels.
  - `residual_max_px`: the largest part of the geometry beyond that shift.
  - `psnr_frame0`, `psnr_real`: O_0 and O_k as the standard metric scores them.
  - `psnr_forward`: O_0 moved to frame k. `psnr_reverse`: O_k moved back to frame 0.
  - `gain_real` (measured), `gain_forward`, `gain_reverse`, `gain_corrected` (the chosen estimate), `penalty`.
  - `fidelity_forward`, `fidelity_reverse`: how closely each moved output matches the one it was moved onto, in
    dB. High values mean the outputs differ mainly by the geometry.
  - `common_frac`: the share of pixels scored.
- **`summary.csv`** and the printed tables: the mean per frame with 95% bootstrap intervals over bursts, the
  forward-vs-reverse comparison, and the best frame per burst before and after correction.

**Sanity checks.**
- **Frame 0:** penalty exactly 0.
- **`probe_half_px`:** about 0 (resampling costs nothing).
- **`fidelity_*`:** around 43-50 dB on BurstSR outputs. Much lower values point to a failed geometry estimate.

## Results

### v2: 5 bursts, all 14 reference frames (2026-11-22)

Bursts 0008_0024, 0010_0027, 0019_0020, 0033_0001 and 0036_0009. Homography geometry, reverse estimate, pair
region (61% of pixels). Gains are relative to frame 0, in dB, with 95% bootstrap intervals over the 5 bursts.

| Frame | Shift (output px) | Penalty | Measured gain | Corrected gain |
|---|---|---|---|---|
| 1 | 2 | +0.03 | −0.02 | +0.01 [−0.06, +0.10] |
| 4 | 9 | +0.18 | −0.10 | +0.08 [−0.02, +0.18] |
| 7 | 17 | +0.32 | −0.27 | +0.06 [−0.11, +0.23] |
| 10 | 29 | +1.04 | −1.02 | +0.01 [−0.14, +0.23] |
| 13 | 40 | +1.55 | −1.63 | −0.08 [−0.40, +0.23] |

Full table: [logs/shift_control_v2_2026-11-22/five_bursts_14_frames/summary.txt](logs/shift_control_v2_2026-11-22/five_bursts_14_frames/summary.txt).

- **The metric's penalty for position grows with the shift, reaching 1.55 dB at frame 13 (40 px on average).**
  Almost all of the measured loss of the later frames is this penalty.
- **Corrected, no frame beats frame 0.** Every interval includes zero, and the mean corrected gains stay between
  −0.08 and +0.08 dB.
- **Burst 0019_0020 drives most of the penalty.** At frame 13 its camera has moved 110 output px.
- **The best frame per burst gains +0.03 dB measured and +0.18 dB [+0.06, +0.33] corrected.** Part of the corrected
  figure is selection noise: it is the maximum of 13 noisy estimates.

### v1: frames 0-4, 40 dev and 40 test bursts (2026-11-21)

These runs used the forward estimate with a translation, which the benchmark shows runs high (+0.09 dB bias
overall). Their reverse column is the better estimate. They have not yet been rerun with v2.

| Frame | Shift (output px), dev / test | Measured gain, test | Forward (v1), test | Reverse, test |
|---|---|---|---|---|
| 1 | 8 / 5 | +0.00 | +0.06 [+0.01, +0.12] | +0.04 [−0.02, +0.08] |
| 2 | 16 / 11 | −0.06 | +0.07 [+0.01, +0.13] | +0.03 [−0.03, +0.08] |
| 3 | 20 / 15 | −0.01 | +0.12 [+0.04, +0.20] | +0.07 [+0.00, +0.14] |
| 4 | 27 / 16 | −0.08 | +0.10 [+0.05, +0.14] | +0.08 [+0.03, +0.13] |

On the test scenes the reverse estimate puts frames 3 and 4 slightly ahead of frame 0, by +0.07 to +0.08 dB, with
intervals at or above zero. That run used the translation-only geometry and the clipped warp of v1.

## Limits

- **Corrected numbers are our own evaluation, not the official BurstSR metric.** Published results, including
  Burstormer's 48.82 dB, use the standard metric. Under it, frame 0 keeps its advantage.
- **The benchmark changes content only by blur and noise.** Real reconstructions from different reference frames
  differ in more complex ways. The reverse estimate makes no assumption about content, but a structural difference
  could disturb the geometry estimate. The fidelity columns flag such cases.
- **Translation alone is nearly as accurate as a homography on these bursts** (benchmark RMSE 0.029 vs 0.030 dB).
  The homography is the default for scenes with more rotation, at no cost in accuracy.
- **The samples are small:** 5 bursts for v2. Intervals are wide.
- **Repeated runs differ by about 0.01 dB.** Optical flow on the GPU is not bit-exact between runs.
