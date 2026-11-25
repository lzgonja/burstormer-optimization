# Shift control for all three metrics, 2026-11-24

Branch `shift-control-all-metrics`, off main at 81878c7. RTX 4070 Super.

Before this change, the shift control and its benchmark scored only PSNR, although `FastAlignedMetrics.score`
already computed SSIM and LPIPS on the same pixels. The project reports all three metrics, so the control now
reports each one.

## 1. What changed

- **`shift_control.py`.** Every score, gain, estimate and penalty is reported for PSNR, SSIM and LPIPS, and the
  summary prints one set of tables per metric. The PSNR columns keep their names. The SSIM and LPIPS columns carry
  an `ssim_` or `lpips_` prefix, and `summary.csv` has a `metric` column.
- **Sign convention.** LPIPS is lower-is-better, so its gains are reductions (frame 0 minus frame k). For every
  metric, a positive gain means frame k is better and a positive penalty means its position costs it score.
- **`shift_control_benchmark.py`.** The known answer, the warp round-trip loss and the error of every estimator are
  computed for all three metrics.
- **The method itself is unchanged.** The geometry, warp, fill, colour reference and scored region are the same for
  every metric.

## 2. Checks

- The 29 unit tests pass (`tests.test_shift_control`, `tests.test_shift_geometry`).
- **PSNR is unchanged.** On the 5 bursts, every PSNR column matches the v2 run
  ([../shift_control_v2_2026-11-22/five_bursts_14_frames/](../shift_control_v2_2026-11-22/five_bursts_14_frames/))
  exactly, in all 80 rows. The benchmark's PSNR numbers match the v2 benchmark: reverse/homography RMSE 0.068 dB,
  bias −0.011 dB.

## 3. Benchmark, 9 bursts ([benchmark_9_bursts/](benchmark_9_bursts/))

The same 9 bursts and 468 cases as the v2 benchmark. Errors are the estimate minus the known answer, in each
metric's own units.

```
python shift_control_benchmark.py --bursts_file ../docs/logs/shift_control_v2_2026-11-22/benchmark_9_bursts/bursts.txt --out_dir ./Results/Real/sc_benchmark_all_metrics
```

| Metric | Estimator | Bias | RMSE | Worst error | RMSE 0-8 px | 8-24 px | 24-60 px | 60+ px |
|---|---|---|---|---|---|---|---|---|
| PSNR (dB) | Forward, translation (v1) | +0.038 | 0.385 | 4.16 | 0.132 | 0.161 | 0.543 | 0.708 |
| PSNR (dB) | **Reverse, homography (default)** | **−0.011** | **0.068** | 0.38 | 0.033 | 0.045 | 0.085 | 0.126 |
| SSIM | Forward, translation (v1) | −0.0012 | 0.0067 | 0.068 | 0.0010 | 0.0010 | 0.0079 | 0.0165 |
| SSIM | **Reverse, homography (default)** | **−0.00003** | **0.00022** | 0.0022 | 0.00011 | 0.00010 | 0.00031 | 0.00038 |
| LPIPS | Forward, translation (v1) | −0.0033 | 0.0158 | 0.148 | 0.0039 | 0.0035 | 0.0211 | 0.0338 |
| LPIPS | **Reverse, homography (default)** | **−0.00007** | **0.00098** | 0.011 | 0.00035 | 0.00030 | 0.00155 | 0.00142 |

RMSE of the default (reverse, homography) by content change:

| Metric | None | Blur 0.5 | Blur 1.0 | Noise 0.01 |
|---|---|---|---|---|
| PSNR (dB) | 0.063 | 0.063 | 0.052 | 0.088 |
| SSIM | 0.00008 | 0.00008 | 0.00006 | 0.00042 |
| LPIPS | 0.00011 | 0.00012 | 0.00008 | 0.00196 |

- **The default (reverse, homography) is the most accurate estimator for all three metrics.** Translation and
  affine are about as good. Dense has a lower RMSE for PSNR but larger worst-case errors for every metric.
- **The forward (v1) method is 6 times less accurate for PSNR, 30 times for SSIM and 16 times for LPIPS.**
- **The expected edge effects are small.** SSIM's 11×11 window reaches past the scored region, and
  LPIPS sees a zeroed border. Even so, without a content change the default's error is only 0.00008 SSIM and
  0.00011 LPIPS. The warp round trip loses at most 0.0001 SSIM and 0.00015 LPIPS.
- **Most of the LPIPS error comes from the noise case.** Without noise its RMSE is about 0.0001. With noise it is
  0.002, against a known answer of 0.039 RMS, so about 5% of the effect.

How large the error is next to the effect being corrected (5 bursts, frame 13; reverse-homography RMSE at 24-60 px):

| Metric | Penalty at frame 13 | Benchmark RMSE | Ratio |
|---|---|---|---|
| PSNR | 1.57 dB | 0.085 dB | 5% |
| SSIM | 0.0047 | 0.00031 | 7% |
| LPIPS | 0.0009 | 0.00155 (about 0.0001 over all shifts without the noise case) | over 100% (about 15% without noise) |

For SSIM the correction is about as reliable as for PSNR. For LPIPS the position penalty is so small that the
correction hardly matters, and the benchmark cannot resolve it when the content change is noise-like.

## 4. The 5 bursts, 14 frames ([five_bursts_14_frames/](five_bursts_14_frames/))

```
python shift_control.py --bursts_file ../docs/logs/shift_control_v2_2026-11-22/bursts.txt --refs 14 --out_dir ./Results/Real/sc_five_all_metrics
```

Gains over frame 0, positive = frame k better, 95% bootstrap intervals over bursts. PSNR is as in the v2 log.

| Frame | Shift (px) | SSIM penalty | SSIM measured | SSIM corrected | LPIPS penalty | LPIPS measured | LPIPS corrected |
|---|---|---|---|---|---|---|---|
| 1 | 2.3 | +0.0001 | −0.0001 | +0.0000 [−0.0000, +0.0001] | +0.0001 | −0.0004 | −0.0003 [−0.0006, +0.0000] |
| 4 | 8.7 | +0.0004 | −0.0006 | −0.0002 [−0.0009, +0.0002] | +0.0003 | +0.0002 | +0.0005 [+0.0000, +0.0010] |
| 7 | 17.2 | +0.0008 | −0.0009 | −0.0001 [−0.0007, +0.0003] | +0.0004 | +0.0002 | +0.0006 [−0.0004, +0.0020] |
| 9 | 22.7 | +0.0011 | −0.0016 | −0.0006 [−0.0013, −0.0000] | +0.0006 | −0.0014 | −0.0009 [−0.0024, +0.0001] |
| 11 | 32.4 | +0.0030 | −0.0038 | −0.0008 [−0.0022, +0.0000] | +0.0011 | −0.0010 | +0.0001 [−0.0013, +0.0013] |
| 13 | 39.9 | +0.0047 | −0.0056 | −0.0008 [−0.0022, +0.0001] | +0.0009 | −0.0005 | +0.0004 [−0.0012, +0.0020] |

- **SSIM penalises position as PSNR does.** The penalty grows with the shift and removes most of the measured
  loss. After correction, frames 8-13 are still slightly below frame 0 (−0.0004 to −0.0008). Every interval
  reaches zero or past it.
- **LPIPS hardly penalises position:** at most about 0.001, and the forward and reverse estimates agree (per-burst
  r 0.95-1.00 for most frames). After correction, no frame is clearly better or worse than frame 0.
- **The three metrics do not agree on which frame is best.** Corrected PSNR favours frames 4, 5 and 7 by about
  0.08-0.14 dB. SSIM and LPIPS show no gain at those frames that clears their intervals. These are 5 bursts only.

## 5. 40 dev and 40 test bursts, 14 frames ([dev_40_bursts/](dev_40_bursts/), [test_40_bursts/](test_40_bursts/))

```
python shift_control.py --bursts_file ./splits/dev.txt --limit 40 --refs 14 --out_dir ./Results/Real/sc_dev_all_metrics
python shift_control.py --bursts_file ./splits/test.txt --limit 40 --refs 14 --out_dir ./Results/Real/sc_test_all_metrics
```

About 12 minutes per split. The PSNR columns match the v2 runs exactly in all 640 rows of each split.

Corrected gains over frame 0, positive = frame k better, 95% bootstrap intervals over bursts. **Bold** marks an
interval that is entirely above zero.

### Test scenes

| Frame | Shift (px) | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|---|
| 1 | 4.5 | +0.048 [−0.006, +0.101] | −0.0000 [−0.0003, +0.0002] | **+0.0006 [+0.0001, +0.0012]** |
| 3 | 14.7 | **+0.116 [+0.056, +0.182]** | **+0.0002 [+0.0001, +0.0004]** | **+0.0012 [+0.0004, +0.0020]** |
| 4 | 16.2 | **+0.115 [+0.057, +0.172]** | **+0.0002 [+0.0001, +0.0004]** | **+0.0016 [+0.0009, +0.0025]** |
| 5 | 19.7 | **+0.116 [+0.062, +0.172]** | +0.0001 [−0.0002, +0.0003] | +0.0006 [−0.0002, +0.0014] |
| 7 | 27.2 | **+0.091 [+0.026, +0.156]** | +0.0002 [−0.0000, +0.0004] | **+0.0014 [+0.0005, +0.0023]** |
| 9 | 37.5 | **+0.089 [+0.029, +0.150]** | +0.0001 [−0.0000, +0.0004] | **+0.0019 [+0.0010, +0.0028]** |
| 11 | 45.5 | +0.005 [−0.040, +0.051] | +0.0000 [−0.0001, +0.0002] | **+0.0017 [+0.0009, +0.0026]** |
| 13 | 50.9 | +0.011 [−0.053, +0.076] | +0.0000 [−0.0002, +0.0002] | **+0.0014 [+0.0005, +0.0023]** |

Position penalty at frame 13: PSNR 1.29 dB, SSIM 0.0029, LPIPS 0.0007.

### Dev scenes

| Frame | Shift (px) | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|---|
| 1 | 8.2 | +0.178 [−0.005, +0.361] | +0.0002 [−0.0002, +0.0006] | **+0.0045 [+0.0021, +0.0073]** |
| 3 | 20.0 | +0.211 [−0.028, +0.449] | +0.0000 [−0.0005, +0.0006] | **+0.0060 [+0.0028, +0.0097]** |
| 5 | 35.2 | +0.204 [−0.072, +0.494] | +0.0000 [−0.0006, +0.0007] | **+0.0074 [+0.0037, +0.0116]** |
| 7 | 46.4 | +0.193 [−0.102, +0.508] | +0.0001 [−0.0006, +0.0008] | **+0.0066 [+0.0032, +0.0105]** |
| 9 | 56.3 | +0.146 [−0.163, +0.476] | −0.0001 [−0.0009, +0.0007] | **+0.0058 [+0.0029, +0.0092]** |
| 11 | 64.5 | +0.170 [−0.105, +0.472] | −0.0000 [−0.0008, +0.0008] | **+0.0055 [+0.0028, +0.0087]** |
| 13 | 67.8 | +0.211 [−0.070, +0.503] | −0.0000 [−0.0009, +0.0008] | **+0.0055 [+0.0027, +0.0087]** |

Position penalty at frame 13: PSNR 3.02 dB, SSIM 0.0052, LPIPS 0.0031.

- **On the test scenes, the three metrics agree that frames 3-4 beat frame 0.** PSNR favours frames 3-9 (+0.09 to
  +0.12 dB). SSIM gains are tiny (+0.0002) and clear zero only at frames 3 and 4 (at frames 6 and 8 the interval
  touches zero). LPIPS favours most frames, including 10-13, where PSNR is level with frame 0.
- **On the dev scenes, LPIPS says every frame beats frame 0, but SSIM shows no difference.** The LPIPS gain is
  +0.0045 to +0.0074 with every interval above zero, and it is positive even before correction. PSNR means are
  +0.15 to +0.22 dB, but every interval includes zero, as in v2. SSIM is flat (within ±0.0002).
- **A few bursts drive the dev means, for LPIPS as for PSNR.** Pooled over frames 1-13, the median corrected gain
  on dev is −0.035 dB PSNR, −0.0001 SSIM and +0.0012 LPIPS, well below the means.
- **SSIM's position penalty explains almost all of its measured loss.** After correction, SSIM is within ±0.0002
  of frame 0 for every frame in both splits.
- **The LPIPS penalty is larger on these runs** than on the 5 bursts: up to 0.0031 on dev at 68 px. It is still
  smaller than the LPIPS gains on dev, which are already positive before correction. The benchmark bias for LPIPS
  is −0.00007, so averaged over 40 bursts the corrected LPIPS gains are not an artefact of the correction.
- **Per burst and frame, the metrics' corrected gains are correlated.** The correlation is 0.73-0.83 between each
  pair of metrics on dev and 0.39-0.57 on test.

## 6. Files

| Path | Contents |
|---|---|
| `Burst Super-resolution/shift_control.py` | All three metrics |
| `Burst Super-resolution/shift_control_benchmark.py` | All three metrics |
| `docs/SHIFT_CONTROL.md` | Column documentation |
| [benchmark_9_bursts/](benchmark_9_bursts/) | `per_frame.csv`, `summary.csv`, `summary.txt`, `bursts.txt` |
| [five_bursts_14_frames/](five_bursts_14_frames/) | `per_burst.csv`, `summary.csv`, `summary.txt`, `bursts.txt` |
| [dev_40_bursts/](dev_40_bursts/), [test_40_bursts/](test_40_bursts/) | `per_burst.csv`, `summary.csv`, `summary.txt` (bursts: `splits/dev.txt`, `splits/test.txt`, `--limit 40`) |
