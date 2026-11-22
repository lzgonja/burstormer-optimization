# Reference-frame sweep: does the choice of reference frame matter?

Run of 2026-11-21. Ticket: #1, "Investigate the impact of reference-frame selection". Raw results:
[logs/ref_frame_sweep_2026-11-21/](logs/ref_frame_sweep_2026-11-21/).

## Setup

| | |
|---|---|
| Model | `burstormer_burstsr_train.pth` (T02, step 10,000; see [BURSTSR_TRAINING.md](BURSTSR_TRAINING.md)) |
| Bursts | 23 BurstSR val bursts, one per scene (`--limit 23` spreads the picks round-robin over scenes) |
| References | All 14 frames. The chosen frame goes first; the other frames keep capture order |
| Metric | Standard aligned PSNR / SSIM / LPIPS (`FastAlignedMetrics`, same numbers as `utils/metrics.py`) |
| PSNR_common | The same PSNR, but every reference is scored on the pixels that all 14 outputs cover (56% of pixels on average) |
| Intervals | Paired with frame 0 per burst, 95% bootstrap over the 23 bursts |
| Runtime | About 3 minutes on an RTX 4070 Super |

Why two PSNR columns: the ground truth is registered to frame 0. An output built on frame k is displaced from it,
so the standard metric has to warp it further and covers less of the ground truth. PSNR_common removes the
coverage difference. It does not remove the extra warping cost; see "Limits".

## Results

| Reference | PSNR | Gain vs frame 0 | Wins | PSNR_common | Gain vs frame 0 | Wins | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|
| **0 (default)** | 47.22 | | | 47.34 | | | 0.9760 | 0.054 |
| 1 | 47.29 | +0.07 [-0.07, +0.22] | 14/23 | 47.44 | +0.10 [-0.05, +0.25] | 17/23 | 0.9763 | 0.056 |
| 2 | 47.30 | +0.08 [-0.01, +0.20] | 14/23 | 47.48 | **+0.14 [+0.02, +0.29]** | 14/23 | 0.9761 | 0.062 |
| 3 | 47.10 | -0.12 [-0.31, +0.05] | 9/23 | 47.40 | +0.06 [-0.09, +0.21] | 11/23 | 0.9758 | 0.067 |
| 4 | 46.71 | -0.50 [-1.17, -0.08] | 6/23 | 47.31 | -0.02 [-0.14, +0.09] | 10/23 | 0.9746 | 0.070 |
| 5 | 46.48 | -0.74 [-1.44, -0.19] | 7/23 | 47.19 | -0.15 [-0.31, +0.02] | 8/23 | 0.9738 | 0.076 |
| 6 | 45.70 | -1.52 [-2.76, -0.50] | 5/23 | 46.96 | -0.38 [-0.68, -0.12] | 6/23 | 0.9708 | 0.082 |
| 7 | 45.28 | -1.93 [-3.39, -0.73] | 4/23 | 46.73 | -0.61 [-1.01, -0.29] | 4/23 | 0.9678 | 0.087 |
| 8 | 44.63 | -2.59 [-4.29, -1.13] | 2/23 | 46.45 | -0.88 [-1.33, -0.49] | 3/23 | 0.9629 | 0.094 |
| 9 | 44.15 | -3.06 [-4.82, -1.56] | 2/23 | 46.28 | -1.06 [-1.59, -0.60] | 2/23 | 0.9591 | 0.100 |
| 10 | 43.36 | -3.86 [-5.67, -2.24] | 1/23 | 45.96 | -1.38 [-2.02, -0.81] | 1/23 | 0.9536 | 0.107 |
| 11 | 42.94 | -4.28 [-6.15, -2.58] | 1/23 | 45.72 | -1.61 [-2.32, -0.98] | 1/23 | 0.9489 | 0.113 |
| 12 | 42.36 | -4.86 [-6.81, -3.06] | 1/23 | 45.27 | -2.07 [-2.95, -1.26] | 1/23 | 0.9444 | 0.118 |
| 13 | 41.80 | -5.42 [-7.45, -3.55] | 1/23 | 44.88 | -2.46 [-3.76, -1.41] | 1/23 | 0.9353 | 0.124 |
| Best frame per burst (oracle) | 47.43 | **+0.21 [+0.10, +0.34]** | 20/23 | 47.59 | **+0.25 [+0.13, +0.39]** | 20/23 | | |

"Wins" counts the bursts where that reference scores higher than frame 0. The oracle uses the ground truth to pick
the best frame per burst. It mostly picks early frames: frames 1 and 2 in 15 of 23 bursts, frame 0 in 3.

## Findings

- **The reference frame matters.** Across the 14 frames, mean PSNR spans 5.4 dB under the standard metric and
  2.6 dB on the common region. SSIM and LPIPS get worse in the same order.
- **Frame 0 is a good default, but not the best one.** Frames 1 and 2 match or slightly beat it (+0.07 to
  +0.14 dB). Only frame 2 on the common region has an interval that excludes zero. Frame 3 is level with frame 0,
  and frames 4 and later are worse.
- **Late frames are clearly worse references,** even on the common region: -0.4 dB at frame 6, -2.5 dB at
  frame 13. Part of the loss under the standard metric is framing. The common region shrinks the frame-13 gap from
  -5.4 to -2.5 dB.
- **There is headroom for per-burst selection.** Choosing the best frame per burst would gain +0.21 dB (standard)
  to +0.25 dB (common region), and it beats frame 0 in 20 of 23 bursts. This is an upper bound: picking the maximum
  of 14 noisy scores inflates it somewhat.
- **LPIPS prefers frame 0 at every step.** It rises steadily from frame 0 (0.054) to frame 13 (0.124). Frames 1
  and 2 gain PSNR but lose a little LPIPS, so they are not a clear win on every metric.

What this means for reference selection: a selector should look at the first few frames, not the whole burst.
The realistic gain on clean bursts is a fraction of a dB. A larger benefit is avoiding a damaged default frame.

## How to reproduce

From `Burst Super-resolution/`, with the BurstSR val set in `./burstsr_dataset` and the weights from the
[t02-burstsr-v1 release](https://github.com/lzgonja/burstormer-optimization/releases/tag/t02-burstsr-v1):

```
python reference_frame_sweep.py --weights ./Trained_models/Real/burstormer_burstsr_train.pth \
    --limit 23 --out_dir ./Results/Real/t01_sweep_scenes
python summarize_reference_sweep.py --sweep_dir ./Results/Real/t01_sweep_scenes
```

The first command writes `per_burst.csv` (one row per burst and reference) and `summary.csv` (means per reference,
plus simple input-only policies). The second writes `vs_frame0.csv`, the paired gains and intervals above, and
needs no GPU.

## Limits

- **Small sample:** 23 bursts, one per scene. The intervals for frames 1-3 are wide.
- **The metric still favours frame 0.** PSNR_common equalises the pixels scored, but an output built on a later
  frame still has to be warped further onto the ground truth. The shift-control experiment (`shift-control`
  branch, `docs/SHIFT_CONTROL.md`) measures this cost at 0.03-0.44 dB for frames 1-4. The gains of frames 1-4
  above are therefore, if anything, underestimates.
- **Not the official metric.** PSNR_common and the paired intervals are our own evaluation. Published BurstSR
  numbers use the standard metric over the full val set.
- **Repeated runs differ by about 0.01 dB.** Optical flow on the GPU is not bit-exact between runs.
