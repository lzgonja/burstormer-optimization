# Shift control: does the metric favour frame 0?

Script: `Burst Super-resolution/shift_control.py`. Run of 2026-11-21. Raw results and work log:
[logs/shift_control_2026-11-21/](logs/shift_control_2026-11-21/) ([LOG.md](logs/shift_control_2026-11-21/LOG.md)).
Why the correction can be trusted: [SHIFT_CONTROL_VALIDATION.md](SHIFT_CONTROL_VALIDATION.md).

## What it is

The BurstSR ground truth is lined up with frame 0 of each burst. When Burstormer uses another frame as its
reference, the output is shifted by a few pixels. To score it, the metric has to warp the output further back onto
the ground truth, and that costs some PSNR. A later frame could therefore look worse only because of its position,
not because its reconstruction is worse.

The shift control measures that penalty on its own:

1. Run the model with references 0 to 4.
2. Measure how far each frame-k output is shifted from the frame-0 output (phase correlation).
3. Move the frame-0 output by exactly that shift. Mirror padding and an exact Fourier shift add no blur and no
   ringing.
4. Score the shifted copy exactly as the frame-k output is scored.

The shifted copy has the same content as the frame-0 output, so any drop in PSNR is the metric's penalty for
position. Adding the penalty back to frame k's real score gives its **corrected gain**: how much better or worse
frame k really is as a reference.

It works like calibrating a scale: weigh a known object, find the error, and subtract it from every other
measurement.

Two probes show where the penalty comes from. Both use the frame-0 output, scored as reference 0, with input
frame 0 moved along with the output:
- **probe_half_px** shifts by half an output pixel, so it measures resampling only.
- **probe_whole_px** shifts by 8 x 8 output pixels, so it measures displacement only.

All scores use only the pixels that every image covers, so no reference is scored on more or less area.

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
# Unit tests of the shift operations (CPU, seconds)
python -m unittest tests.test_shift_control -v

# Quick check: 2 bursts
python shift_control.py --bursts_file ./splits/dev.txt --limit 2 --out_dir ./Results/Real/shift_control_smoke

# The runs below, a few minutes each on an RTX 4070 Super
python shift_control.py --bursts_file ./splits/dev.txt --limit 40 --out_dir ./Results/Real/shift_control_dev
python shift_control.py --bursts_file ./splits/test.txt --limit 40 --out_dir ./Results/Real/shift_control_test
```

Options:
- `--weights`, `--data_root`, `--pwcnet`: file locations.
- `--refs N`: test references 0 to N-1 (default 5).
- `--bursts_file`: a list of burst names, for example the scene splits in `./splits/`.
- `--limit N`: use N bursts spread across the scenes.
- `--reverse_check`: also estimate the corrected gain the other way round. Each frame-k output is moved back to
  frame 0's position and scored as frame 0 (column `gain_reverse`, plus an agreement table). About 40% slower.

Outputs in `--out_dir`:
- **`per_burst.csv`**: one row per burst and reference frame. Columns:
  - `shift_dy`, `shift_dx`: the measured shift, in output pixels.
  - `psnr_real`: the real frame-k output.
  - `psnr_shifted`: the shifted frame-0 output.
  - `penalty`, `gain_real`, `gain_corrected`.
  - `shifted_vs_real_psnr`: how close the shifted copy is to the real frame-k output. High values (40 dB or
    more) mean the two outputs differ mainly by a shift, so the correction is meaningful.
- **`summary.csv`** and the printed table: the mean per frame with a 95% bootstrap interval over bursts, plus the
  per-burst oracle before and after correction.

**Sanity checks.**
- **Penalty for frame 0:** must be exactly 0.
- **`probe_half_px`:** should be about 0. If it is not, resampling itself costs PSNR.
- **`shifted_vs_real_psnr`:** should be well above the PSNR between unshifted outputs. In our check, 38 dB with
  the measured shift against 19 dB with the opposite sign or no shift.

## Results (model `burstormer_burstsr_train.pth`)

40 dev and 40 test bursts. The region common to every image is 62-63% of pixels. Gains are relative to frame 0,
with 95% bootstrap intervals over bursts.

| Frame | Shift (output px), dev / test | Penalty, dev | Penalty, test | Measured gain, test | Corrected gain, test | Corrected gain, dev |
|---|---|---|---|---|---|---|
| 1 | 8 / 5 | +0.03 [-0.02, +0.10] | +0.06 [+0.02, +0.11] | +0.00 | +0.06 [+0.01, +0.12] | +0.18 [-0.02, +0.38] |
| 2 | 16 / 11 | +0.17 [+0.07, +0.29] | +0.13 [+0.07, +0.19] | -0.06 | +0.07 [+0.01, +0.13] | +0.19 [-0.02, +0.42] |
| 3 | 20 / 15 | +0.22 [+0.12, +0.33] | +0.13 [+0.06, +0.20] | -0.01 | +0.12 [+0.04, +0.20] | +0.13 [-0.11, +0.35] |
| 4 | 27 / 16 | +0.44 [+0.27, +0.62] | +0.17 [+0.10, +0.25] | -0.08 | +0.10 [+0.05, +0.14] | +0.03 [-0.19, +0.26] |

| Probe | Penalty, dev | Penalty, test |
|---|---|---|
| Half-pixel shift | -0.01 [-0.03, +0.00] | +0.00 [-0.01, +0.02] |
| 8 x 8 pixel shift | +0.14 [+0.07, +0.23] | +0.09 [+0.04, +0.13] |

| Best frame per burst (oracle) | Measured | Corrected |
|---|---|---|
| Dev | +0.33 | +0.43 |
| Test | +0.13 | +0.23 |

What this shows:
- **The metric charges a penalty for position.** It grows with the shift: up to 0.44 dB for frame 4 on dev.
- **The penalty comes from the displacement, not from resampling.** A half-pixel shift costs nothing.
- **Corrected, frames 3 and 4 are better references than frame 0 on the test scenes, by 0.07-0.12 dB.** This
  holds under both estimates. Frames 1 and 2 beat frame 0 under this estimate (+0.06 and +0.07 dB), but the
  reverse check gives about half that, with intervals that include zero. Frame 0 is the best frame in only 3-4 of
  40 test bursts.
- **The real headroom for choosing the reference is larger than the standard metric shows:** +0.20 to +0.23 dB
  on test, against +0.13. The per-burst oracle includes some selection noise.
- **Read corrected gains as accurate to about +/-0.05 dB.** That is how far the two estimates differ on average
  (see the validation document).

## Limits

- **The correction models only translation.** Outputs from different references also differ slightly in other
  ways (41-45 dB between the shifted copy and the real output).
- **Corrected numbers are our own evaluation, not the official BurstSR metric.** Published results, including
  Burstormer's 48.82 dB, use the standard metric. Under it, frame 0 keeps its advantage.
- **The sample is small:** 40 bursts per split. The dev intervals are wide.
- **Repeated runs differ by about 0.01 dB.** Optical flow on the GPU is not bit-exact between runs.
