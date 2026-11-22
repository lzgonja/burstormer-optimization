# Shift control: all 14 reference frames on 5 bursts, 2026-11-22

How much PSNR does the metric take from later reference frames only because of their position? Method:
[SHIFT_CONTROL.md](../../SHIFT_CONTROL.md). Earlier runs (frames 0-4, 40 dev and 40 test bursts):
[../shift_control_2026-11-21/LOG.md](../shift_control_2026-11-21/LOG.md).

Setup: model `burstormer_burstsr_train.pth` (T02), BurstSR val, the 5 bursts in
[five_bursts_14_frames/bursts.txt](five_bursts_14_frames/bursts.txt). `shift_control.py` at commit 356c195. RTX 4070 Super.

From `Burst Super-resolution/`:

```
python shift_control.py --bursts_file bursts.txt --refs 14 --reverse_check --out_dir ./Results/Real/shift_control_five_14
```

Files in [five_bursts_14_frames/](five_bursts_14_frames/): `per_burst.csv` and `summary.csv` as the script wrote
them, and `summary.txt`, the printed summary.

## 1. Padding fix

The first attempt stopped on burst 0019_0020: the shift of a late frame was larger than the fixed 64 px mirror
padding, which had been sized for frames 0-4. Frame 13 of that burst is 110 output px from frame 0.

- **Fix (commit 356c195):** the padding is now max(64, shift + 4). Shifts under 60 px are handled exactly as
  before, so the 2026-11-21 results are unchanged.
- **Tests:** the test that expected large shifts to be rejected was replaced by two. One checks that a shift past
  the default padding equals moving the pixels (> 80 dB away from the edges). The other checks that a shift as
  large as the image is still rejected. All 10 tests pass.

## 2. Results

Mean over the 5 bursts, in dB, with 95% bootstrap intervals. Gains are relative to frame 0.

| Frames | Shift (output px) | Penalty | Measured gain | Corrected gain |
|---|---|---|---|---|
| 1-4 | 2-9 | -0.03 to +0.06 | -0.04 to +0.03 | -0.02 to +0.02 |
| 5-7 | 12-17 | +0.15 to +0.27 | -0.14 to -0.20 | +0.00 to +0.07 |
| 8-9 | 21-23 | +0.47 to +0.61 | -0.43 to -0.52 | +0.04 to +0.08 |
| 10-13 | 29-40 | +1.06 to +1.71 | -0.95 to -1.63 | +0.05 to +0.13 |

- **The penalty grows with the shift, and burst 0019_0020 drives most of it.** Its camera moves a lot: at frame 13
  it loses 7.4 dB measured, 7.5 dB of which is the penalty. In the other four bursts the penalty never goes above
  0.55 dB.
- **After correction no frame clearly beats frame 0.** Mean corrected gains stay within about +/-0.13 dB, and
  almost every interval includes zero.
- **Best frame per burst:** +0.06 dB measured, +0.20 dB corrected, +0.17 dB reverse. The best measured frames
  (8, 2, 2, 1, 0) are the same as in the earlier reference-frame sweep.

## 3. Limits

- **The two corrected estimates disagree burst by burst.** Their per-burst correlation is often near zero or
  negative. On 0019_0020, frame 13 is +0.06 dB forward and -0.68 dB reverse. Read corrected gains for frames 9
  and later as rough, to about +/-0.2 dB.
- **The correction models translation only.** Averaged per frame, the moved copy matches the real output at
  42-49 dB. The worst pairs are at 31-32 dB: frames 9-13 of 0033_0001, whose shifts are small (about 5 px) but
  which is the hardest burst (26.6 dB for frame 0), and frame 13 of 0019_0020.
- **Scored on 58% of pixels,** the area all 14 outputs cover (48% for 0019_0020). The 2026-11-21 runs scored
  62-63%, so the numbers are not directly comparable.
- **Five bursts give wide intervals.** This gauges the size of the effect; it is not a final result.
