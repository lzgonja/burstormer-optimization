# Is the shift control valid?

This document justifies the change on the `shift-control` branch: correcting reference-frame comparisons for the
penalty the BurstSR metric gives a reference just for its position. Method and usage:
[SHIFT_CONTROL.md](SHIFT_CONTROL.md). Record of the work: [logs/shift_control_2026-11-21/LOG.md](logs/shift_control_2026-11-21/LOG.md).
All numbers are from 40 dev and 40 test bursts, with model `burstormer_burstsr_train.pth`.

## The claim

The standard aligned PSNR scores a reference frame partly on its position. The ground truth is registered to
frame 0, so an output made with frame k must be warped further, and that costs PSNR. The control measures this
penalty by scoring a copy of the frame-0 output that has only been moved to frame k's position. Adding the penalty
back gives a fairer comparison of how good each frame is as a reference.

Result, in short: on the test scenes, frames 3 and 4 are better references than frame 0 by about 0.07-0.12 dB.
The standard metric hides this and shows them as slightly worse.

## What the method assumes, and the evidence for each assumption

The correction is valid if all of the following hold. Each one was tested.

### 1. Moving an image changes its position and nothing else

If the shift itself blurred the image or added artefacts, part of the "penalty" would be caused by the control.

| Test | Result |
|---|---|
| Zero shift (unit test) | Output equals input (tolerance 1e-5) |
| Whole-pixel shift vs moving pixels (unit test) | Identical away from the edges (> 80 dB) |
| Sub-pixel shift there and back (unit test) | Original recovered (> 60 dB): no blur |
| Half-pixel shift (unit test) | Detail (gradient energy) kept within 5%; bilinear interpolation would lose it |
| **Half-pixel shift on real outputs, scored by the metric** | **-0.01 [-0.03, +0.00] dB dev, +0.00 [-0.01, +0.02] dB test** |

The last row is the key one. Moving a real output by 0.7 pixels, the worst case for interpolation, changes its
score by nothing measurable. The shift operation is neutral.

### 2. The measured shift is accurate and points the right way

| Test | Result |
|---|---|
| Accuracy on 30 random known shifts in [-20, 20] px | Mean error 0.07 px, largest 0.13 px |
| Direction (unit test) | The measured shift lands the image on its target, at least 15 dB closer than the opposite direction |
| Direction on a real burst (0156_0036, frame 4) | Shifted copy vs the real frame-4 output: 38.3 dB. Opposite direction: 18.8 dB. No shift: 19.9 dB |

An error of 0.13 px is far smaller than the 0.7 px shift that costs nothing (point 1), so it cannot affect the
penalty.

### 3. Outputs from different references differ mainly by a shift

The correction models the difference between references as a translation. If outputs also differed in other ways
(rotation, different content), moving the frame-0 output would not reproduce frame k's situation.

- **The shifted frame-0 output matches the real frame-k output at 41-45 dB on average.** The worst single burst
  is 28 dB on dev and 32 dB on test.
- **Unshifted, the same pair matched at 19.9 dB** on the burst checked in point 2.

So the translation explains most of the difference between references. What is left is small next to the
position effect, and it is what the reverse check (point 4) covers.

### 4. The penalty depends on position, not on image content

The correction moves the frame-0 output to frame k's position, then applies that penalty to frame k's own output.
This is only fair if the penalty would be the same for frame k's content.

The **reverse check** (`--reverse_check`) tests this. It estimates the corrected gain a second, independent way:
1. Move frame k's own output back to frame 0's position.
2. Score it as frame 0 would be scored.

The two estimates share no shifted image. If the assumption holds, they agree.

| Frame | Dev: forward | Dev: reverse | Dev: difference | Dev: r | Test: forward | Test: reverse | Test: difference | Test: r |
|---|---|---|---|---|---|---|---|---|
| 1 | +0.178 | +0.127 | -0.05 [-0.11, +0.01] | 0.96 | +0.064 | +0.035 | -0.03 [-0.07, +0.01] | 0.69 |
| 2 | +0.193 | +0.182 | -0.01 [-0.10, +0.07] | 0.94 | +0.070 | +0.025 | -0.05 [-0.08, -0.02] | 0.87 |
| 3 | +0.127 | +0.161 | +0.03 [-0.04, +0.11] | 0.95 | +0.116 | +0.070 | -0.05 [-0.09, -0.00] | 0.81 |
| 4 | +0.032 | +0.111 | +0.08 [-0.06, +0.24] | 0.84 | +0.095 | +0.082 | -0.01 [-0.05, +0.03] | 0.71 |

Corrected gain of frame k over frame 0, in dB. "Forward" moves the frame-0 output to frame k; "reverse" moves
the frame-k output to frame 0. The difference is reverse minus forward, with a 95% bootstrap interval. r is the
per-burst correlation between the two estimates.

- **Burst by burst, the two estimates agree closely** (r = 0.69-0.96). They rank the bursts the same way.
- **On dev, the averages agree:** every interval for the difference includes zero.
- **On test, the reverse estimate is 0.01-0.05 dB lower**, significantly for frames 2 and 3. The forward
  estimate is slightly generous there. Corrected gains should be read as accurate to about +/-0.05 dB, and the
  reverse estimate used when a conservative number is needed (next section).

### 5. The penalty has a geometric cause

If the penalty is caused by position, it should grow with the size of the shift. If it were noise, it would not.

| | Dev | Test |
|---|---|---|
| Rank correlation, penalty vs shift (160 burst-frame pairs) | 0.57 (p = 2e-15) | 0.27 (p = 5e-4) |
| Slope | 0.017 dB per output pixel | 0.009 dB per output pixel |
| Mean penalty, shift under 8 px | 0.02 dB | 0.05 dB |
| Mean penalty, shift of 24 px or more | 0.62 dB | 0.30 dB |
| Probe: the frame-0 output moved 11 px, input frame moved along | +0.14 [+0.07, +0.23] dB | +0.09 [+0.04, +0.13] dB |

The penalty grows with the shift, and moving the frame-0 output on purpose reproduces it. Together with the
half-pixel result in point 1, this places the cause in larger displacements: optical-flow error and content near
the patch edges, not resampling.

### 6. Every image is scored on the same pixels, with consistent colour matching

- **All scores use one region per burst:** the pixels that are valid and in view for every real, shifted and
  moved-back image (62-63% of each output). No reference gains or loses area.
- **The metric colour-matches with the reference input frame, so that frame must sit where the output sits.**
  For the shifted copies this holds automatically: they are scored with frame k as reference, at frame k's
  position. For the probes, input frame 0 is moved together with the output.
- **The first version of the whole-pixel probe did not move the input frame, and showed a false +1.6 dB
  penalty.** The 2-burst smoke test caught this, and it was fixed before any reported run (see the log). The
  real-frame results were never affected.

### 7. The results reproduce

- **Rerunning the full control gave almost identical per-burst numbers:** at most 0.02 dB different, and less
  than 0.001 dB on average. The second run was part of the reverse check. Summary tables were identical to three
  decimals.
- **The script also ran from the `shift-control` branch alone,** with only code from `main`, and reproduced the
  smoke-test numbers to within 0.01 dB.
- **The 9 unit tests pass:** `python -m unittest tests.test_shift_control -v`.

## What can be claimed

Gain over frame 0 on the test scenes, in dB:

| Frame | Standard metric | Corrected, forward | Corrected, reverse (conservative) | Mean of both |
|---|---|---|---|---|
| 1 | +0.00 | +0.06 [+0.01, +0.12] | +0.04 [-0.02, +0.08] | +0.05 [+0.00, +0.10] |
| 2 | -0.06 | +0.07 [+0.01, +0.13] | +0.03 [-0.03, +0.08] | +0.05 [-0.01, +0.10] |
| 3 | -0.01 | +0.12 [+0.04, +0.20] | +0.07 [+0.00, +0.14] | +0.09 [+0.03, +0.16] |
| 4 | -0.08 | +0.10 [+0.05, +0.14] | +0.08 [+0.03, +0.13] | +0.09 [+0.04, +0.13] |
| Best frame per burst (oracle) | +0.13 | +0.23 [+0.17, +0.30] | +0.20 [+0.15, +0.25] | |

Supported by both estimates:
- **The standard metric understates frames other than 0.** The penalty is up to 0.44 dB (frame 4, dev), and it
  grows with the shift.
- **On the test scenes, frames 3 and 4 are better references than frame 0, by 0.07-0.12 dB.** Every interval is
  above zero.
- **Frame 0 is the best reference in only 3-4 of 40 test bursts** (forward and reverse).
- **The ceiling for choosing the reference per burst is +0.20 to +0.23 dB on test,** against +0.13 measured.

Supported by the forward estimate only:
- **Frames 1 and 2 beating frame 0 on test** (+0.06 and +0.07 dB). The reverse estimate is about half that, with
  intervals that include zero.

Not established:
- **On dev, no single frame beats frame 0 with an interval above zero.** Dev has only 4 scenes and wide intervals.

## Threats to validity

- **Translation only.** The correction ignores rotation and other small differences between outputs. The reverse
  check bounds the effect: the two estimates differ by at most about 0.05 dB on average.
- **One model.** Everything uses `burstormer_burstsr_train.pth`. A different model could have a different
  penalty, so rerun the control for it.
- **The bias is in the metric's alignment step,** PWC-Net flow plus per-patch colour matching. Other metrics need
  their own control.
- **The sample is small:** 40 bursts per split, 4 dev scenes and 4 test scenes. Intervals are bootstrapped over
  bursts, so scene-to-scene variation is underrepresented.
- **This is not the official metric.** Published BurstSR numbers use the standard metric. Report corrected gains
  as a separate, clearly labelled analysis.

## How to reproduce

From `Burst Super-resolution/`:

```
python -m unittest tests.test_shift_control -v
python shift_control.py --bursts_file ./splits/dev.txt --limit 40 --reverse_check --out_dir ./Results/Real/shift_control_reverse_dev
python shift_control.py --bursts_file ./splits/test.txt --limit 40 --reverse_check --out_dir ./Results/Real/shift_control_reverse_test
```

Compare with `docs/logs/shift_control_2026-11-21/reverse_dev` and `reverse_test`. Differences of about 0.01 dB are
expected, because GPU optical flow is not bit-exact between runs.
