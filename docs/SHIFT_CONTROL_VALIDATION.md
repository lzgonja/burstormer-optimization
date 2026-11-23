# Is the shift control valid?

This document justifies correcting reference-frame comparisons for the penalty the BurstSR metric gives a
reference just for its position. Method and usage: [SHIFT_CONTROL.md](SHIFT_CONTROL.md).

- **Part 1: v2 (branch `shift-control-v2`).** A known-answer benchmark measures how accurate each estimator is,
  and the method changes to the one that wins. Record:
  [logs/shift_control_v2_2026-11-22/LOG.md](logs/shift_control_v2_2026-11-22/LOG.md).
- **Part 2: v1 (branch `shift-control`).** The original self-consistency checks, on 40 dev and 40 test bursts.
  Record: [logs/shift_control_2026-11-21/LOG.md](logs/shift_control_2026-11-21/LOG.md).

Model: `burstormer_burstsr_train.pth` throughout.

# Part 1: v2, measured against known answers

## Why a benchmark was needed

v1 checked itself only against itself. On the 14-frame run, its two estimates of the corrected gain disagreed:
- forward, which moves the frame-0 output to frame k;
- reverse, which moves the frame-k output back to frame 0.

Their per-burst correlation was often near zero, and they differed by up to 0.19 dB on average at frames 9-13.
Without a known answer there was no way to say which was right.

## The benchmark (`shift_control_benchmark.py`)

For each burst and reference frame k:

1. Take the frame-0 output O_0 and the real geometry from O_0 to O_k. That geometry is the PWC-Net flow between
   the outputs, kept where the flow both ways agrees within 1 px (elsewhere the homography fit), then smoothed.
2. Change O_0's content in a known way, C = D(O_0): none, Gaussian blur of 0.5 or 1.0 output px, or noise with
   σ = 0.01.
3. Build a synthetic O_k* by moving C with that geometry. Where frame 0's view ends, O_k* shows the real frame-k
   output, as a real output would. Its raw colour reference is frame 0's raw frame, moved the same way.
4. **The known answer** is score(C as frame 0) minus score(O_0 as frame 0). Both images are at frame 0's position,
   so no penalty is involved.

Every estimator sees only O_0, O_k* and the burst, exactly as on a real output.

The final benchmark uses 9 bursts: the 5 used throughout, plus the 4 hardest from the 40-burst runs (0105_0001,
0077_0014, 0212_0020, 0156_0051). That gives 9 x 13 frames x 4 changes = 468 cases, with shifts up to 110 px.

**The benchmark's own error floor is small.** Moving C there and back loses −0.004 dB on average, at worst 0.10 dB.

## Results

| Estimator | Bias (dB) | RMSE (dB) | Worst error (dB) | RMSE at 60+ px shifts |
|---|---|---|---|---|
| Forward, translation (v1 method) | +0.038 [+0.001, +0.071] | 0.385 | 4.17 | 0.708 |
| Forward, homography | +0.036 | 0.384 | 4.17 | 0.705 |
| Reverse, translation | −0.008 | 0.070 | 0.38 | 0.131 |
| **Reverse, homography (v2 default)** | **−0.011 [−0.017, −0.005]** | **0.068** | **0.38** | **0.126** |
| Reverse, dense flow | −0.004 | 0.072 | 0.80 | 0.088 |

RMSE by content change, in dB:

| Estimator | None | Blur 0.5 | Blur 1.0 | Noise |
|---|---|---|---|---|
| Forward, translation | 0.085 | 0.094 | 0.283 | 0.705 |
| Reverse, homography | 0.063 | 0.063 | 0.052 | 0.088 |

- **The forward estimate is valid only when frame k's content matches frame 0's.** It measures the penalty on frame
  0's content, and the penalty depends on content: a blurred or noisier image is aligned differently by the metric.
  With strongly different content (blur of 1 px, noise) it is off by 0.28-0.71 dB RMS, more at large shifts. This explains why v1's two estimates
  disagreed.
- **The reverse estimate is accurate to 0.07 dB RMS, with a bias of −0.011 dB.** It scores frame k's own
  reconstruction at frame 0's position, which is exactly the quantity wanted, so it needs no assumption about
  content. Its error is 0.03 dB under 8 px of shift and 0.13 dB at 60 px and more. Per burst it is 0.005-0.06 dB,
  except on the two dev bursts with 60-70 px shifts (0.11 and 0.13 dB).
- **The direction matters far more than the geometry model.** Reverse with translation is nearly as accurate as
  with a homography.
  - Dense flow is not better overall, and the true geometry is built from the same flow network, which flatters
    it.
  - The homography is the default because it is fitted independently of the flow and cannot follow content
    differences.

## Flaws found on the way, and fixed

Five flaws were found, each by a benchmark or a real run that did not add up. All are in how an image is moved and
scored, not in the metric.

1. **The warp raised noise.** 4x upsampling plus bicubic interpolation raised white-noise power by 4% (0.18 dB).
   Now 8x, which keeps it within 0.002 dB.
2. **Clipping cut noise near black.** Clipping to [0, 1] cut noise power by 25% there. An exact shift undershoots
   slightly below 0, and clipping removes that. BurstSR outputs are linear RGB and often dark. The warp no longer
   clips.
3. **The strip a move uncovers was mirror-filled.** Found on the 40-burst runs.
   - The metric fits one colour correction per patch over all its pixels, so the mirrored content spoiled whole
     patches, even though the strip itself was not scored.
   - On a burst with 60 px shifts, the reverse estimate was off by 1.1 dB RMS, and moving an image there and back
     lost up to 2.7 dB.
   - Fix: fill the strip with the real content of the image being moved onto, and leave it out of the score with
     an exact coverage mask.
4. **The homography failed on noisy content.** It missed its own tiles by 14-19 px, against at most 0.7 px
   otherwise. It now falls back to the global shift when its fit misses by more than 1 px RMS.
5. **The colour reference stayed behind.**
   - The metric colour-matches an output with its reference input frame. A moved output was scored with the raw
     frame of its new position.
   - When exposure changed between frames, both estimates paid several dB. On test burst 0105_0001 they were
     7.5 dB apart.
   - The raw frame now moves along with the output, as the probes always did. On that burst the estimates now
     agree to within 0.33 dB.

The benchmark also used the raw flow as true geometry at first, which included an 11 px flow failure. The flow is
now kept only where it is consistent both ways.

## Tests of the warp and the geometry (`tests/test_shift_geometry.py`, 19 tests)

| Test | Result required |
|---|---|
| Sampling on the pixel grid | Identity (> 70 dB) |
| A constant sub-pixel residual | Equals the exact Fourier shift (> 45 dB) |
| A half-pixel move | Gradient energy kept within 5% |
| White noise; noise near black, moved by a fractional geometry | Noise power kept within 0.01 dB |
| Rotation of 0.5° with scale, there and back | Original recovered (> 45 dB) |
| The inverse geometry | Composes to the identity within 0.01 px |
| Homography, affine and perspective fits | Known geometry recovered within 0.1-0.12 px |
| The direction of the estimate | The moved image lands on the target, at least 10 dB closer than a translation |
| Unreliable tiles | Fall back to the global shift |
| The fill | Replaces exactly the uncovered strip |
| Patch masks | Follow the metric's 384 px patches |
| The raw geometry; moving the reference | The output geometry scaled by 1/8; only the first frame changes |

Together with the 10 tests of v1 (`tests/test_shift_control.py`), all 29 pass.

## Regression

`--geometry translation --region all --estimator forward --fill mirror --colour_ref fixed` is the v1 method. It
reproduces the 2026-11-22 v1 numbers to within 0.012 dB per row. The only change is that the warp no longer clips.

## What v2 supports

All 14 reference frames, on 40 dev and 40 test bursts ([log](logs/shift_control_v2_2026-11-22/LOG.md), section 8):

- **The metric's penalty for position grows with the shift.** It reaches 1.3 dB at frame 13 on test (51 px) and
  3.0 dB on dev (68 px). Almost all of the later frames' measured loss is this penalty.
- **On the test scenes, frames 3-9 are better references than frame 0 by 0.08-0.12 dB.** Every interval is above
  zero. Frames 10-13 are level with frame 0. Forward and reverse agree here, to within 0.08 dB.
- **Frame 0 is the best reference in 1 of 40 test bursts.**

Not established:
- **On the dev scenes, no frame beats frame 0 with an interval above zero.** The means are positive, but the
  medians are near zero.

## Threats to validity (v2)

- **The benchmark changes content only by blur and noise.** Real reconstructions differ in more complex ways. The
  reverse estimate makes no assumption about content, but a structural difference could disturb its geometry
  estimate. The fidelity columns flag that.
- **The true geometry of the benchmark comes from real outputs but is smoothed.** Fine-scale parallax is not
  represented.
- **The geometry is mostly a translation on real data.** The homography falls back to the global shift in 35-60% of
  real comparisons.
- **Four scenes per split.** Intervals over bursts do not capture scene-to-scene variation, which may be why dev
  and test differ.

# Part 2: v1 self-consistency checks (2026-11-21)

The sections below are unchanged from v1. They describe the forward estimate with a translation, on 40 dev and
40 test bursts, frames 0-4. Part 1 shows that the forward estimate is off by 0.39 dB RMS when content differs, so
its numbers are superseded by Part 1.

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
- **The 10 unit tests pass:** `python -m unittest tests.test_shift_control -v`.

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
