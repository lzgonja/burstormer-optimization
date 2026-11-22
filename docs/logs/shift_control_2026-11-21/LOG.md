# Shift control: work log, 2026-11-21

Record of how the shift control was built, checked and validated, in order. Method: [SHIFT_CONTROL.md](../../SHIFT_CONTROL.md).
Justification: [SHIFT_CONTROL_VALIDATION.md](../../SHIFT_CONTROL_VALIDATION.md).

Setup: model `burstormer_burstsr_train.pth` (T02, step 10,000), BurstSR val. 40 dev and 40 test bursts chosen with
`--limit 40`, spread across the scenes of `splits/dev.txt` and `splits/test.txt`. RTX 4070 Super, Python 3.14,
torch 2.11.

## 1. Why

Reference-frame sweeps (Algorithm 2 work, branch `stanley-cao-algorithm-2`) found frame 0 to be the best or
nearly best reference on clean bursts. Under the standard metric, a perfect per-burst choice added only 0.13 dB
on the test scenes.

The question was whether the metric favours frame 0, because its ground truth is registered to frame 0. The
metric already aligns per patch with optical flow, and the sweeps already scored every reference on a common
region. What was left was the cost of warping a displaced output, so the control was built to measure that.

## 2. First version and smoke test (2 dev bursts)

Built `Burst Super-resolution/shift_control.py`. It runs references 0-4, measures each output's shift from the
frame-0 output, moves the frame-0 output by that shift, and scores it as reference k.

The smoke test found one problem: the whole-pixel probe showed a +1.64 dB penalty. The probe moved the output
but not input frame 0, which the metric uses for colour matching, so the colour fit was misaligned.

- **Fix:** move input frame 0 together with the probe (by dy/8, dx/8 in raw pixels).
- **The real-frame comparisons were not affected,** because they are scored with frame k as reference, which
  sits at the shifted position.

Direction check on burst 0156_0036, frame 4. Measured shift: (-3.94, -19.98) output px. The shifted frame-0
output vs the real frame-4 output:
- with the measured shift: 38.32 dB
- with the opposite sign: 18.75 dB
- with no shift: 19.86 dB

The direction is correct.

## 3. Full runs (40 dev, 40 test)

```
python shift_control.py --bursts_file ./splits/dev.txt --limit 40 --out_dir ./Results/Real/shift_control_dev
python shift_control.py --bursts_file ./splits/test.txt --limit 40 --out_dir ./Results/Real/shift_control_test
```

Results are in `dev/` and `test/`.

| | Dev | Test |
|---|---|---|
| Penalty, frames 1-4 | +0.03 to +0.44 dB, growing with the shift | +0.06 to +0.17 dB |
| Corrected gain, frames 1-4 | +0.03 to +0.19 dB | +0.06 to +0.12 dB |
| Half-pixel probe | -0.01 dB | +0.00 dB |
| Whole-pixel probe (11 px) | +0.14 dB | +0.09 dB |
| Oracle gain, measured | +0.33 dB | +0.13 dB |
| Oracle gain, corrected | +0.43 dB | +0.23 dB |

On test, all corrected gains for frames 1-4 had intervals above zero.

## 4. Separate branch

- **Created branch `shift-control` from `main`** (commit d4c1243), so the control can be tested without the
  Algorithm 2 work. It needs only code already on `main` (`reference_frame_sweep.py`).
- **A 2-burst run from the branch reproduced the smoke test** to within about 0.01 dB.

## 5. Validation

1. **Unit tests** (`tests/test_shift_control.py`, 9 tests, CPU):
   - identity
   - whole-pixel shift equals moving pixels
   - lossless sub-pixel round trip
   - half-pixel shift keeps detail
   - too-large shifts rejected
   - integer and sub-pixel shift recovery
   - accuracy over random shifts
   - direction

   One test first failed. The measured shift reproduced the target at 38.3 dB against a guessed 40 dB threshold,
   because of the sub-pixel estimation error. That error was then measured on 30 random shifts: mean 0.07 px,
   largest 0.13 px. The threshold was set to 35 dB, and an explicit accuracy test (under 0.15 px) was added.
   All 9 tests pass.
2. **Penalty vs shift size**, from the saved runs:

   | | Dev | Test |
   |---|---|---|
   | Rank correlation | 0.57 (p = 2e-15) | 0.27 (p = 5e-4) |
   | Mean penalty, shift under 8 px | 0.02 dB | 0.05 dB |
   | Mean penalty, shift of 24 px or more | 0.62 dB | 0.30 dB |

3. **Reverse check** (`--reverse_check`, commit d9ca9f7): a second estimate of the corrected gain, made by moving
   each frame-k output back to frame 0's position.

   ```
   python shift_control.py --bursts_file ./splits/dev.txt --limit 40 --reverse_check --out_dir ./Results/Real/shift_control_reverse_dev
   python shift_control.py --bursts_file ./splits/test.txt --limit 40 --reverse_check --out_dir ./Results/Real/shift_control_reverse_test
   ```

   Results are in `reverse_dev/` and `reverse_test/`.
   - The per-burst correlation between the two estimates is 0.69-0.96.
   - Dev: the two agree on average; every interval for the difference includes zero.
   - Test: the reverse estimate is 0.01-0.05 dB lower, significantly for frames 2 and 3.
   - Conservative (reverse) corrected gains on test: frame 1 +0.04, frame 2 +0.03, frame 3 +0.07, frame 4 +0.08
     dB. Frames 3 and 4 have intervals above zero. The oracle is +0.20 dB.
4. **Reproducibility.** The reverse-check run repeated the forward control. Per-burst values differed by at most
   0.02 dB, and by less than 0.001 dB on average. The summary tables were identical to three decimals.

## 6. Conclusion

The control is valid within about +/-0.05 dB. The standard metric penalises references other than frame 0 for
their position. On the test scenes, frames 3 and 4 are better references than frame 0 by 0.07-0.12 dB, and the
per-burst ceiling is +0.20-0.23 dB instead of +0.13. Details and limits: [SHIFT_CONTROL_VALIDATION.md](../../SHIFT_CONTROL_VALIDATION.md).

## Files

| Path | Content |
|---|---|
| `dev/`, `test/` | First full runs: `per_burst.csv`, `summary.csv`, printed summary (`summary.txt`) |
| `reverse_dev/`, `reverse_test/` | Runs with `--reverse_check` (extra column `gain_reverse`) |
| `Burst Super-resolution/shift_control.py` | The control |
| `Burst Super-resolution/tests/test_shift_control.py` | Unit tests |
