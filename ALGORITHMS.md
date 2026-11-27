
# Reference-Frame Selection Algorithms

This document describes the three reference-frame selection algorithms developed for our Burstormer project.

## Overview

Our goal is to investigate whether adaptive reference-frame selection can improve Burstormer's reconstruction quality compared to its default selection of frame 0.

### Shared Interface

All three algorithms must follow the same interface:

- **Input:** An image burst containing N frames.
- **Output:** An integer representing the selected reference-frame index (0 to N-1).
- **Constraint:** Selection must not rely on ground-truth images or reconstruction metrics.

---

## Algorithm 1: Gradient-Based Reference Selection (GBRS)

**Contributor:** Lana Zgonjanin

### Motivation
Burstormer's reconstruction quality depends on the quality of its reference
frame. Conventional sharpness metrics can incorrectly favour noisy images
because random noise introduces high-frequency intensity variations. This
approach uses directional gradient coherence to distinguish meaningful image
structure from random noise, aiming to select a sharp and relatively clean
reference frame.

### Method
### Method

This approach adapts the no-reference image quality metric proposed by
Zhu and Milanfar (2010) for reference-frame selection.

The underlying assumption is that meaningful image structures, such as edges,
produce gradients with consistent local orientations, whereas random noise
produces gradients with less directional consistency. Blurring, on the other
hand, reduces gradient strength.

**1. Gradient computation**

Compute the horizontal and vertical gradients of each RAW channel and divide
them into non-overlapping 8 × 8 patches.

For each patch, construct a gradient matrix:

$$
G =
\begin{bmatrix}
g_{x,1} & g_{y,1} \\
\vdots & \vdots \\
g_{x,64} & g_{y,64}
\end{bmatrix}
$$

**2. Singular value decomposition**

Apply SVD to the gradient matrix to obtain two singular values,
$s_1 \geq s_2$.

These describe the gradient strength along the dominant and perpendicular
directions, respectively.

**3. Directional coherence**

Calculate the directional coherence of each patch:

$$
R = \frac{s_1-s_2}{s_1+s_2}
$$

Coherence measures how consistently the gradients within a patch are oriented along the same direction. A value close to 1 indicates strong directional structure, while a value
close to 0 indicates little directional consistency.

**4. Patch quality score**

Combine gradient strength and directional coherence:

$$
Q = \begin{cases}
s_1R, & R \geq 0.234 \\
0, & \text{otherwise}
\end{cases}
$$

The threshold of 0.234 is adopted from Zhu and Milanfar (2010). It reduces the contribution of patches that lack meaningful directional structure.

**5. Frame quality score and reference selection**

Average the patch quality scores within each RAW channel, including
patches assigned a score of zero. Average the four channel scores to
obtain the overall quality score of the frame.

The reference frame is selected by maximizing this score across the burst:

$$
i^* = \arg\max_i Q(I_i)
$$

This approach aims to favour frames containing strong, coherent image
structure while reducing sensitivity to blur and random noise.

### Implementation
- **Source:** `reference_selection/algorithm1.py`
- **Tests:** `tests/test_algorithm1.py`
- **Patch size:** 8 × 8
- **Coherence threshold:** 0.234

The algorithm operates directly on the four packed RAW channels without
modifying the original burst. It requires no ground-truth images or
trained model weights.

### References
- Zhu, X., & Milanfar, P. (2010). Automatic Parameter Selection for
  Denoising Algorithms Using a No-Reference Measure of Image Content.
  IEEE Transactions on Image Processing, 19(12), 3116–3132.
  https://doi.org/10.1109/TIP.2010.2052820

---

## Algorithm 2: Central, exposure-consistent reference selection

**Contributor:** Stanley Cao

### Motivation
Burstormer aligns every frame to its reference and reconstructs the scene as the reference sees it. Once the
metric's penalty for a displaced reference is removed (shift control, `docs/SHIFT_CONTROL.md`), frame 0 turns out
to be the worst reference on average, and frames near the middle of the burst the best. Frame 0 sits at the edge of
the burst: on the dev scenes it is on average 5.4 px from the other frames, against 2.6-2.8 px for frames 5-8. Its
brightness also differs most from the rest of the burst.

The first version of this algorithm (below, "First version") kept frame 0 unless it looked damaged and only
considered frames 0-4. That preference came from the uncorrected metric. HDR+ (Hasinoff et al., 2016) also chooses
among the first frames, but to avoid shutter lag, not for quality.

Image-quality scores do not tell which frame Burstormer reconstructs best. On 80 labelled bursts, HDR+ sharpness,
the Zhu & Milanfar Q score (algorithm 1), a noise estimate and the consistency cost all have within-burst rank
correlations of about 0.1 or less with the corrected gain, with signs that differ between dev and test. No
burst-only feature tested ranks the frames of a burst well, centrality included. What carries over between dev and
test is coarser: a reference in the middle of the burst's motion, with the burst's brightness, beats one at its
edge.

### Method
For each frame k, from the burst alone:

- **Centrality:** the mean translation from k to every other frame (phase correlation on the green channel,
  sub-pixel). The most central frame leaves the smallest motions to align.
- **Brightness deviation:** `|log(brightness_k / median brightness of the burst)|`. Buades et al. (2022) normalise
  every frame to the reference exposure before comparing frames. Our tests showed that a reference whose
  brightness differs from the rest hurts Burstormer: one twice as bright costs 4-5 dB with no clipping or extra
  noise.
- **Consistency (health gate):** frame k is aligned to its 4 nearest frames in time, and its cost is the median of
  `mean|k - j|` over them, divided by the burst's typical brightness (exposure normalisation, so the cost does not
  depend on the scene's level). Noise, blur, under-exposure, over-exposure and clipping all raise it.

Rule:

    score(k) = z(centrality_k) + 0.5 * z(brightness deviation_k)        (lower is better)

z() standardises over the 14 frames of the burst. The brightness spread is floored at 1%, so sub-percent noise in
an evenly exposed burst cannot outweigh its motion. Frames whose consistency cost is more than 30% above the burst
median are excluded. The reference is the healthy frame with the lowest score.

### Results
Shift-corrected gain over frame 0, 40 dev and 40 test bursts, 95% bootstrap over bursts. Details and the damage
tests: [docs/ALGORITHM2.md](docs/ALGORITHM2.md).

| Test bursts | PSNR (dB) | SSIM (x1e-3) | LPIPS (x1e-3) |
|---|---|---|---|
| Algorithm 1 (GBRS) | +0.059 [+0.005, +0.118] | +0.019 [-0.091, +0.149] | +0.928 [+0.278, +1.652] |
| Algorithm 2, first version | +0.011 [+0.000, +0.031] | +0.039 [+0.000, +0.116] | +0.280 [+0.000, +0.813] |
| **Algorithm 2** | **+0.127 [+0.056, +0.204]** | **+0.217 [+0.020, +0.425]** | **+1.369 [+0.525, +2.269]** |
| Oracle (best frame per burst, uses ground truth) | +0.276 | +0.585 | +3.145 |

On dev, algorithm 2 gains +0.197 dB PSNR, +0.079e-3 SSIM and +6.80e-3 LPIPS. Only the LPIPS interval excludes zero
there, because the dev scenes vary more. On clean bursts it is statistically tied with always using a middle frame
(frame 3 on test: +0.116 dB, +0.218e-3, +1.182e-3): most of the gain comes from leaving the edge of the burst. Its
adaptivity is what a fixed frame cannot do: follow uneven motion, and avoid a damaged frame. When the frame it
would pick is damaged, it takes a neighbouring healthy
frame instead: in all or nearly all of 80 bursts for noise of 0.003 or more, blur sigma 2 and every exposure error
tested (x0.125 to x8), and in 71 of 80 for blur sigma 1.

### Implementation
`Burst Super-resolution/reference_selection/algorithm2.py`, `select_reference_frame(burst) -> int`, the same
interface as algorithm 1. It takes a burst `(N, 4, H, W)` as a torch tensor or numpy array (packed RGGB in [0, 1];
`(1, N, 4, H, W)` is also accepted) and returns the frame index as a Python int. A wrong shape, an empty burst or
non-finite values raise `ValueError`. It runs on CPU or GPU (about 40 ms per burst on the CPU) and needs no network
or ground truth. Parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `BRIGHTNESS_WEIGHT` | 0.5 | Weight of the brightness term. Chosen on dev (0.25 and 0.5 tied; 1.0 and 2.0 worse) |
| `BRIGHTNESS_FLOOR` | 0.01 | Smallest brightness spread used to standardise (1%; the median spread is 0.8%) |
| `HEALTH_MARGIN` | 0.3 | A frame whose consistency cost is >30% above the burst median is never picked |
| `NEAREST` | 4 | The health gate compares each frame with this many neighbours in time |

`frame_features(burst)` returns the three features and `choose_from_features(features)` applies the rule.
`score_ref_selectors.py --labels <per_burst.csv>` scores every selector against shift-control labels without
running the network. Tests: `python -m pytest tests/test_algorithm2.py`.

### First version (kept as `select_reference_frame_v1`)
Each of the first 5 frames was compared with the 8 frames that follow it, and

    cost(c) = median over the 8 frames j of mean|c - j|  /  typical brightness of the frames j

Frame 0 was kept unless its cost was more than 20% above the median cost of frames 1-4. Then the earliest frame
within 20% of that median was used. It recovered almost all of the loss from a damaged frame 0, but on clean bursts
it kept frame 0 in 92-95% of cases, so its gain there was zero by construction. Also tried and dropped in that
version: the paper's well-exposedness weight `w(x) = 1 - (2x - 1)^12`, which rewards brighter frames in BurstSR's
dark scenes and so kept an over-exposed reference.

### References
- A. Buades, O. Martorell, M. Sánchez-Beeckman, "Joint denoising and HDR for RAW video sequences",
  arXiv:2201.07066, 2022 (exposure normalisation, Sec. 3.2; well-exposedness weight, Sec. 4, tested and dropped).
- S. W. Hasinoff et al., "Burst photography for high dynamic range and low-light imaging on mobile cameras",
  SIGGRAPH Asia 2016 (HDR+: sharpness-based choice among the first frames, limited to them for shutter lag).
- X. Zhu, P. Milanfar, "Automatic parameter selection for denoising algorithms using a no-reference measure of
  image content", IEEE TIP 19(12), 2010 (the Q score of algorithm 1, tested here as a per-frame feature).

---


## Algorithm 3: Iterative Reference Frame Refinement

**Contributor:** Victoria Kreinin

### Motivation
Traditional burst super-resolution methods use an image from the original burst as the reference frame. However, the initial reconstruction may be limited in recovering the scene details, as they may not be as clearly represented in the original reference frame.

This approach investigates whether feeding the model's own reconstruction back into the network as a new reference can improve the subsequent reconstruction. The hypothesis is that the reconstructed image may provide a more informative reference than the original frame.

### Method
The algorithm uses an iterative, two-pass approach:

1. Initial reconstruction: Feed the original 14-frame burst into Burstormer using the original frame 0 as the reference. Generate an initial 640 × 640 RGB reconstruction.

2. Synthetic reference generation: Convert the RGB reconstruction into an approximate 4-channel, 80 × 80 packed RGGB pseudo-RAW image. This conversion involves downsampling the RGB output, recreating the RGGB Bayer pattern, and packing the resulting mosaic into four channels.

3. Reference replacement: Replace frame 0 in a copy of the original burst with the synthetic pseudo-RAW reference. Optional: blend the synthetic reference with the original frame 0 using a blending parameter α:

reference = α × synthetic_reference + (1 − α) × original_reference

4. Iterative reconstruction: Feed the modified burst into Burstormer for a second reconstruction. The remaining 13 original frames are kept unchanged.

5. Evaluation: Compare the second reconstruction against the baseline using PSNR, SSIM, and LPIPS. Both predictions are evaluated using the original burst to maintain consistent reference-frame colour matching.

### Implementation
The algorithm uses an iterative, two-pass approach:

1. Initial reconstruction: Feed the original 14-frame burst into Burstormer using the original frame 0 as the reference. Generate an initial 640 × 640 RGB reconstruction.

2. Synthetic reference generation: Convert the RGB reconstruction into an approximate 4-channel, 80 × 80 packed RGGB pseudo-RAW image. This conversion involves downsampling the RGB output, recreating the RGGB Bayer pattern, and packing the resulting mosaic into four channels.

3. Reference replacement: Replace frame 0 in a copy of the original burst with the synthetic pseudo-RAW reference. Optionally, blend the synthetic reference with the original frame 0 using a blending parameter α: reference = α × synthetic_reference + (1 − α) × original_reference

4. Iterative reconstruction: Feed the modified burst into Burstormer for a second reconstruction. The remaining 13 original frames are kept unchanged.

5. Evaluation: Compare the second reconstruction against the baseline using PSNR, SSIM, and LPIPS. Both predictions are evaluated using the original burst to maintain consistent reference-frame colour matching.

### Results
Initial experiments did not demonstrate an improvement over the baseline. In particular, increasing the synthetic-reference contribution resulted in progressively worse PSNR, SSIM, and LPIPS scores for the tested burst 0156_0036. This suggests that the approximate RGB-to-RAW conversion or the feedback mechanism may introduce errors that negatively affect reconstruction quality.

### References
- Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, Mohammad Norouzi. 2021-06-30. Image Super-Resolution via Iterative Refinement. https://searcharxiv.org/abs/2104.07636
- Bannore, V., & Swierkowski, L. (2006). An iterative approach to image SUPER-RESOLUTION. International Federation of Info Processing. https://opendl.ifip-tc6.org/db/conf/ifip12/iip2006/BannoreS06.pdf 

