
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

## Algorithm 2: TBD

**Contributor:** Stanley Cao

### Motivation
Why might this approach identify a better reference frame?

### Method
Describe how the algorithm works and how it selects a frame.

### Implementation
Describe the implementation and any important parameters.

### References
- TBD

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

