import os
import cv2
import numpy as np
import torch

from utils.metrics import AlignedPred
from utils.postprocessing_functions import process_burstsr_image_rgb

# def save_comparison(outputs, selections, burst_name, meta_info_gt, alignment_net, output_dir):
#     gt = outputs["gt"]
#     algorithm3 = outputs["algorithm3"]

#     frame = []

#     for method, selected_ref in selections.items():
#         frame.append(selected_ref)

#     baseline = outputs[frame[0]]
#     algo1 = outputs[frame[1]]
#     algo2 = outputs[frame[2]]

import os
import re

import cv2
import numpy as np
import torch

from reference_frame_sweep import reorder
from utils.metrics import AlignedPred
from utils.postprocessing_functions import process_burstsr_image_rgb


def safe_name(name):
    """Make a method or burst name safe for use as a filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def to_srgb(image, meta, norm):
    """
    Convert linear camera RGB [3,H,W] into an 8-bit BGR image.
    Uses the same camera-processing pipeline as visualize_reference_frames.py.
    """
    rgb = process_burstsr_image_rgb(
        image.detach().cpu(),
        meta,
        external_norm_factor=norm
    )

    rgb = (
        rgb.permute(1, 2, 0)
        .numpy() * 255
    )

    rgb = np.round(rgb).clip(0, 255).astype(np.uint8)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_comparison(outputs, selections, burst_name,
                    meta_info_gt, alignment_net, burst, output_dir):
    """
    Save individual reconstructions and a labelled comparison montage.

    outputs:
        Dictionary containing reference-indexed reconstructions,
        Algorithm 3, and ground truth.

    selections:
        Dictionary mapping selection method names to reference indices.

    burst:
        Original burst in its original frame-0 ordering.
    """

    # ---------------------------------------------------------
    # STEP 1: Set up output folder and retrieve tensors
    # ---------------------------------------------------------

    burst_dir = os.path.join(output_dir, safe_name(burst_name))
    os.makedirs(burst_dir, exist_ok=True)

    gt = outputs["gt"]
    baseline = outputs[0]

    # Create the alignment helper using the existing PWCNet.
    aligner = AlignedPred(alignment_net=alignment_net)

    # Shared display normalization, following visualize_reference_frames.py.
    norm = float((gt[0] * meta_info_gt["norm_factor"]).max())

    # Save ground truth.
    gt_image = to_srgb(gt[0], meta_info_gt, norm)

    cv2.imwrite(
        os.path.join(burst_dir, "gt.jpg"),
        gt_image,
        [cv2.IMWRITE_JPEG_QUALITY, 92]
    )

    # ---------------------------------------------------------
    # STEP 2: Align baseline and calculate shared display gain
    # ---------------------------------------------------------

    baseline_aligned, _, valid = aligner(baseline, gt, burst)

    # Calculate the per-channel display gain using the baseline.
    # Every other reconstruction will use this exact same gain.
    v = valid[0].float()

    display_gain = (
        (gt[0] * v).sum((1, 2))
        / ((baseline_aligned[0] * v).sum((1, 2)) + 1e-8)
    )

    baseline_display = baseline_aligned[0] * display_gain.view(3, 1, 1)

    baseline_image = to_srgb(
        baseline_display,
        meta_info_gt,
        norm
    )

    # cv2.imwrite(
    #     os.path.join(burst_dir, "baseline_ref00.jpg"),
    #     baseline_image,
    #     [cv2.IMWRITE_JPEG_QUALITY, 92]
    # )

    # Save the baseline reconstruction separately.
    cv2.imwrite(
        os.path.join(burst_dir, "baseline_ref00.jpg"),
        baseline_image,
        [cv2.IMWRITE_JPEG_QUALITY, 92]
    )

    # Include the baseline exactly once in the comparison montage.
    comparison_images = [
        ("Ground Truth", gt_image),
        ("Baseline (Ref 00)", baseline_image)
    ]

    # Frame 0 is already represented by the baseline.
    aligned_by_ref = {
        0: baseline_image
    }

    # ---------------------------------------------------------
    # STEP 3: Save each reference-selection method
    # ---------------------------------------------------------

    for method, selected_ref in selections.items():


        if method == "frame0":
            print(f"Skipping {method}: selected frame 0 (baseline already included).")
            continue

        # Retrieve the reconstruction associated with this reference.
        prediction = outputs[selected_ref]

        # Reuse an existing aligned image if another method selected
        # exactly the same reference frame.
        if selected_ref in aligned_by_ref:
            method_image = aligned_by_ref[selected_ref]

        else:
            # Reorder the original burst so the selected reference
            # is in position 0, matching the model's inference setup.
            reference_burst = reorder(burst, selected_ref)

            aligned, _, _ = aligner(
                prediction,
                gt,
                reference_burst
            )

            prediction_display = (
                aligned[0] * display_gain.view(3, 1, 1)
            )

            method_image = to_srgb(
                prediction_display,
                meta_info_gt,
                norm
            )

            aligned_by_ref[selected_ref] = method_image

        # Save one file per METHOD, not per reference index.
        filename = safe_name(method) + ".jpg"

        cv2.imwrite(
            os.path.join(burst_dir, filename),
            method_image,
            [cv2.IMWRITE_JPEG_QUALITY, 92]
        )

        comparison_images.append(
            (str(method), method_image)
        )

    # ---------------------------------------------------------
    # STEP 4: Save Algorithm 3
    # ---------------------------------------------------------

    if "algorithm3" in outputs:

        algorithm3 = outputs["algorithm3"]

        # Algorithm 3 uses the original frame-0 viewpoint.
        # Therefore, align against the original burst.
        algorithm3_aligned, _, _ = aligner(
            algorithm3,
            gt,
            burst
        )

        algorithm3_display = (
            algorithm3_aligned[0] * display_gain.view(3, 1, 1)
        )

        algorithm3_image = to_srgb(
            algorithm3_display,
            meta_info_gt,
            norm
        )

        cv2.imwrite(
            os.path.join(burst_dir, "algorithm3.jpg"),
            algorithm3_image,
            [cv2.IMWRITE_JPEG_QUALITY, 92]
        )

        comparison_images.append(
            ("Algorithm 3", algorithm3_image)
        )

    # ---------------------------------------------------------
    # STEP 5: Create labelled comparison montage
    # ---------------------------------------------------------

    panels = []

    for label, image in comparison_images:

        # Add a header above each image.
        header = np.full(
            (45, image.shape[1], 3),
            255,
            dtype=np.uint8
        )

        cv2.putText(
            header,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        panel = np.concatenate(
            [header, image],
            axis=0
        )

        panels.append(panel)

    comparison = np.concatenate(panels, axis=1)

    cv2.imwrite(
        os.path.join(burst_dir, "comparison.jpg"),
        comparison,
        [cv2.IMWRITE_JPEG_QUALITY, 92]
    )

    print(f"Saved visualizations for {burst_name} to {burst_dir}")