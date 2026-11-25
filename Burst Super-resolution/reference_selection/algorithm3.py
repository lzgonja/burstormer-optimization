"""
Iterative Synthetic-Reference Refinement for Burstormer.

Pass 1:
    Run the original 14-frame burst with original frame 0 as reference.

Conversion:
    Convert the 640x640 RGB reconstruction into an approximate 4-channel,
    80x80 packed RGGB pseudo-RAW frame.

Pass 2:
    Replace ONLY frame 0 in a copy of the original burst with that
    pseudo-RAW frame (or a blend of it with original frame 0), then run
    Burstormer again.

Evaluation:
    Score every prediction against the ground truth using FastAlignedMetrics.
    Metrics always receive the ORIGINAL burst, so colour matching uses the
    original frame 0 for baseline and iterative predictions.

Visualization:
    Save a second, consistently processed set of images using the same
    AlignedPred + display-gain + camera-RGB-to-sRGB procedure as
    visualize_reference_frames.py.

IMPORTANT:
    Ground truth is used only for evaluation and visualization. It is NEVER
    used to create the synthetic reference fed back into Burstormer.

The RGB-to-pseudo-RAW conversion is experimental, not a physically accurate
simulation of the Samsung sensor.
"""

import argparse
import csv
import os
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

NUM_FRAMES = 14
PACKED_SIZE = 80
MOSAIC_SIZE = PACKED_SIZE * 2
FINAL_ALPHA = 0.25


def rgb_output_to_packed_rggb(
    rgb_output: torch.Tensor,
    packed_size: int = PACKED_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert [1, 3, 640, 640] RGB output into:
      - [1, 4, 80, 80] packed RGGB pseudo-RAW
      - [1, 3, 160, 160] downsampled RGB for inspection

    This is an approximation. It does not model Samsung sensor spectral
    response, black level, white balance, or camera colour matrices.
    """
    if rgb_output.ndim != 4 or rgb_output.shape[0] != 1 or rgb_output.shape[1] != 3:
        raise ValueError(
            f"Expected RGB output [1, 3, H, W], got {tuple(rgb_output.shape)}"
        )

    target_size = packed_size * 2
    rgb_small = F.interpolate(
        rgb_output,
        size=(target_size, target_size),
        mode="area",
    )

    rgb = rgb_small[0]
    mosaic = torch.empty(
        (target_size, target_size),
        dtype=rgb.dtype,
        device=rgb.device,
    )

    # RGGB Bayer layout:
    # R G
    # G B
    mosaic[0::2, 0::2] = rgb[0, 0::2, 0::2]
    mosaic[0::2, 1::2] = rgb[1, 0::2, 1::2]
    mosaic[1::2, 0::2] = rgb[1, 1::2, 0::2]
    mosaic[1::2, 1::2] = rgb[2, 1::2, 1::2]

    # Match datasets/burstsr_dataset.py::pack_raw_image.
    packed = torch.stack(
        [
            mosaic[0::2, 0::2],  # R
            mosaic[0::2, 1::2],  # G
            mosaic[1::2, 0::2],  # G
            mosaic[1::2, 1::2],  # B
        ],
        dim=0,
    ).unsqueeze(0)

    expected = (1, 4, packed_size, packed_size)
    if tuple(packed.shape) != expected:
        raise RuntimeError(
            f"Packing produced {tuple(packed.shape)}, expected {expected}"
        )

    return packed.clamp(0.0, 1.0), rgb_small.clamp(0.0, 1.0)


def build_synthetic_reference_burst(
    original_burst: torch.Tensor,
    frame0_output: torch.Tensor,
    alpha: float = FINAL_ALPHA,
) -> torch.Tensor:
    """Replace only frame 0 with a fixed blend of it and the reconstructed pseudo-RAW reference."""
    if original_burst.ndim != 5 or original_burst.shape[0] != 1 or original_burst.shape[1] != NUM_FRAMES:
        raise ValueError(f"Expected original burst [1, 14, 4, H, W], got {tuple(original_burst.shape)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    synthetic_raw, _ = rgb_output_to_packed_rggb(frame0_output, packed_size=original_burst.shape[-1])
    modified = original_burst.clone()
    modified[:, 0] = (alpha * synthetic_raw + (1.0 - alpha) * original_burst[:, 0]).clamp(0.0, 1.0)
    return modified


def save_rgb_tensor(tensor: torch.Tensor, path: str) -> None:
    """Save a raw [1,3,H,W] RGB tensor as an 8-bit PNG (not evaluation-colour-matched)."""
    image = (
        tensor.detach()[0]
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    image = np.round(image * 255.0).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def to_srgb(im: torch.Tensor, meta: dict, norm: float) -> np.ndarray:
    """
    Same display conversion used by visualize_reference_frames.py:
    linear camera RGB -> repository post-processing -> uint8 BGR.
    """
    from utils.postprocessing_functions import process_burstsr_image_rgb

    rgb = process_burstsr_image_rgb(
        im.cpu(),
        meta,
        external_norm_factor=norm,
    )
    return cv2.cvtColor(
        (rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def save_aligned_visualization(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    original_burst: torch.Tensor,
    aligner: "AlignedPred",
    meta_gt: dict,
    norm: float,
    output_path: str,
    display_gain: torch.Tensor | None,
) -> torch.Tensor:
    """
    Align and colour-match a prediction exactly as the reference-frame
    visualization does, then apply the shared per-burst display gain.

    If display_gain is None, calculate it from this first prediction
    (the baseline). All later predictions reuse that same gain.
    """
    pred_m, _, valid = aligner(prediction, labels, original_burst)

    if display_gain is None:
        # Calculate one gain per RGB channel from the baseline prediction.
        # Reuse it for every iterative prediction in this burst.
        v = valid[0].float()
        display_gain = (
            (labels[0] * v).sum((1, 2))
            / ((pred_m[0] * v).sum((1, 2)) + 1e-8)
        )

    pred_display = pred_m[0] * display_gain.view(3, 1, 1)
    cv2.imwrite(
        output_path,
        to_srgb(pred_display, meta_gt, norm),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    return display_gain


def main() -> None:
    from datasets.burstsr_dataset import BurstSRDataset
    from pwcnet.pwcnet import PWCNet
    from reference_frame_sweep import FastAlignedMetrics, load_model, select_bursts
    from utils.metrics import AlignedPred
    parser = argparse.ArgumentParser(
        description="Test iterative synthetic-reference refinement with Burstormer"
    )
    parser.add_argument(
        "--weights",
        default="./Trained_models/Real/burstormer_burstsr_train.pth",
        type=str,
    )
    parser.add_argument("--data_root", default="./burstsr_dataset", type=str)
    parser.add_argument(
        "--pwcnet",
        default="./pwcnet/pwcnet-network-default.pth",
        type=str,
    )
    parser.add_argument(
        "--bursts_file",
        default="./splits/dev.txt",
        type=str,
        help="Split file containing burst names. Use dev for experimentation, not test.",
    )
    parser.add_argument(
        "--out_dir",
        default="./Results/Real/iterative_synthetic_reference",
        type=str,
    )
    parser.add_argument(
        "--limit",
        default=1,
        type=int,
        help="Number of bursts to process. Start with 1.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
        help="Synthetic-reference blend weights. 1.0 means fully synthetic.",
    )
    args = parser.parse_args()

    if not args.alphas or any(a < 0.0 or a > 1.0 for a in args.alphas):
        raise ValueError("--alphas must contain values between 0 and 1.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Deterministic inference settings, matching reference_frame_sweep.py.
    torch.manual_seed(50)
    torch.cuda.manual_seed_all(50)
    np.random.seed(50)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script currently expects a CUDA GPU."
        )

    print("Loading Burstormer...")
    model = load_model(args.weights)

    print("Loading PWC-Net and evaluation metrics...")
    alignment_net = PWCNet(
        load_pretrained=True,
        weights_path=args.pwcnet,
    ).cuda().eval()

    # FastAlignedMetrics calculates the reported numerical metrics.
    metrics = FastAlignedMetrics(alignment_net)

    # AlignedPred reproduces the image-alignment path used by
    # visualize_reference_frames.py for saved comparison images.
    visual_aligner = AlignedPred(alignment_net=alignment_net)

    # Keep frame 0 first and preserve the original capture order.
    dataset = BurstSRDataset(
        root=args.data_root,
        split="val",
        burst_size=NUM_FRAMES,
        crop_sz=PACKED_SIZE,
        random_flip=False,
        ref_id=0,
        shuffle_frames=False,
    )

    chosen_indices = select_bursts(
        dataset.burst_list,
        args.bursts_file,
        args.limit,
    )

    csv_path = os.path.join(args.out_dir, "iterative_metrics.csv")
    fields = [
        "burst",
        "method",
        "alpha",
        "psnr",
        "ssim",
        "lpips",
        "pass1_seconds",
        "pass2_seconds",
    ]

    start_all = time.time()

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        # Index the dataset directly rather than using a DataLoader so that
        # meta_gt remains in the same unbatched form as in the visualization
        # script (visualize_reference_frames.py).
        for burst_index, dataset_index in enumerate(chosen_indices):
            burst, labels, meta_burst, meta_gt, burst_name = dataset[dataset_index]
            name = str(burst_name)

            folder = os.path.join(args.out_dir, safe_name(name))
            comparison_folder = os.path.join(folder, "comparison")
            os.makedirs(comparison_folder, exist_ok=True)

            original_burst = burst.unsqueeze(0).cuda()
            labels = labels.unsqueeze(0).cuda()

            if tuple(original_burst.shape) != (
                1,
                NUM_FRAMES,
                4,
                PACKED_SIZE,
                PACKED_SIZE,
            ):
                raise RuntimeError(
                    f"Unexpected input burst shape: {tuple(original_burst.shape)}"
                )

            # Shared display normalization, exactly as in
            # visualize_reference_frames.py.
            gt_lin = labels[0] * meta_gt["norm_factor"]
            display_norm = float(gt_lin.max())

            # Save the ground truth in the same sRGB/display pipeline as all
            # aligned predictions. Ground truth is not used by inference.
            cv2.imwrite(
                os.path.join(comparison_folder, "gt.jpg"),
                to_srgb(labels[0], meta_gt, display_norm),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            # ------------------------- PASS 1 -------------------------
            # Run the model with the original burst and original frame 0.
            t0 = time.time()
            with torch.inference_mode():
                baseline_output = model(original_burst).clamp(0.0, 1.0)
            pass1_seconds = time.time() - t0

            if tuple(baseline_output.shape) != (1, 3, 640, 640):
                raise RuntimeError(
                    "Expected model output [1,3,640,640], "
                    f"got {tuple(baseline_output.shape)}"
                )

            # Convert the direct model output to pseudo-RAW. No ground-truth
            # information is used in this conversion.
            with torch.inference_mode():
                synthetic_raw, synthetic_rgb_small = rgb_output_to_packed_rggb(
                    baseline_output
                )

            # Preserve the original unprocessed predictions for debugging.
            save_rgb_tensor(
                baseline_output,
                os.path.join(folder, "pass1_baseline.png"),
            )
            save_rgb_tensor(
                synthetic_rgb_small,
                os.path.join(folder, "synthetic_rgb_160.png"),
            )
            np.save(
                os.path.join(folder, "synthetic_packed_rggb.npy"),
                synthetic_raw[0].detach().cpu().numpy(),
            )

            packed_np = synthetic_raw[0].detach().cpu().numpy()
            plane_images = [
                np.round(np.clip(p, 0, 1) * 255).astype(np.uint8)
                for p in packed_np
            ]
            cv2.imwrite(
                os.path.join(folder, "synthetic_packed_planes.png"),
                np.concatenate(plane_images, axis=1),
            )

            # Score the baseline against the ground truth. The ORIGINAL burst
            # is supplied so the metric's reference frame remains frame 0.
            with torch.inference_mode():
                base_psnr, base_ssim, base_lpips = metrics(
                    baseline_output,
                    labels,
                    original_burst,
                )

            # Save the aligned/colour-matched baseline. Its per-channel
            # display gain becomes the shared gain for all methods in this burst.
            with torch.inference_mode():
                display_gain = save_aligned_visualization(
                    prediction=baseline_output,
                    labels=labels,
                    original_burst=original_burst,
                    aligner=visual_aligner,
                    meta_gt=meta_gt,
                    norm=display_norm,
                    output_path=os.path.join(
                        comparison_folder, "baseline.jpg"
                    ),
                    display_gain=None,
                )

            writer.writerow(
                {
                    "burst": name,
                    "method": "baseline_original_frame0",
                    "alpha": 0.0,
                    "psnr": base_psnr,
                    "ssim": base_ssim,
                    "lpips": base_lpips,
                    "pass1_seconds": pass1_seconds,
                    "pass2_seconds": "",
                }
            )
            csv_file.flush()

            print(
                f"\n[{burst_index + 1}/{len(chosen_indices)}] {name}\n"
                f"  Baseline: PSNR={base_psnr:.4f}, SSIM={base_ssim:.5f}, "
                f"LPIPS={base_lpips:.5f} ({pass1_seconds:.1f}s)"
            )

            # ------------------------- PASS 2 -------------------------
            # Replace only frame 0. The other 13 original frames are unchanged.
            for alpha in args.alphas:
                iterative_burst = build_synthetic_reference_burst(original_burst, baseline_output, alpha)

                t1 = time.time()
                with torch.inference_mode():
                    iterative_output = model(iterative_burst).clamp(0.0, 1.0)
                pass2_seconds = time.time() - t1

                # Numerical evaluation against the same ground truth, with
                # the ORIGINAL burst used for reference-frame colour matching.
                with torch.inference_mode():
                    psnr, ssim, lpips_val = metrics(
                        iterative_output,
                        labels,
                        original_burst,
                    )

                    # Visualize with the same alignment/display pipeline as
                    # the baseline. Reuse the baseline's display gain.
                    tag = f"alpha_{alpha:.2f}".replace(".", "p")
                    save_aligned_visualization(
                        prediction=iterative_output,
                        labels=labels,
                        original_burst=original_burst,
                        aligner=visual_aligner,
                        meta_gt=meta_gt,
                        norm=display_norm,
                        output_path=os.path.join(
                            comparison_folder,
                            f"iterative_{tag}.jpg",
                        ),
                        display_gain=display_gain,
                    )

                # Also preserve the raw network output, before visualization
                # alignment and colour matching, for debugging.
                save_rgb_tensor(
                    iterative_output,
                    os.path.join(folder, f"pass2_{tag}.png"),
                )

                writer.writerow(
                    {
                        "burst": name,
                        "method": "synthetic_reference_blend",
                        "alpha": alpha,
                        "psnr": psnr,
                        "ssim": ssim,
                        "lpips": lpips_val,
                        "pass1_seconds": pass1_seconds,
                        "pass2_seconds": pass2_seconds,
                    }
                )
                csv_file.flush()

                print(
                    f"  alpha={alpha:.2f}: PSNR={psnr:.4f}, "
                    f"SSIM={ssim:.5f}, LPIPS={lpips_val:.5f} "
                    f"({pass2_seconds:.1f}s)"
                )

                del iterative_burst, iterative_output
                torch.cuda.empty_cache()

            print(f"  Visual comparison saved to: {comparison_folder}")
            print(
                "  Shared per-channel display gain:",
                [round(float(x), 4) for x in display_gain],
            )

            del (
                original_burst,
                labels,
                baseline_output,
                synthetic_raw,
                synthetic_rgb_small,
            )
            torch.cuda.empty_cache()

    print(f"\nFinished in {(time.time() - start_all) / 60:.1f} min.")
    print(f"Metrics CSV: {csv_path}")
    print("Raw network outputs remain in each burst folder.")
    print("Aligned visual comparisons are in each burst's comparison/ folder.")


if __name__ == "__main__":
    main()
