# Training Burstormer on the BurstSR train set

Run of 2026-11-20 to 2026-11-21. Ticket: T02, last task ("download the BurstSR train set for a model comparable to
the paper"). Raw logs: [logs/burstsr_train_2026-11-21/](logs/burstsr_train_2026-11-21/).

**Weights:** `burstormer_burstsr_train.pth` (step 10,000, 13 MB) is attached to the GitHub release
[t02-burstsr-v1](https://github.com/lzgonja/burstormer-optimization/releases/tag/t02-burstsr-v1). Put it in
`Burst Super-resolution/Trained_models/Real/`. SHA-256:
`a9fea90f0456d87cdf988df757f654da020da1915a74940ff66d26697d069a53`.

## Setup

| | |
|---|---|
| Data | BurstSR train set, 5,405 real bursts (Samsung Galaxy S8 RAW, Canon DSLR ground truth). No scene is shared with val, so all 882 val bursts are held out |
| Start weights | `burstormer_real.pth` (the downloaded checkpoint that scores below bilinear) |
| Recipe | Track 2 loss (aligned L1 + 0.0095 x LPIPS-VGG), Adam lr 1e-5 with cosine decay to 1e-6, batch 1, 32 px crops, random flips and frame order |
| Length | 135,125 steps = 25 epochs, as in the paper. About 8 h of GPU time on an RTX 4070 Super |
| Model selection | Every 5,000 steps on 40 bursts from the 4 dev scenes of val; the best is kept |

Paper recipe differences: the paper starts from its synthetic-trained model, uses 24 px crops and gradient
clipping at 0.001. We could not use the synthetic checkpoint (different network class, untrusted pickle).

## Results on the full val set (882 bursts, 23 scenes)

Standard aligned metrics, frame 0 as reference. PSNR differences are paired per burst, 95% interval resampled over
scenes.

| Model | PSNR | SSIM | LPIPS | PSNR vs bilinear |
|---|---|---|---|---|
| Bilinear upsampling of frame 0 | 42.25 | 0.9557 | 0.2135 | |
| Downloaded checkpoint | 31.20 | 0.8493 | 0.2565 | -11.04 [-11.94, -9.90] |
| Calibrated (3k steps on 15 val scenes, earlier) | 47.70* | 0.9849 | 0.0483 | +5.45 [+5.00, +5.84] |
| **Train set, best (step 10,000)** | **47.97** | **0.9853** | 0.0507 | **+5.72 [+5.38, +6.03]** |
| Train set, final (step 135,125) | 46.47 | 0.9812 | 0.0623 | +4.23 [+3.91, +4.49] |
| Paper (PSNR as reported) | 48.82 | | | |

\* The calibrated model was trained on 15 of the 23 val scenes, so its full-val number is not a clean test.

On the 8 scenes held out from every model (dev + test, 316 bursts):

| Model | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Bilinear | 43.02 | 0.9630 | 0.2042 |
| Calibrated | 48.14 | 0.9865 | 0.0532 |
| Train set, best | 48.78 | 0.9875 | 0.0548 |
| Train set, final | 47.31 | 0.9833 | 0.0638 |

- Best minus calibrated: **+0.64 dB [+0.25, +1.10]**.
- Best minus final, all val: **+1.49 dB [+1.26, +1.75]**; the best checkpoint wins in all 23 scenes. LPIPS is also
  worse for the final checkpoint (0.0623 vs 0.0507), so the drop is not a PSNR-for-perception trade.

## What the run shows

1. **We now have a working real-data model trained the standard way.** 47.97 dB on the full val set, 5.7 dB above
   bilinear, 0.85 dB below the paper. On the 4 test scenes alone it gets 48.85 dB, but those scenes are not the
   paper's full set.
2. **Almost all of the gain comes in the first 2 epochs.** Dev PSNR peaks at step 10,000 (46.07 dB on 40 dev
   bursts) and then declines slowly to 44.8 dB at the end. Training loss falls only 7% after epoch 2
   (0.0052 to 0.0048).
3. **Longer training hurts on held-out scenes.** Every val scene gets worse from step 10,000 to the end, and so does
   LPIPS. Train loss still falls slightly, so this looks like over-fitting to the train scenes, or a train/val
   difference the loss rewards. Not yet tested.
4. **For later experiments, use `burstormer_burstsr_train.pth` (step 10,000)** in place of the calibrated model. It
   was trained without touching val, so every val scene can be used for evaluation.

Possible next steps: a shorter run (about 2-3 epochs) with evaluation every 1,000 steps to place the peak; lower
learning rate after the first epochs; check whether train-set PSNR keeps rising while val falls (over-fitting
check).

## Commands

From `Burst Super-resolution/`:

```
python scripts/download_burstsr_dataset.py .
python finetune_real.py --data_split train --train_file '' --init ./Trained_models/Real/burstormer_real.pth \
    --out ./Trained_models/Real/burstormer_burstsr_train.pth --steps 135125 --eval_every 5000 --eval_bursts 40 --resume
python evaluate_checkpoints.py --weights downloaded=./Trained_models/Real/burstormer_real.pth \
    calibrated=./Trained_models/Real/burstormer_real_calibrated.pth \
    best=./Trained_models/Real/burstormer_burstsr_train.pth final=./Trained_models/Real/burstormer_burstsr_train.pth.last \
    --out_dir ./Results/Real/burstsr_train_eval
```
