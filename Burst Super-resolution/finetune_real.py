"""Short fine-tune of Burstormer on real BurstSR bursts with the Track 2 loss (aligned L1 + LPIPS).

Why this exists: the available real-data checkpoint produces output in a different colour space from its input, so
the aligned metrics (which colour-match using the input reference frame) score it below bilinear upsampling. A short
fine-tune with the repo's own loss pulls the output back into the input's colour space. The same script is the base
for later experiments (noise augmentation, random reference frame).

Two data setups:
- Val only (default): bursts are split by scene (./splits), training on the train scenes and evaluating on scenes the
  model never sees. Numbers from such a model are comparable with each other, not with the paper.
- BurstSR train set (--data_split train --train_file ''): every train burst is used and the whole val set stays
  held out, as in the paper. Download it with scripts/download_burstsr_dataset.py.

Long runs write <out>.last (model, optimizer, scheduler, step) at every evaluation; --resume continues from it.

Examples:
    python finetune_real.py --init ./Trained_models/Real/burstormer_real.pth \
        --out ./Trained_models/Real/burstormer_real_calibrated.pth --steps 3000
    python finetune_real.py --data_split train --train_file '' --init ./Trained_models/Real/burstormer_real.pth \
        --out ./Trained_models/Real/burstormer_burstsr_train.pth --steps 100000 --eval_every 5000 --eval_bursts 40
"""

import argparse
import math
import os
import random
import time
from typing import List, Tuple

import lpips
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader

from Network_Real_burstSR import Base_Model
from datasets.burstsr_dataset import BurstSRDataset
from pwcnet.pwcnet import PWCNet
from reference_frame_sweep import (NUM_FRAMES, FastAlignedMetrics, bilinear_reference, load_model, select_bursts,
                                   set_seed)
from robustness_sweep import degrade
from utils.metrics import AlignedPred


def evaluate(model: torch.nn.Module, metrics: FastAlignedMetrics, bursts: List[Tuple]) -> float:
    model.eval()
    psnrs = []
    with torch.no_grad():
        for burst, gt in bursts:
            psnrs.append(metrics(model(burst).clamp(0.0, 1.0), gt, burst)[0])
    model.train()
    return float(np.mean(psnrs))


def main() -> None:
    parser = argparse.ArgumentParser(description='Fine-tune Burstormer on real BurstSR bursts')
    parser.add_argument('--init', default='./Trained_models/Real/burstormer_real.pth', type=str,
                        help="Starting weights (.pth or Lightning .ckpt), or 'scratch' for random initialisation")
    parser.add_argument('--out', default='./Trained_models/Real/burstormer_real_calibrated.pth', type=str)
    parser.add_argument('--data_root', default='./burstsr_dataset', type=str)
    parser.add_argument('--data_split', default='val', type=str, help="Dataset folder to read: 'val' or 'train'")
    parser.add_argument('--train_file', default='./splits/train.txt', type=str,
                        help="Burst names to train on; '' trains on every burst in --data_split")
    parser.add_argument('--eval_file', default='./splits/dev.txt', type=str, help='Burst names for periodic evaluation')
    parser.add_argument('--pwcnet', default='./pwcnet/pwcnet-network-default.pth', type=str)
    parser.add_argument('--steps', default=3000, type=int)
    parser.add_argument('--lr', default=1e-5, type=float, help='Track 2 training used 1e-5')
    parser.add_argument('--crop', default=32, type=int, help='Training crop in packed raw pixels (Track 2 used 24)')
    parser.add_argument('--lpips_weight', default=0.0095, type=float, help='Track 2 training used 0.0095 (VGG)')
    parser.add_argument('--noise_max', default=0.0, type=float,
                        help='Noise augmentation: with probability --noise_prob, add shot + read noise to the input '
                             'burst with a shot level drawn log-uniformly from [1e-4, noise_max] (0 = off). '
                             'Same noise model and level scale as robustness_sweep.py')
    parser.add_argument('--noise_prob', default=0.5, type=float)
    parser.add_argument('--blur_max', default=0.0, type=float,
                        help='Blur augmentation: with probability --blur_prob, blur every input frame with a Gaussian '
                             'whose sigma is drawn uniformly from [0.3, blur_max] packed raw pixels (0 = off). '
                             'Same blur as robustness_sweep.py. Applied before the noise, as in a camera')
    parser.add_argument('--blur_prob', default=0.5, type=float)
    parser.add_argument('--eval_every', default=500, type=int)
    parser.add_argument('--eval_bursts', default=12, type=int, help='Eval bursts, spread across the eval scenes')
    parser.add_argument('--workers', default=3, type=int)
    parser.add_argument('--resume', action='store_true', help='Continue from <out>.last if it exists')
    args = parser.parse_args()
    if args.data_split == 'val' and not args.train_file:
        parser.error("--train_file '' with --data_split val would train on the evaluation bursts")

    set_seed(0)
    alignment_net = PWCNet(load_pretrained=True, weights_path=args.pwcnet).cuda()
    for p in alignment_net.parameters():
        p.requires_grad_(False)
    aligner = AlignedPred(alignment_net=alignment_net)
    metrics = FastAlignedMetrics(alignment_net)
    lpips_vgg = lpips.LPIPS(net='vgg', verbose=False).cuda().eval() if args.lpips_weight > 0 else None

    model = (Base_Model().cuda() if args.init == 'scratch' else load_model(args.init)).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr / 10)
    last_path = args.out + '.last'
    resumed = None
    if args.resume and os.path.isfile(last_path):
        resumed = torch.load(last_path, map_location='cuda', weights_only=False)
        model.load_state_dict(resumed['model_state_dict'])
        optimizer.load_state_dict(resumed['optimizer_state_dict'])
        scheduler.load_state_dict(resumed['scheduler_state_dict'])
        print(f"Resumed from {last_path} at step {resumed['step']}", flush=True)

    # Training data: random crops, flips and frame order, as in Burstormer_Track_2_training.py
    train_set = BurstSRDataset(root=args.data_root, split=args.data_split, burst_size=NUM_FRAMES, crop_sz=args.crop,
                               random_flip=True, ref_id=0, shuffle_frames=True)
    train_set = Subset(train_set, select_bursts(train_set.burst_list, args.train_file or None))
    loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)

    # Fixed evaluation bursts from scenes that are not trained on, kept on the GPU
    eval_set = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                              random_flip=False, ref_id=0, shuffle_frames=False)
    eval_bursts = []
    for i in select_bursts(eval_set.burst_list, args.eval_file, args.eval_bursts):
        burst, gt, _, _, _ = eval_set[i]
        eval_bursts.append((burst[None].cuda(), gt[None].cuda()))
    with torch.no_grad():
        bilinear = float(np.mean([metrics(bilinear_reference(b), g, b)[0] for b, g in eval_bursts]))
    best = evaluate(model, metrics, eval_bursts) if resumed is None else resumed['best']
    print(f'{len(train_set)} training bursts, {len(eval_bursts)} eval bursts. '
          f'PSNR {"before fine-tuning" if resumed is None else "best so far"} {best:.2f} dB, '
          f'bilinear baseline {bilinear:.2f} dB', flush=True)

    noise_gen = torch.Generator(device='cuda')
    noise_gen.manual_seed(0)
    step = 0 if resumed is None else resumed['step']
    first_step, running, start = step, [], time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    while step < args.steps:
        for burst, gt, _, _, _ in loader:
            burst, gt = burst.cuda(), gt.cuda()
            model_input = burst
            if args.blur_max > 0 and random.random() < args.blur_prob:
                model_input = degrade(model_input, 'blur', random.uniform(0.3, args.blur_max), noise_gen)
            if args.noise_max > 0 and random.random() < args.noise_prob:
                shot = math.exp(random.uniform(math.log(1e-4), math.log(args.noise_max)))
                model_input = degrade(model_input, 'noise', shot, noise_gen)
            pred = model(model_input)
            # Same alignment + colour matching as the evaluation metric; gradients flow through the warp. The colour
            # transform is always fitted on the clean burst, so the target does not depend on the injected noise/blur.
            pred_m, gt_c, valid = aligner(pred, gt, burst)
            validf = valid.float()
            if validf.sum() < 1:
                continue
            loss = ((pred_m - gt_c).abs() * validf).sum() / (validf.sum() * pred_m.shape[1])
            if lpips_vgg is not None:
                loss = loss + args.lpips_weight * lpips_vgg(2 * pred_m - 1, 2 * gt_c - 1).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            running.append(loss.item())

            if step % 50 == 0:
                print(f'step {step}/{args.steps}  loss {np.mean(running[-50:]):.5f}  '
                      f'{(time.time() - start) / (step - first_step):.2f}s/step', flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                psnr = evaluate(model, metrics, eval_bursts)
                note = ''
                if psnr > best:
                    best = psnr
                    torch.save({'epoch': step, 'model_state_dict': model.state_dict()}, args.out)
                    note = f'  -> saved {args.out}'
                print(f'[eval] step {step}: PSNR {psnr:.2f} dB (bilinear {bilinear:.2f}, best {best:.2f}){note}',
                      flush=True)
                torch.save({'step': step, 'best': best, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()}, last_path + '.tmp')
                os.replace(last_path + '.tmp', last_path)
            if step >= args.steps:
                break
    print(f'Done. Best eval PSNR {best:.2f} dB vs bilinear {bilinear:.2f} dB')


if __name__ == '__main__':
    main()
