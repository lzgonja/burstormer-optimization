## Burst Image Restoration and Enhancement
## Akshay Dudhane, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2110.03680

import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torchvision
import random

######################################## Model and Dataset ########################################################
from Network_Real_burstSR import Base_Model
###################################################################################################################

###################################################################################################################
from torch.utils.data.dataloader import DataLoader
from datasets.burstsr_dataset import BurstSRDataset

from utils.data_format_utils import convert_dict
from utils.postprocessing_functions import BurstSRPostProcess

from pwcnet.pwcnet import PWCNet

import data_processing.camera_pipeline as rgb2raw
from data_processing.camera_pipeline import *
from reference_frame_sweep import FastAlignedMetrics
from reference_selection import (reconstruct_selected_references, reconstruction_filename,
                                 select_available_references, write_experiment_csv)
from shift_control import evaluate_reference_pair, evaluate_same_reference_pair

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(50)
##################################################################################################################


parser = argparse.ArgumentParser(description='Real burst super-resolution using Burstormer')

parser.add_argument('--result_dir', default='./Results/Real/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./Trained_models/Real/model_best.pth', type=str, help='Path to pre-trained weights')
parser.add_argument('--skip_algorithm3', action='store_true',
                    help='Run only Part 1 while iterative reference refinement is being debugged')

args = parser.parse_args()

##################################################################################################################


class BurstSR_Test_Network():

    def __init__(self, args):
        super().__init__()
        
        PWCNet_weight_PATH = './pwcnet/pwcnet-network-default.pth'        
        alignment_net = PWCNet(load_pretrained=True, weights_path=PWCNet_weight_PATH)
        alignment_net = alignment_net.cuda()
        self.alignment_net = alignment_net
        self.shift_metrics = FastAlignedMetrics(alignment_net)
        
    def test(self):               
        # Postprocessing function to obtain sRGB images
        postprocess_fn = BurstSRPostProcess(return_np=True)

        result_dir = args.result_dir + 'Testing Phase'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)     
        
        model = Base_Model()                
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.cuda() 
        
        ################## DATA Loaders ########################################
        
        # Selection indices must refer to the original RAW capture order.
        test_dataset = BurstSRDataset(root="./burstsr_dataset", split='val', burst_size=14, crop_sz=80,
                                      random_flip=False, ref_id=0, shuffle_frames=False)
        test_data_loader = DataLoader(test_dataset, batch_size=1)
        
        experiment_rows = []
        csv_path = os.path.join(result_dir, 'per_burst.csv')
        
        for i, data in enumerate(test_data_loader):

            burst, labels, meta_info_burst, meta_info_gt, burst_name = data            
            meta_info_burst = convert_dict(meta_info_burst, burst.shape[0])
            
            # Part 1: adaptive selection among original RAW reference frames.
            selections = select_available_references(burst)
            burst = burst.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                outputs = reconstruct_selected_references(burst, model, selections)
                outputs = {reference: output.clamp(0.0, 1.0) for reference, output in outputs.items()}

            gain_by_ref = {}
            for method, selected_ref in selections.items():
                output = outputs[selected_ref]
                if selected_ref not in gain_by_ref:
                    gain_by_ref[selected_ref] = evaluate_reference_pair(
                        outputs, selected_ref, burst, labels, self.shift_metrics, self.alignment_net)
                corrected_gain = gain_by_ref[selected_ref]
                experiment_rows.append({'burst_id': burst_name[0], 'selection_method': method,
                                        'experiment_type': 'adaptive_reference_selection',
                                        'selected_reference_index': selected_ref,
                                        'psnr_gain_vs_frame0': corrected_gain,
                                        'ssim_gain_vs_frame0': '', 'lpips_improvement_vs_frame0': '', 'alpha': '',
                                        'evaluation_method': 'v2_reverse_shift_control'})
                net_pred_np = (output[0].permute(1, 2, 0) * 2 ** 14).cpu().numpy().astype(np.uint16)
                cv2.imwrite(os.path.join(result_dir, reconstruction_filename(burst_name[0], method)), net_pred_np)
                print(f'Burst {i}, {method}, reference {selected_ref}: '
                      f'v2 corrected PSNR gain {corrected_gain:+.3f} dB')

            # Persist Part 1 before the optional/experimental Part 2 path.
            write_experiment_csv(csv_path, experiment_rows)

            # Part 2: iterative refinement in the original frame-0 viewpoint.
            if not args.skip_algorithm3:
                try:
                    from reference_selection.algorithm3 import FINAL_ALPHA, build_synthetic_reference_burst

                    with torch.no_grad():
                        algorithm3_burst = build_synthetic_reference_burst(burst, outputs[0], FINAL_ALPHA)
                        algorithm3_output = model(algorithm3_burst).clamp(0.0, 1.0)
                        algorithm3_gains = evaluate_same_reference_pair(
                            outputs[0], algorithm3_output, burst, labels, self.shift_metrics)
                    experiment_rows.append({'burst_id': burst_name[0], 'selection_method': 'algorithm3',
                                            'experiment_type': 'iterative_reference_refinement',
                                            'selected_reference_index': '',
                                            **algorithm3_gains, 'alpha': FINAL_ALPHA,
                                            'evaluation_method': 'same_reference_pair'})
                    net_pred_np = (algorithm3_output[0].permute(1, 2, 0) * 2 ** 14).cpu().numpy().astype(np.uint16)
                    cv2.imwrite(os.path.join(result_dir, reconstruction_filename(burst_name[0], 'algorithm3')),
                                net_pred_np)
                    print(f'Burst {i}, algorithm3, synthetic reference alpha {FINAL_ALPHA}: '
                          f'PSNR gain {algorithm3_gains["psnr_gain_vs_frame0"]:+.3f} dB, '
                          f'SSIM gain {algorithm3_gains["ssim_gain_vs_frame0"]:+.5f}, '
                          f'LPIPS improvement {algorithm3_gains["lpips_improvement_vs_frame0"]:+.5f}')
                    write_experiment_csv(csv_path, experiment_rows)
                except Exception as error:
                    print(f'Burst {i}, algorithm3 failed; Part 1 results were preserved: {error}')
            """
            burst = burst.cpu()
            output = output.cpu()
            labels = labels.cpu()
            
            meta_info_gt = convert_dict(meta_info_gt, labels.shape[0])

            # Apply simple post-processing to obtain RGB images
            input_burst = postprocess_fn.process(burst_rgb[0], meta_info_burst[0])
            output = postprocess_fn.process(output[0], meta_info_gt[0])
            labels = postprocess_fn.process(labels[0], meta_info_gt[0])
            
            input_burst = cv2.cvtColor(input_burst, cv2.COLOR_RGB2BGR)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            labels = cv2.cvtColor(labels, cv2.COLOR_RGB2BGR)

            #output = np.concatenate((input_burst, output, labels), axis=1)
            
            cv2.imwrite('{}/{}'.format(result_dir, burst_name[0] +'.png'), output)
            """
        
        write_experiment_csv(csv_path, experiment_rows)

if __name__ == '__main__':
    BurstSR_Test_Network(args).test()
