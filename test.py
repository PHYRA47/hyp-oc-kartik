import argparse
import os
import numpy as np
# import cv2 # Not used
import sys
import torch.utils
from tqdm import tqdm
# import argparse # Handled by config.py
from datetime import datetime
# import time # Not explicitly used
import torch
import torchvision.transforms as transforms # Keep if SkinPatchDataset uses it
import torch.nn.functional as F
# import torch.nn as nn # Not directly used for model definition here, but F is
# import torch.optim as optim # Not used in test
# from torch.optim import lr_scheduler # Not used in test
from torch.utils.data import DataLoader
# import torch.distributions # Not used in test

import config # Use the same config as train.py
from models import hyp_classifier # Vgg_face_dag, load_vgg_face removed
from networks.HSNet import HSNet # Import HSNet
# from utils.utils import save_checkpoint # Not used in test
from datasets.SkinPatchDataset import SkinPatchDataset # Assuming it's in a 'datasets' subfolder
from utils.preprocessing import global_contrast_normalization # If used by transforms
import statistics # type: ignore
# from loss import TPC_loss_hyp # Not used in test

def test(args):
    # Params and Config
    dataset_name = args.dataset # Should be "HSDataset" from config
    expt_name = args.expt_name

    # Ensure log directory exists
    log_dir = os.path.join(args.log_root, dataset_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    
    log_file_path = f"{log_dir}/{expt_name}_test.txt"
    
    # Redirect stdout to log file
    # Check if file is already open (e.g. if called multiple times in a session)
    original_stdout = sys.stdout
    if hasattr(sys.stdout, 'name') and sys.stdout.name == log_file_path:
        pass # Already logging to the correct file
    else:
        if hasattr(sys.stdout, 'close') and sys.stdout is not sys.__stdout__:
            sys.stdout.close()
        file_handle = open(log_file_path, "a")
        sys.stdout = file_handle
    
    print("---"*30)
    print(f"Starting testing for experiment: {expt_name} on dataset: {dataset_name}")
    for arg_name in vars(args):
        num_space = 25 - len(arg_name)
        print(arg_name + " " * num_space + str(getattr(args, arg_name)))
    print("---"*30)
    device = "cuda:" + args.device

    # Dataloaders for HSDataset
    # Ensure your SkinPatchDataset is configured to output (batch, 31, 32, 32) tensors
    # Use the same transform logic as in train.py for consistency, or a simplified one for testing
    test_transform = transforms.Compose([
        # Add transforms.ToTensor() if SkinPatchDataset outputs PIL Images or numpy arrays
        # transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')), # Verify this works for (C,H,W)
        # transforms.Normalize(channel_means, channel_stds) # Apply per-channel normalization
        # Using your provided transform from train.py:
        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
        transforms.Normalize([-2.0743157863616943] * 31, [(3.0839202404022217 - (-2.0743157863616943))] * 31)
    ])

    # Use test-specific arguments for SkinPatchDataset from config.py
    testset = torch.utils.data.ConcatDataset([
        SkinPatchDataset(
            num_subjects=100, 
            patches_per_subject=10,
            patch_size=32,
            isRealSkin=True, 
            applyRandomIllumination=True, 
            transform=test_transform
        ),
        SkinPatchDataset(
            num_subjects=100, 
            patches_per_subject=10,
            patch_size=32,
            isRealSkin=False, 
            applyRandomIllumination=True, 
            transform=test_transform
        )
    ])
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)


    # Model Initialization
    encoder = HSNet().to(device)
    # HSNet has rep_dim = 128, this is our feature_dimension
    # args.feature_dimension should be set from config (default 128)
    print(f"Using HSNet encoder with feature dimension: {encoder.rep_dim}")
    
    model = hyp_classifier(c=args.curvature).to(device)
    
    # Load checkpoint
    checkpoint_filename = None
    if args.pretrained_model_path:
        if os.path.exists(args.pretrained_model_path):
            checkpoint_filename = args.pretrained_model_path
            print(f"Attempting to load manually specified checkpoint: {checkpoint_filename}")
        else:
            print(f"ERROR: Manually specified checkpoint file not found at {args.pretrained_model_path}")
            if sys.stdout is not original_stdout:
                sys.stdout.close()
            sys.stdout = original_stdout # Reset stdout
            return
    else:
        # The checkpoint path should correspond to the model trained by train.py
        constructed_checkpoint_filename = f"{args.save_root}/{dataset_name}/{expt_name}/best_epoch.pth" # Or last_epoch.pth
        if os.path.exists(constructed_checkpoint_filename):
            checkpoint_filename = constructed_checkpoint_filename
            print(f"Attempting to load checkpoint based on experiment name: {checkpoint_filename}")
        else:
            # Fallback to try last_epoch.pth if best_epoch.pth is not found
            constructed_checkpoint_filename = f"{args.save_root}/{dataset_name}/{expt_name}/last_epoch.pth"
            if os.path.exists(constructed_checkpoint_filename):
                checkpoint_filename = constructed_checkpoint_filename
                print(f"Attempting to load checkpoint based on experiment name (last_epoch): {checkpoint_filename}")


    if not checkpoint_filename or not os.path.exists(checkpoint_filename):
        print(f"ERROR: Checkpoint file not found.")
        print(f"  Tried manual path: {args.pretrained_model_path if args.pretrained_model_path else 'Not provided'}")
        print(f"  Tried constructed path: {args.save_root}/{dataset_name}/{expt_name}/best_epoch.pth (and last_epoch.pth)")
        if sys.stdout is not original_stdout:
            sys.stdout.close()
        sys.stdout = original_stdout # Reset stdout
        return

    print(f"Loading checkpoint from: {checkpoint_filename}")
    
    with torch.no_grad():
        encoder.eval()
        model.eval()
        labels_list = []
        predictions_list = []
        
        test_pbar = tqdm(test_dataloader, leave=True, desc=f"Testing {expt_name}")
        for batch_data in test_pbar:
            images, labels, *_ = batch_data # Unpack
            images = images.to(device)
            # labels from HSDataset are all 0 (real skin)
            
            features = encoder(images)
            # For testing, we only need the classifier's output, not intermediate features usually
            _, classifier_output = model(features) # classifier_features might not be needed

            # Predictions: Softmax output, index 1 is the score for the "pseudo-negative" class (anomaly score)
            predictions = F.softmax(classifier_output, dim=1).cpu().numpy()[:, 1]
            
            labels_np = labels.cpu().numpy()
            labels_list.extend(labels_np)
            predictions_list.extend(predictions)

        labels_list_np = np.array(labels_list)
        predictions_list_np = np.array(predictions_list)

        # Calculate Metrics
        # The calculate_metrics function expects labels (0 for normal, 1 for attack)
        # and predictions (scores for attack class).
        # Since your test labels_list is all 0s (real skin), it will primarily calculate NPCER.
        APCER, NPCER, ACER, EER, HTER, roc_auc, threshold, accuracy_threshold = statistics.calculate_metrics(labels_list_np, predictions_list_np)
        
        metrics = {
            "APCER": APCER, "NPCER": NPCER, "ACER": ACER, 
            "EER": EER, "HTER": HTER, "ROC_AUC_Score": roc_auc, 
            "Threshold": threshold, "Accuracy_threshold": accuracy_threshold
        }

        # Print results
        print(f"\n##### TEST SET RESULTS for {expt_name} #####")
        print(f"APCER: {metrics['APCER']*100:.2f}%") # Will be 0 or NaN if no attack samples in test
        print(f"NPCER: {metrics['NPCER']*100:.2f}%") 
        print(f"ACER:  {metrics['ACER']*100:.2f}%")
        print(f"HTER:  {metrics['HTER']*100:.2f}%") # HTER = (APCER + NPCER) / 2
        print(f"EER: {metrics['EER']*100:.2f}%")
        print(f"ROC_AUC_Score: {metrics['ROC_AUC_Score']:.4f}")
        print(f"Optimal Threshold for EER/HTER: {metrics['Threshold']:.4f}")
        print(f"Accuracy at Optimal Threshold: {metrics['Accuracy_threshold']*100.0:.2f}%")
        print("--- --- ---")
        
    if sys.stdout is not original_stdout: # Close the log file if it was opened
        sys.stdout.close()
        sys.stdout = original_stdout # Reset stdout to console

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Hyp-OC Testing")
    
    parser.add_argument('--pretrained_model_path', type=str, help="Path to pretrained VGG model (unused by HSNet)", default=None) # Will be parsed but not used by HSNet logic

    parser.add_argument('--expt_name', type=str, help="Experiment name for logging", default="test_experiment")
    parser.add_argument('--save_root', type=str, default="./output/weights", help="Root directory where trained weights are saved")
    parser.add_argument('--log_root', type=str, default="./output/results", help="Root directory where test logs will be saved")
    parser.add_argument('--dataset', type=str, default="HSDataset", help="Dataset name (used for path organization, fixed to HSDataset functionality)")
    parser.add_argument('--device', type=str, default="0", help="CUDA device ID (e.g., '0', '1')")
    parser.add_argument('--batch_size_test', type=int, default=32, help="Batch size for testing")
    parser.add_argument('--curvature', type=float, default=1.0, help="Curvature of the hyperbolic ball (c > 0)")
    
    args = parser.parse_args()
    test(args)