import os
import numpy as np
import sys
from tqdm import tqdm
from datetime import datetime
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributions
import config
from models import hyp_classifier 
from networks.HSNet import HSNet # Added HSNet
from utils.utils import save_checkpoint
import statistics # type: ignore
from loss import TPC_loss_hyp
from datasets.SkinPatchDataset import SkinPatchDataset 
from utils.preprocessing import global_contrast_normalization

import torchvision.transforms as transforms 

def train(args):
    #Params and Config
    # Simplify logging paths if dataset is fixed
    dataset_name = "HSDataset" # Hardcode or get from args if you plan to have variants
    
    if os.path.isdir(os.path.join(args.save_root, dataset_name, args.expt_name)) == False:
        os.makedirs(os.path.join(args.save_root, dataset_name, args.expt_name))
    if os.path.isdir(os.path.join(args.log_root, dataset_name)) == False:
        os.makedirs(os.path.join(args.log_root, dataset_name))
    
    log_file_path = f"{args.log_root}/{dataset_name}/{args.expt_name}_train.txt"
    # Check if file is already open (e.g. if called multiple times in a session)
    if sys.stdout.name != log_file_path :
        if hasattr(sys.stdout, 'close') and sys.stdout is not sys.__stdout__:
            sys.stdout.close()
        file_handle = open(log_file_path, "a")
        sys.stdout = file_handle
    
    print("---"*30)
    print(f"Starting training for experiment: {args.expt_name} on dataset: {dataset_name}")
    for arg in vars(args):
        num_space = 25 - len(arg)
        print(arg + " " * num_space + str(getattr(args, arg)))
    print("---"*30)
    device = "cuda:" + args.device

    # Transformations
    transform = transforms.Compose([
        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
        transforms.Normalize([-2.0743157863616943] * 31, [(3.0839202404022217 - (-2.0743157863616943))] * 31)
    ])

    PATCH_SIZE = 32
    NOISE_SCALE = 0.025
    
    trainset = SkinPatchDataset(
        num_subjects=100,
        patches_per_subject=100,
        patch_size=PATCH_SIZE,
        isRealSkin=True, 
        applyRandomIllumination=True, 
        transform=transform
    )
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
    
    valset = SkinPatchDataset(
        num_subjects=100,
        patches_per_subject=10,
        patch_size=PATCH_SIZE,
        isRealSkin=True,
        noise_scale=NOISE_SCALE,
        applyRandomIllumination=False, 
        transform=transform
    )
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4, pin_memory=True)

    #Metric Initialization
    best_APCER = 1.0
    best_NPCER = 1.0
    best_ACER = 1.0
    best_EER = 1.0 # Or 0.5 if that's the typical starting point
    best_HTER = 1.0 # Or 0.5
    best_roc_auc = 0.0
    best_threshold = 0.0
    best_accuracy_threshold = 0.0
    # Initialize metrics dictionary
    metrics = {"APCER": best_APCER, "NPCER": best_NPCER, "ACER": best_ACER, "EER": best_EER, "HTER": best_HTER, "ROC_AUC_Score": best_roc_auc, "Threshold": best_threshold, "Accuracy_threshold": best_accuracy_threshold}

    # Model Initialization
    encoder = HSNet().to(device)

    # HSNet has rep_dim = 128, this is our feature_dimension
    args.feature_dimension = encoder.rep_dim # Override or ensure this is passed correctly via config
    print(f"Using HSNet encoder with feature dimension: {args.feature_dimension}")
    
    # Ensure hyp_classifier in models.py accepts feature_dim
    model = hyp_classifier(c=args.curvature).to(device)
    
    if args.resume:
        checkpoint_path = f'{args.save_root}/{dataset_name}/{args.expt_name}/best_epoch.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            model.load_state_dict(checkpoint['classifier_state_dict'])
            # Load metrics carefully, ensure keys match
            if 'metrics' in checkpoint:
                 # Update best_ metrics from checkpoint if they are better
                if checkpoint['metrics'].get("HTER", 1.0) < best_HTER:
                    best_HTER = checkpoint['metrics']["HTER"]
                    best_APCER = checkpoint['metrics'].get("APCER",1.0)
                    best_NPCER = checkpoint['metrics'].get("NPCER",1.0)
                    best_ACER = checkpoint['metrics'].get("ACER",1.0)
                    best_EER = checkpoint['metrics'].get("EER",1.0)
                    best_roc_auc = checkpoint['metrics'].get("ROC_AUC_Score",0.0)
                metrics = checkpoint['metrics'] # Store last saved metrics

            start_epoch = checkpoint.get('epoch', 1) + 1 # Resume from next epoch
            print(f"Resumed from epoch {start_epoch-1}. Current best HTER: {best_HTER*100:.2f}%")
        else:
            print(f"Checkpoint not found at {checkpoint_path}, starting from scratch.")
    
    # Set requires_grad for all parameters of HSNet and hyp_classifier
    for p in model.parameters():
        p.requires_grad = True
    for p in encoder.parameters():
        p.requires_grad = True
    
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.optim_lr},
        {"params": filter(lambda p: p.requires_grad, encoder.parameters()), "lr": args.optim_lr_encoder if hasattr(args, 'optim_lr_encoder') else args.optim_lr}
    ]
    
    optimizer = optim.Adam(optimizer_dict, lr=args.optim_lr, betas=(0.9, 0.999), weight_decay=args.optim_weight_decay)
    
    criterion = {
        'ce_loss': nn.CrossEntropyLoss().to(device),
    }

    # Train Function
    start_epoch = 1 # Initialize start_epoch
    for num_epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        model.train()
        train_loss_sum = 0 # Use sum for accumulating loss
        num_batches = 0
        
        # Initialize mean_vector for pseudo-negative sampling
        # This should ideally be initialized once or loaded if resuming and it was saved.
        # For simplicity, re-initializing here. Consider saving/loading if EMA is critical across resumes.
        if num_epoch == start_epoch and not (args.resume and os.path.exists(f'{args.save_root}/{dataset_name}/{args.expt_name}/mean_vector.pt')):
             mean_vector = torch.zeros(args.feature_dimension, device=device)
        elif args.resume and os.path.exists(f'{args.save_root}/{dataset_name}/{args.expt_name}/mean_vector.pt'):
            mean_vector = torch.load(f'{args.save_root}/{dataset_name}/{args.expt_name}/mean_vector.pt', map_location=device)
            print("Loaded existing mean_vector for pseudo-negative sampling.")
        # else: mean_vector persists from previous epoch within the same run

        pbar = tqdm(train_dataloader, leave=True)
        for batch_idx, batch_data in enumerate(pbar):
            optimizer.zero_grad()

            images, labels, *_ = batch_data 
            images = images.to(device) 
            labels = labels.to(device) # labels are all 0 for HSDataset, used for creating classifier_ground_truth
            features = encoder(images)

            # Sample pseudo negative sample
            current_batch_mean = torch.mean(features, axis=0)
            mean_vector = args.alpha * mean_vector.detach() + (1 - args.alpha) * current_batch_mean.detach()
            
            sampler = torch.distributions.multivariate_normal.MultivariateNormal(mean_vector, args.std_dev * torch.eye(args.feature_dimension, device=device))
            noise = sampler.sample((features.shape[0],)).to(device)
            classifier_input = torch.cat([features, noise], dim=0)
            classifier_features, classifier_output = model(classifier_input)
            
            real_labels = torch.zeros(features.shape[0], device=device, dtype=torch.long)
            pseudo_negative_labels = torch.ones(noise.shape[0], device=device, dtype=torch.long)
            classifier_ground_truth = torch.cat([real_labels, pseudo_negative_labels], dim=0)

            # Loss
            tpc_loss = TPC_loss_hyp(classifier_features[:features.size(0)], c=args.curvature)
            classifier_loss = criterion['ce_loss'](classifier_output, classifier_ground_truth)
            
            loss = classifier_loss + args.lambda_tpc * tpc_loss # lambda_tpc is a hyperparameter for TPC loss
            # TPC stands
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()

            train_loss_sum += loss.item()
            num_batches += 1
            pbar.set_description(f"Epoch {num_epoch}/{args.epochs} Loss: {loss.item():.4f}", refresh=True)
        
        avg_train_loss = train_loss_sum / num_batches if num_batches > 0 else 0
        print(f"Epoch ({num_epoch}/{args.epochs}) | Average Train Loss: {avg_train_loss:.4f}")
    
        ### Validation ###
        if (num_epoch % args.val_check_after_epoch == 0):
            with torch.no_grad():
                encoder.eval()
                model.eval()
                labels_list = []
                predictions_list = []
                val_pbar = tqdm(val_dataloader, leave=True, desc=f"Validating Epoch {num_epoch}")
                for batch_val_data in val_pbar:
                    images_val, labels_val, *_ = batch_val_data
                    images_val = images_val.to(device)
                    
                    features_val = encoder(images_val)
                    classifier_features_val, classifier_output_val = model(features_val)

                    predictions = F.softmax(classifier_output_val, dim=1).cpu().numpy()[:, 1]
                    
                    # labels_val from HSDataset are all 0 (real skin)
                    labels_val_np = labels_val.cpu().numpy()
                    labels_list.extend(labels_val_np) 
                    predictions_list.extend(predictions)

                labels_list_np = np.array(labels_list)
                predictions_list_np = np.array(predictions_list)

                current_APCER, current_NPCER, current_ACER, current_EER, current_HTER, current_roc_auc, current_threshold, current_accuracy_threshold = statistics.calculate_metrics(labels_list_np, predictions_list_np)
                
                # Update metrics dictionary for logging and checkpointing
                metrics["APCER"], metrics["NPCER"], metrics["ACER"], metrics["EER"], metrics["HTER"], metrics["ROC_AUC_Score"], metrics["Threshold"], metrics["Accuracy_threshold"] = current_APCER, current_NPCER, current_ACER, current_EER, current_HTER, current_roc_auc, current_threshold, current_accuracy_threshold

                print(f"\n--- Validation Results Epoch {num_epoch} ---")
                print(f"APCER: {current_APCER*100:.2f}%") # Will be 0 or NaN if no attack samples in val
                print(f"NPCER: {current_NPCER*100:.2f}%") 
                print(f"ACER:  {current_ACER*100:.2f}%")
                print(f"HTER:  {current_HTER*100:.2f}%")
                print(f"ROC_AUC_Score: {current_roc_auc:.4f}")
                print(f"Accuracy@{current_threshold:.4f}: {current_accuracy_threshold*100.0:.2f}%")
                print("--- --- ---")

                if args.save_for_each_val_epoch: # Corrected arg name
                    filename = f"{args.save_root}/{dataset_name}/{args.expt_name}/epoch_{num_epoch}.pth"
                    save_checkpoint(num_epoch, encoder, model, metrics, filename)
                    torch.save(mean_vector, f"{args.save_root}/{dataset_name}/{args.expt_name}/mean_vector_epoch_{num_epoch}.pt")


                # Using HTER as the primary metric for saving best model, can be changed to ROC_AUC
                if current_HTER < best_HTER:
                    print(f"New best HTER: {current_HTER*100:.2f}% (previously {best_HTER*100:.2f}%)")
                    best_APCER, best_NPCER, best_ACER, best_EER, best_HTER, best_roc_auc, best_threshold, best_accuracy_threshold = current_APCER, current_NPCER, current_ACER, current_EER, current_HTER, current_roc_auc, current_threshold, current_accuracy_threshold
                    
                    # Update the main metrics dict with the new best values before saving
                    metrics["APCER"], metrics["NPCER"], metrics["ACER"], metrics["EER"], metrics["HTER"], metrics["ROC_AUC_Score"], metrics["Threshold"], metrics["Accuracy_threshold"] = best_APCER, best_NPCER, best_ACER, best_EER, best_HTER, best_roc_auc, best_threshold, best_accuracy_threshold
                    
                    filename = f"{args.save_root}/{dataset_name}/{args.expt_name}/best_epoch.pth"
                    save_checkpoint(num_epoch, encoder, model, metrics, filename)
                    torch.save(mean_vector, f"{args.save_root}/{dataset_name}/{args.expt_name}/mean_vector.pt") # Save best mean_vector


                if num_epoch == args.epochs:
                    filename = f"{args.save_root}/{dataset_name}/{args.expt_name}/last_epoch.pth"
                    save_checkpoint(num_epoch, encoder, model, metrics, filename)
                    if not os.path.exists(f"{args.save_root}/{dataset_name}/{args.expt_name}/mean_vector.pt"): # Save if best wasn't this epoch
                        torch.save(mean_vector, f"{args.save_root}/{dataset_name}/{args.expt_name}/mean_vector_last_epoch.pt")


    if sys.stdout is not sys.__stdout__: # Close the log file if it was opened
        sys.stdout.close()
        sys.stdout = sys.__stdout__ # Reset stdout to console

if __name__ == '__main__':
    args = config.get_args()
    # Add/ensure these arguments are in config.py and set appropriately for HSDataset:
    # args.skin_num_subjects_train, args.skin_patches_per_subject_train
    # args.skin_num_subjects_val, args.skin_patches_per_subject_val
    # args.skin_patch_size (should match HSNet's expectation, e.g., 32)
    # args.feature_dimension (will be set by HSNet.rep_dim, but good to have a default in config)
    # args.lambda_tpc (e.g., 0.1)
    # args.clip_grad_norm (e.g., 3.0)
    # args.optim_lr_encoder (can be same as optim_lr or different)
    # args.dataset = "HSDataset" # Can be set in config.py default or via command line
    
    # Ensure dataset name in args matches what's used for path creation if it's configurable
    # If args.dataset is used, ensure it's "HSDataset" or your chosen name.
    
    train(args)
        
    print("Parsed arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")