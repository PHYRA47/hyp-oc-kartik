import argparse
import sys
import os
from time import gmtime, strftime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_name', type=str)
    ### PATHS ###
    parser.add_argument('--save_root', type=str, help="Weights are saved here", default="output/weights")
    parser.add_argument('--log_root', type=str, help="Training logs are saved here", default="output/logs")
    ### Training ###
    parser.add_argument('--dataset', type=str, default="HSDataset", help="Dataset name")
    parser.add_argument('--device', type=str, default="0", help="0" or "1" or "2")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size_train', type=int, default=8)
    parser.add_argument('--batch_size_val', type=int, default=128)
    parser.add_argument('--val_check_after_epoch', type=int, default=1)
    parser.add_argument('--save_for_each_val_epoch', type=bool, default=False)
    parser.add_argument('--resume', action='store_true', help="Resume training from the best_epoch.pth checkpoint") # Changed to action

    
    ### Optimizer Parameters ###
    parser.add_argument('--optim_lr', type=float, default=1e-6)
    parser.add_argument('--optim_weight_decay', type=float, default=1e-6)
    
    ### Model & Loss Parameters ###
    parser.add_argument('--feature_dimension', type=int, default=128)
    parser.add_argument('--curvature', type=float, default=0.1, help="Curvature of the hyperbolic ball")
    parser.add_argument('--std_dev', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--lambda_tpc', type=float, default=1, help="Weight for the Hyp-PC loss component")
    parser.add_argument('--clip_grad_norm', type=float, default=3.0, help="Max norm for gradient clipping")

    
    args = parser.parse_args()

    # No need to parse finetune_params anymore
    # args.fintune_params = [str(x) for x in args.finetune_params.split(',')] 
    
    return args

if __name__ == '__main__':
    args = get_args()
    print("Parsed arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")