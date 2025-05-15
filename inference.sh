#!/bin/bash
 
#SBATCH --job-name=hypoc_inference
#SBATCH --partition=student,shared,sharedp
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=destc0strapp03
#SBATCH --output=slurm-logs/inference/output.log
#SBATCH --error=slurm-logs/inference/error.log

# Load conda/mamba properly for SLURM
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate hypoc

# Print GPU information
nvidia-smi

echo "Starting training..."

# Run the training script
python test.py \
    --expt_name "hypoc_inference_SL1DB_50sub2pps_SkinPatch_100f_withillum" \
    --pretrained_model_path "output/hypoc_trained_on_SkinPatch_100r100f_noIllum/weights/HSDataset/hypoc_run3_SkinPatch_100_sub_10_pps/best_epoch.pth" \
    --log_root "output/hypoc_trained_on_SkinPatch_100r100f_noIllum/test_logs" 