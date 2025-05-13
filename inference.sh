#!/bin/bash
 
#SBATCH --job-name=hypoc_training
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
    --expt_name "hypoc_run2_test" \
    --pretrained_model_path "/home/denegasf/repo/negasa-fromsa-teshome-msc-thesis/src/hyp-oc-kartik/output/run2/weights/HSDataset/hypoc_run2/best_epoch.pth" \
    --save_root "output/run2/weights" \
    --log_root "output/run2/test_logs" \