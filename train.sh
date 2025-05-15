#!/bin/bash
 
#SBATCH --job-name=hypoc_training
#SBATCH --partition=student,shared,sharedp
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=destc0strapp03
#SBATCH --output=slurm-logs/training/output.log
#SBATCH --error=slurm-logs/training/error.log

# Load conda/mamba properly for SLURM
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate hypoc

# Print GPU information
nvidia-smi

echo "Starting training..."

# Run the training script
python train.py \
    --expt_name "hypoc_SkinPatch_100r100f_run3" \
    --save_root "output/run3/weights" \
    --log_root "output/run3/logs" \
    --epochs 10 \
    --batch_size_train 32 \
    --batch_size_val 32 \