#!/bin/bash

# --- Slurm Job Directives ---
#SBATCH -A stf
#SBATCH --partition=ckpt-all
#SBATCH --job-name=r-zero-train     # Job name
#SBATCH --nodes=1                   # Request one node
#SBATCH --ntasks-per-node=1         # Run one task (our script) on the node
#SBATCH --gpus-per-node=h200:4           # Request 8 GPUs on that node
#SBATCH --time=3-00:00:00             # Set a 24-hour time limit (adjust as needed)
#SBATCH --output=logs/r-zero-train-%j.out  # Standard output log file
#SBATCH --error=logs/r-zero-train-%j.err   # Standard error log file

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

conda activate zero

export STORAGE_PATH="$SCRATCH/R-Zero-storage"

BASE_MODEL="Qwen/Qwen2.5-3B"
MODEL_ABBR="qwen2.5-3b"

echo "Starting R-Zero training..."
echo "Base Model: $BASE_MODEL"
echo "Model Abbreviation: $MODEL_ABBR"
echo "Storage Path: $STORAGE_PATH"

bash scripts/main.sh $BASE_MODEL $MODEL_ABBR

