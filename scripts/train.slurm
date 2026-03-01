#!/bin/bash

# --- Slurm Job Directives ---
#SBATCH -A stf
#SBATCH --partition=ckpt-all
#SBATCH --job-name=r-zero-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/r-zero-train-%j.out
#SBATCH --error=logs/r-zero-train-%j.err

echo "SLURM_JOB_ID:       $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

module load apptainer

# Navigate to project root
cd /mmfs1/gscratch/scrubbed/jchae3/R-Zero

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
export STORAGE_PATH="$SCRATCH/R-Zero-storage"
export HUGGINGFACENAME="HunterJ"

# Keep HuggingFace model cache on scratch to avoid filling home quota.
# Before the first run, log in once so downloads work without prompts:
#   apptainer exec --nv r-zero.sif huggingface-cli login
# Or set HF_TOKEN here:
#   export HF_TOKEN="hf_..."
export HF_HOME="$SCRATCH/.cache/huggingface"

BASE_MODEL="Qwen/Qwen2.5-3B"
MODEL_ABBR="qwen2.5-3b"

# Path to the Apptainer image. Build it first with:
#   apptainer build r-zero.sif r-zero.def
CONTAINER="$SCRATCH/r-zero.sif"

echo "Base Model:   $BASE_MODEL"
echo "Storage Path: $STORAGE_PATH"
echo "HF Cache:     $HF_HOME"
echo "Container:    $CONTAINER"

# Create required directories before entering the container
mkdir -p logs \
    "$STORAGE_PATH"/{models,generated_question,temp_results,evaluation} \
    "$HF_HOME"

# ------------------------------------------------------------------
# Launch training inside the Apptainer container
#
#   --nv            passes through the host NVIDIA drivers and GPUs
#   --bind /mmfs1   makes the project dir and all of gscratch visible
#                   at the same paths inside the container
#   --bind /dev/shm passes through host shared memory so Ray's object
#                   store is not constrained by Apptainer's default tmpfs
#
# All variables exported above are automatically inherited by the
# container (Apptainer passes the host environment by default).
# ------------------------------------------------------------------
apptainer exec \
    --nv \
    --bind /mmfs1:/mmfs1 \
    --bind /dev/shm:/dev/shm \
    "$CONTAINER" \
    bash scripts/main.sh "$BASE_MODEL" "$MODEL_ABBR"
