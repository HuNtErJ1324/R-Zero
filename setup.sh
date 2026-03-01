#!/bin/bash
# =============================================================
#  R-Zero Full Setup Script
#  Sets up the entire environment, directories, and API keys
#  needed to run: bash scripts/main.sh <model> <abbr>
#
#  Usage:
#    bash setup.sh
#
#  You will be prompted to enter your API keys interactively.
#  To skip prompts (e.g. in CI), set these env vars beforehand:
#    HF_TOKEN, WANDB_API_KEY, STORAGE_PATH, HUGGINGFACENAME
# =============================================================

set -e  # Exit on any error

ENV_NAME="zero"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  R-Zero Full Environment Setup"
echo "============================================"
echo ""

# =============================================================
# Step 1: Gather user configuration
# =============================================================
echo "[Step 1/5] Gathering configuration..."
echo ""

# --- STORAGE_PATH ---
if [ -z "$STORAGE_PATH" ]; then
    default_storage="$SCRATCH/R-Zero-storage"
    read -rp "  Storage path for checkpoints/data [$default_storage]: " input_storage
    STORAGE_PATH="${input_storage:-$default_storage}"
fi
echo "  STORAGE_PATH = $STORAGE_PATH"

# --- HUGGINGFACENAME ---
if [ -z "$HUGGINGFACENAME" ]; then
    read -rp "  Your HuggingFace username (for dataset uploads): " HUGGINGFACENAME
fi
echo "  HUGGINGFACENAME = $HUGGINGFACENAME"

# --- HF_TOKEN ---
if [ -z "$HF_TOKEN" ]; then
    read -rp "  HuggingFace API token: " HF_TOKEN
fi
echo "  HF_TOKEN = [set]"

# --- WANDB_API_KEY ---
if [ -z "$WANDB_API_KEY" ]; then
    read -rp "  WandB API key: " WANDB_API_KEY
fi
echo "  WANDB_API_KEY = [set]"

echo ""

# =============================================================
# Step 2: Create storage directories
# =============================================================
echo "[Step 2/5] Creating storage directories..."

mkdir -p \
  "$STORAGE_PATH/evaluation" \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results" \
  logs

echo "  Created: $STORAGE_PATH/evaluation"
echo "  Created: $STORAGE_PATH/models"
echo "  Created: $STORAGE_PATH/generated_question"
echo "  Created: $STORAGE_PATH/temp_results"
echo "  Created: logs/"
echo ""

# =============================================================
# Step 3: Write tokens.json
# =============================================================
echo "[Step 3/5] Writing tokens.json..."

cat > tokens.json <<EOF
{
    "huggingface": "$HF_TOKEN",
    "wandb": "$WANDB_API_KEY"
}
EOF

echo "  tokens.json written successfully."
echo ""

# =============================================================
# Step 4: Create conda environment and install dependencies
# =============================================================
echo "[Step 4/5] Setting up Python environment '$ENV_NAME'..."
echo ""

# Check if conda env already exists
if conda info --envs | grep -qw "$ENV_NAME"; then
    echo "  Conda environment '$ENV_NAME' already exists."
    read -rp "  Recreate it from scratch? [y/N]: " recreate
    if [[ "$recreate" =~ ^[Yy]$ ]]; then
        echo "  Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
        echo "  Creating fresh environment..."
        conda create -n "$ENV_NAME" python=3.10 -y
    fi
else
    echo "  Creating conda environment with Python 3.10..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "  Python: $(python --version)"
echo "  Location: $(which python)"
echo ""

# --- Step 4a: Install system-level dependencies via conda ---
# ffmpeg is needed if any transitive dependency (e.g. av) requires it
echo "  [4a] Installing system dependencies via conda..."
conda install -y -c conda-forge ffmpeg pkg-config

# --- Step 4b: Install PyTorch FIRST (flash-attn needs it at build time) ---
echo ""
echo "  [4b] Installing PyTorch with CUDA 12.6..."
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "  Verifying torch + CUDA..."
python -c "import torch; print(f'    torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
echo ""

# --- Step 4c: Install all other requirements ---
echo "  [4c] Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo ""

# --- Step 4d: Install flash-attn LAST (compiles CUDA kernels against torch) ---
echo "  [4d] Installing flash-attn (building from source, may take 10-30 min)..."
echo "       Loading compiler modules for CUDA kernel compilation..."

# Load newer GCC (>= 9) and CUDA toolkit — required for nvcc host compilation
# Adjust module names if your HPC uses different names (check: module avail gcc)
module load gcc cuda 2>/dev/null || {
    echo "  WARNING: 'module load gcc cuda' failed."
    echo "  Please load a GCC >= 9 and CUDA module manually before running this script."
    echo "  Example: module load gcc/11.4.0 cuda/12.4.1"
    echo "  Check available modules with: module avail gcc"
    exit 1
}
echo "  GCC: $(gcc --version | head -1)"
echo "  nvcc: $(nvcc --version | tail -1)"

FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
echo ""

# =============================================================
# Step 5: Verify the full installation
# =============================================================
echo "[Step 5/5] Verifying installation..."
echo ""

python -c "
import sys
print(f'  Python:         {sys.version.split()[0]}')

import torch
print(f'  PyTorch:        {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version:   {torch.version.cuda}')
print(f'  GPU count:      {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')

try:
    import flash_attn
    print(f'  flash-attn:     {flash_attn.__version__}')
except ImportError:
    print('  flash-attn:     NOT INSTALLED (may need manual install)')

import vllm;           print(f'  vLLM:           {vllm.__version__}')
import ray;            print(f'  Ray:            {ray.__version__}')
import transformers;   print(f'  Transformers:   {transformers.__version__}')
import accelerate;     print(f'  Accelerate:     {accelerate.__version__}')
import wandb;          print(f'  WandB:          {wandb.__version__}')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  To activate the environment:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Before running training, set these env vars"
echo "  (add to your .bashrc or SLURM script):"
echo ""
echo "    export STORAGE_PATH=\"$STORAGE_PATH\""
echo "    export HUGGINGFACENAME=\"$HUGGINGFACENAME\""
echo ""
echo "  Then run training:"
echo "    bash scripts/main.sh Qwen/Qwen2.5-3B qwen2.5-3b"
echo ""
echo "  Or via SLURM:"
echo "    sbatch scripts/train.slurm"
echo "============================================"
