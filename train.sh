#!/bin/bash
#SBATCH --job-name=resnet_lddmm
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# ResNetLDDMM Training Script for SLURM/HPC
# Usage: sbatch train.sh <config_path> [--set KEY.PATH=VALUE ...]
# Example: sbatch train.sh configs/rabbit_pair_poc.yaml --set training.num_steps=5000

set -e

# Paths (update these to match your cluster setup)
CONDA_PATH="/gpfs/home/vbayer/miniconda3"
CONDA_ENV="/gpfs/home/vbayer/environments/geometric_env"
REPO_DIR="/gpfs/home/vbayer/ResNetLDDMM"

cd "$REPO_DIR"
mkdir -p logs

# Activate conda environment
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Add repo to Python path so config module is found
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Log environment
echo "========== SLURM JOB INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --list-gpus 2>/dev/null || echo 'N/A')"
echo "Working dir: $REPO_DIR"
echo "Python: $(which python)"
echo "===================================="

# Parse arguments
CONFIG=${1:-configs/rabbit_pair_poc.yaml}
shift 2>/dev/null || true
EXTRA_ARGS="$@"

# Validate config
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

echo "Config: $CONFIG"
[[ -n "$EXTRA_ARGS" ]] && echo "Overrides: $EXTRA_ARGS"

# Run training
python -m src.resnet_lddmm.cli "$CONFIG" $EXTRA_ARGS

echo "Job complete at $(date)"
