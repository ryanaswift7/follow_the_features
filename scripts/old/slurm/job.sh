#!/bin/bash

#SBATCH --account=xiangren_1854
#SBATCH --job-name=llama_662
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-task=a40:2
#SBATCH --time=48:00:00

export SLURM_TMPDIR='/scratch1/ryanswif'
export UV_CACHE_DIR="$SLURM_TMPDIR/uv-cache"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"

echo "Building environment in $SLURM_TMPDIR..."
uv sync

echo "Starting experiment..."
uv run ftf -c configs/llama_662_slurm.yaml
