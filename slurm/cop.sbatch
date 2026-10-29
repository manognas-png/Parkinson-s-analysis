#!/bin/bash
# Step 11: stance-indexed AP-COP features (after Sifat et al. 2026, arXiv:2609.22956):
# 16-sensor baseline, speed adjustment, fewer sensors and combined models (exploratory).
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/cop.sbatch

#SBATCH --job-name=cop
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python -m src.run_cop
