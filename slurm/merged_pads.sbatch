#!/bin/bash
# Step 12b (EXPLORATORY): merged-sensor classifier with realistic pad models
# (adjacent 50/50, non-overlapping, stacked layers); strides detected from S1 + S2.
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/merged_pads.sbatch

#SBATCH --job-name=merged_pads
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python -m src.run_merged_pads
