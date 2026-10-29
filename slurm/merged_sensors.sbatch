#!/bin/bash
# Step 12 (EXPLORATORY): merged large sensors S1 = 1 + 4, S2 = 4 + 7 per foot with linear
# recovery (A x = b): sparsity, recovery error, PD heel signal, insole-only classifier.
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/merged_sensors.sbatch

#SBATCH --job-name=merged_sensors
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

python -m src.run_merged_sensors
