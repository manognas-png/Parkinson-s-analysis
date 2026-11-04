#!/bin/bash
# Step 10e: insole-only classifier with parameters stored as float64 / float32 / int16.
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/insole_quantized.sbatch

#SBATCH --job-name=insole_quantized
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python -m src.run_insole_quantized
