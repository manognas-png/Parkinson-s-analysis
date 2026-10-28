#!/bin/bash
# Step 10c: per-study paired CIs (insole vs measured speed) and held-out permutation
# importance of the six insole features (BW-normalized and raw insole).
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/insole_explain.sbatch

#SBATCH --job-name=insole_explain
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
export PYTHONUNBUFFERED=1  # write progress to the log immediately

python -m src.run_insole_explain
