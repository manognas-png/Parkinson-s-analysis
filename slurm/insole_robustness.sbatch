#!/bin/bash
# Step 10b: insole-only classifier robustness: age covariate and age-only baseline,
# no body-weight normalization and body-weight-only baseline, age-matched subsets.
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/insole_robustness.sbatch

#SBATCH --job-name=insole_robustness
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

python -m src.run_insole_robustness
