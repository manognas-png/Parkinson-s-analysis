#!/bin/bash
# Step 10: insole-only estimates of walking speed and TUG time (TUG = fall-risk PROXY
# only), and an insole-only PD classifier built on them.
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/insole_regression.sbatch

#SBATCH --job-name=insole_regression
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -m src.run_insole_regression
