#!/bin/bash
# Timing for Job B: run ONE greedy outer fold, one fixed-set evaluation and a
# bootstrap sample, then print the projected runtime of the full Job B.
# Submit from the project root:  sbatch slurm/time_one_fold.sbatch

#SBATCH --job-name=time_one_fold
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1

python -m src.run_supervised_selection --time-one-fold
