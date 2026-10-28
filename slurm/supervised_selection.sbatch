#!/bin/bash
# Job B: supervised forward selection (nested subject-wise CV), 100 random
# 6-sensor sets, bootstrap CIs, success criterion.
# Submit from the project root:  sbatch slurm/supervised_selection.sbatch

#SBATCH --job-name=supervised_selection
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1

python -m src.run_supervised_selection
