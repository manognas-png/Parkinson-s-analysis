#!/bin/bash
# Step 7: classical ML (PD vs control) with subject-wise evaluation.
# Submit from the project root:  sbatch slurm/run_ml.sbatch

#SBATCH --job-name=run_ml
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1

python -m src.run_ml
