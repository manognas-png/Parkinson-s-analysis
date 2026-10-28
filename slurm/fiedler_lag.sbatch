#!/bin/bash
# Step 6b: per-foot, per-stride lagged cross-correlation networks.
# Submit from the project root:  sbatch slurm/fiedler_lag.sbatch

#SBATCH --job-name=fiedler_lag
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1

python -m src.run_fiedler_lag
