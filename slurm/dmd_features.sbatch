#!/bin/bash
# Recompute DMD features for all _01 trials, then the speed-adjusted comparison
# and the key-features figure.
# Submit from the project root:  sbatch slurm/dmd_features.sbatch

#SBATCH --job-name=dmd_features
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1

python -m src.plot_dmd
python -m src.speed_adjust
python -m src.plot_key_features
