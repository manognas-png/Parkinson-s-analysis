#!/bin/bash
# EXPLORATORY (post hoc): heel + arch + medial forefoot (sensors 1, 5, 7 per foot)
# vs anatomical (1, 4, 7) vs all 16 sensors, with and without near-dead-sensor trials.
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/exploratory_layout.sbatch

#SBATCH --job-name=exploratory_layout
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

python -m src.run_exploratory_layout
