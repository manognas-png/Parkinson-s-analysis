#!/bin/bash
# Step 9: gait-deviation autoencoder (trained on control strides), baselines,
# speed-adjusted comparison, simulated int8 quantization, microcontroller estimates.
# CPU job: the model is ~3.9k parameters; 16 small trainings in parallel beat a GPU here.
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/autoencoder.sbatch

#SBATCH --job-name=autoencoder
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -m src.run_autoencoder
