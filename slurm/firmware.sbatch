#!/bin/bash
# Firmware: train and export the insole-only model with the device's exact steps, then check
# the C code (double and float builds) against the Python reference on all recordings.
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/firmware.sbatch

#SBATCH --job-name=firmware
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "===== 1. train and export the model ====="
python firmware/export_model.py
echo "===== 2. C firmware vs Python reference ====="
python firmware/validate_firmware.py
