#!/bin/bash
# Demo phase 2: group medians, group stance curves, pair selection and 10-s windows for
# demo/gait_demo.html (writes demo/demo_data.json and demo/demo_subject_metrics.csv).
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/demo_data.sbatch

#SBATCH --job-name=demo_data
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python demo/build_demo_data.py
