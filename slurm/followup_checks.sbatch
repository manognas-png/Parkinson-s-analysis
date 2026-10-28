#!/bin/bash
# Follow-up checks, run one after the other:
#   1. Step 6b with the lag window at +/- 0.3 s again (adds the window-edge count)
#   2. Step 6b with the lag window widened to +/- 0.5 s
#   3. Sensor-5 check: Job B without the 8 trials that have a near-dead sensor
# Submit from the project root:  sbatch slurm/followup_checks.sbatch

#SBATCH --job-name=followup_checks
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1

echo "===== 1. Step 6b, lag window +/- 0.3 s ====="
python -m src.run_fiedler_lag --max-lag 0.3 --suffix _lag0.3

echo "===== 2. Step 6b, lag window +/- 0.5 s ====="
python -m src.run_fiedler_lag --max-lag 0.5 --suffix _lag0.5

echo "===== 3. Sensor-5 check: supervised selection without near-dead-sensor trials ====="
python -m src.run_supervised_selection --exclude-dead-sensors
