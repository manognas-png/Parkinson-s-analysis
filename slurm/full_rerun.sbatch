#!/bin/bash
# Final audit: rerun every analysis step from scratch, in order, then check the headline
# numbers and compare every metrics CSV with the committed version (src/check_rerun.py).
#
# It first deletes results/metrics/*.csv (all of them are committed in git, so nothing is
# lost: `git checkout results/metrics` restores them), so no step can read a stale file.
# Figures are overwritten. The firmware export and the demo are NOT rerun here (they would
# overwrite firmware/include/*.h and demo/*; their numbers are checked separately).
#
# Submit from the project root:
#   sbatch --account=bioinfmath540f26_class slurm/full_rerun.sbatch
# Past runtimes add up to about 1.5-2 hours.

#SBATCH --job-name=full_rerun
#SBATCH --account=bioinfmath540f26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

step() {
  echo
  echo "===== $(date +%H:%M:%S) $* ====="
  SECONDS=0
  python "$@"
  echo "----- done in ${SECONDS} s"
}

echo "commit $(git rev-parse --short HEAD); uncommitted changes:"
git status --short
rm -f results/metrics/*.csv

# Steps 3-6: strides, SVD, networks, DMD, speed adjustment
step -m src.plot_explore
step -m src.plot_svd
step -m src.plot_network
step -m src.plot_dmd
step -m src.speed_adjust
step -m src.plot_key_features
# Step 6b and follow-ups: lagged networks
step -m src.run_fiedler_lag
step -m src.run_fiedler_lag --max-lag 0.3 --suffix _lag0.3
step -m src.run_fiedler_lag --max-lag 0.5 --suffix _lag0.5
# Step 7: classical ML
step -m src.run_ml
# Minimum sensor set
step -m src.run_sensor_selection
step -m src.run_supervised_selection
step -m src.run_supervised_selection --exclude-dead-sensors
step -m src.run_exploratory_layout
# Step 9: autoencoder
step -m src.run_autoencoder
# Step 10: insole-only regression and classifier
step -m src.run_insole_regression
step -m src.run_insole_robustness
step -m src.run_insole_explain
step -m src.run_insole_groups
step -m src.run_insole_quantized
# Step 11: COP; Step 12: merged sensors (exploratory)
step -m src.run_cop
step -m src.run_merged_sensors
step -m src.run_merged_pads

# Check: headline numbers and every file vs git HEAD
step -m src.check_rerun
